from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv

from .actions import ACTION_MACROS
from .rl2_model import DYNAMICS_TARGETS, RL2TransformerActorCritic, rl2_features
from .trial_context import EPISODE_SUMMARY_DIM, TrialContextTracker


@dataclass
class TrialBatch:
    features: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    episode_ends: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor
    episode_returns: np.ndarray
    episode_entropies: np.ndarray
    total_steps: int


def episode_objective_weights(batch: TrialBatch, final_episode_weight: float) -> torch.Tensor:
    """Return a linear per-transition curriculum from episode one to the last."""
    if final_episode_weight < 1.0:
        raise ValueError("final episode weight must be at least one")
    episode_index = torch.cumsum(batch.episode_ends.to(torch.long), dim=0)
    episode_index = episode_index - batch.episode_ends.to(torch.long)
    episode_count = int(episode_index[batch.valid].max()) + 1
    if episode_count <= 1:
        return torch.ones_like(batch.advantages)
    progress = episode_index.to(torch.float32) / float(episode_count - 1)
    weights = 1.0 + progress * (float(final_episode_weight) - 1.0)
    return torch.where(batch.valid, weights, torch.zeros_like(weights))


def collect_trial_batch(
    env: MarsRoverVecEnv,
    model: RL2TransformerActorCritic,
    *,
    device: torch.device,
    seed: int,
    gamma: float,
    gae_lambda: float,
) -> TrialBatch:
    num_envs = env.num_envs
    episodes_per_trial = env.episodes_per_trial
    maximum_length = env.max_steps * episodes_per_trial
    if env.max_steps <= 0 or episodes_per_trial < 2:
        raise ValueError("RL2 requires a finite horizon and at least two episodes per trial")
    observations = env.reset(seed).copy()
    state = model.initial_state(num_envs, device)
    context_dim = int(getattr(model, "context_dim", 0))
    context: TrialContextTracker | None = None
    if context_dim:
        history, remainder = divmod(context_dim, EPISODE_SUMMARY_DIM + 1)
        if remainder:
            raise ValueError(f"context_dim {context_dim} is not a whole number of summaries")
        context = TrialContextTracker(num_envs, history=history, max_steps=env.max_steps)
    previous_actions = np.full(num_envs, -1, dtype=np.int64)
    previous_rewards = np.zeros(num_envs, dtype=np.float32)
    previous_dones = np.ones(num_envs, dtype=bool)
    active = np.ones(num_envs, dtype=bool)
    episode_indices = np.zeros(num_envs, dtype=np.int32)
    lengths = np.zeros(num_envs, dtype=np.int32)
    episode_returns = np.zeros((num_envs, episodes_per_trial), dtype=np.float64)
    episode_entropy_sums = np.zeros_like(episode_returns)
    episode_step_counts = np.zeros_like(episode_returns)
    rng = np.random.default_rng(seed ^ 0xA5A55A5A)

    feature_dim = model.feature_dim
    features_store = np.zeros((maximum_length, num_envs, feature_dim), dtype=np.float32)
    actions_store = np.zeros((maximum_length, num_envs), dtype=np.int64)
    log_prob_store = np.zeros((maximum_length, num_envs), dtype=np.float32)
    value_store = np.zeros((maximum_length, num_envs), dtype=np.float32)
    reward_store = np.zeros((maximum_length, num_envs), dtype=np.float32)
    valid_store = np.zeros((maximum_length, num_envs), dtype=bool)
    episode_end_store = np.zeros((maximum_length, num_envs), dtype=bool)
    action_macros = np.asarray(ACTION_MACROS, dtype=np.int32)

    model.eval()
    time_index = 0
    with torch.no_grad():
        while bool(active.any()):
            if time_index >= maximum_length:
                raise RuntimeError("trial collector exceeded the configured complete-trial horizon")
            inactive = ~active
            if bool(inactive.any()):
                # Finished native slots still receive dummy batch steps while
                # longer trials complete. Never allow their post-terminal state
                # (which may become non-finite) into the shared Transformer.
                observations[inactive] = 0.0
                previous_actions[inactive] = -1
                previous_rewards[inactive] = 0.0
                previous_dones[inactive] = True
                state = state.reset_rows(
                    torch.as_tensor(inactive, dtype=torch.bool, device=device)
                )
            features = rl2_features(
                observations,
                previous_actions,
                previous_rewards,
                previous_dones,
                device=device,
                trial_context=context.features() if context is not None else None,
            )
            step_observations = observations.copy()
            logits, values, state = model.forward_step(features, state)
            active_device = torch.as_tensor(active, dtype=torch.bool, device=device)
            if not bool(torch.isfinite(logits[active_device]).all()):
                raise FloatingPointError("non-finite RL2 logits in an active trial")
            distribution = torch.distributions.Categorical(logits=logits)
            action_indices = distribution.sample()
            log_probabilities = distribution.log_prob(action_indices)
            entropies = distribution.entropy()
            action_numpy = action_indices.detach().cpu().numpy()
            active_now = active.copy()

            features_store[time_index, active_now] = features.detach().cpu().numpy()[active_now]
            actions_store[time_index, active_now] = action_numpy[active_now]
            log_prob_store[time_index, active_now] = log_probabilities.detach().cpu().numpy()[active_now]
            value_store[time_index, active_now] = values.detach().cpu().numpy()[active_now]
            valid_store[time_index, active_now] = True
            lengths[active_now] += 1

            raw_actions = action_macros[action_numpy]
            observations_view, rewards, terminated, truncated, _ = env.step_uint8(raw_actions)
            observations = observations_view.copy()
            dones = np.logical_or(terminated != 0, truncated != 0)
            reward_store[time_index, active_now] = rewards[active_now]
            episode_end_store[time_index, active_now] = dones[active_now]
            if context is not None:
                # Summarise the state the reward was earned from, before any reset
                # below overwrites `observations` for finished episodes.
                context.observe(step_observations, rewards, dones, active_now)
            for env_index in np.flatnonzero(active_now):
                episode = int(episode_indices[env_index])
                episode_returns[env_index, episode] += float(rewards[env_index])
                episode_entropy_sums[env_index, episode] += float(entropies[env_index].cpu())
                episode_step_counts[env_index, episode] += 1

            for env_index in np.flatnonzero(dones & active_now):
                episode_indices[env_index] += 1
                trial_end = episode_indices[env_index] >= episodes_per_trial
                reset_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                if trial_end:
                    active[env_index] = False
                    # Keep the native slot in a valid reset state while slower trials finish.
                    env.reset_at(env_index, reset_seed, trial_start=True)
                else:
                    env.reset_at(env_index, reset_seed, trial_start=False)
                observations[env_index] = env.obs[env_index]

            previous_actions[active_now] = action_numpy[active_now]
            previous_rewards[active_now] = rewards[active_now]
            previous_dones[active_now] = dones[active_now]
            time_index += 1

    advantages = np.zeros_like(reward_store[:time_index])
    returns = np.zeros_like(reward_store[:time_index])
    for env_index, length in enumerate(lengths):
        gae = 0.0
        for step in range(int(length) - 1, -1, -1):
            trial_continues = step + 1 < int(length)
            next_value = value_store[step + 1, env_index] if trial_continues else 0.0
            delta = (
                reward_store[step, env_index]
                + gamma * float(trial_continues) * next_value
                - value_store[step, env_index]
            )
            gae = delta + gamma * gae_lambda * float(trial_continues) * gae
            advantages[step, env_index] = gae
            returns[step, env_index] = gae + value_store[step, env_index]

    safe_counts = np.maximum(episode_step_counts, 1.0)
    episode_entropies = episode_entropy_sums / safe_counts
    valid_slice = valid_store[:time_index]
    return TrialBatch(
        features=torch.from_numpy(features_store[:time_index]),
        actions=torch.from_numpy(actions_store[:time_index]),
        old_log_probabilities=torch.from_numpy(log_prob_store[:time_index]),
        old_values=torch.from_numpy(value_store[:time_index]),
        rewards=torch.from_numpy(reward_store[:time_index]),
        episode_ends=torch.from_numpy(episode_end_store[:time_index]),
        advantages=torch.from_numpy(advantages),
        returns=torch.from_numpy(returns),
        valid=torch.from_numpy(valid_slice),
        lengths=torch.from_numpy(lengths),
        episode_returns=episode_returns,
        episode_entropies=episode_entropies,
        total_steps=int(valid_slice.sum()),
    )


def ppo_update(
    model: RL2TransformerActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TrialBatch,
    *,
    device: torch.device,
    epochs: int,
    trial_minibatch_size: int,
    chunk_length: int,
    clip_range: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    target_kl: float,
    final_episode_weight: float = 1.0,
    rng: np.random.Generator,
    reference_model: RL2TransformerActorCritic | None = None,
    kl_anchor_coefficient: float = 0.0,
    freeze_policy: bool = False,
    dynamics_coefficient: float = 0.0,
    observation_dim: int = 0,
    belief_kl_coefficient: float = 0.0,
) -> dict[str, float]:
    """`reference_model` is a frozen post-distillation snapshot used as a trust-region
    anchor (KL(current || reference) penalty), not a training target: it stops PPO from
    drifting arbitrarily far from working, distilled behavior in a single noisy update.
    `freeze_policy` zeroes the policy/entropy gradient (value head still trains) for a
    warmup window so the critic has usable estimates before it starts shaping the actor.
    Neither mechanism creates pressure to *use* memory across episodes on its own — they
    only protect distilled (memoryless) competence from collapsing.
    """
    model.train()
    if reference_model is not None:
        reference_model.eval()
    valid = batch.valid
    objective_weights = episode_objective_weights(batch, final_episode_weight)
    normalized_advantages = batch.advantages.clone() * objective_weights
    valid_advantages = normalized_advantages[valid]
    normalized_advantages[valid] = (
        valid_advantages - valid_advantages.mean()
    ) / (valid_advantages.std(unbiased=False) + 1.0e-8)
    num_trials = batch.features.shape[1]
    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "kl_anchor": 0.0,
        "dynamics_loss": 0.0,
        "belief_kl": 0.0,
    }
    metric_weight = 0.0
    grad_norms: list[float] = []
    early_stopped = False

    for _epoch in range(epochs):
        order = rng.permutation(num_trials)
        epoch_kl_sum = 0.0
        epoch_weight = 0.0
        for begin in range(0, num_trials, trial_minibatch_size):
            indices_numpy = order[begin : begin + trial_minibatch_size]
            indices = torch.as_tensor(indices_numpy, dtype=torch.long)
            maximum_length = int(batch.lengths[indices].max())
            state = model.initial_state(len(indices_numpy), device)
            reference_state = (
                reference_model.initial_state(len(indices_numpy), device)
                if reference_model is not None
                else None
            )
            total_valid = float(batch.valid[:maximum_length, indices].sum())
            if total_valid <= 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, maximum_length, chunk_length):
                stop = min(maximum_length, start + chunk_length)
                chunk_valid = batch.valid[start:stop, indices].transpose(0, 1).to(device)
                chunk_weight = float(chunk_valid.sum())
                if chunk_weight <= 0:
                    continue
                chunk_features = batch.features[start:stop, indices].transpose(0, 1).to(device)
                want_dynamics = dynamics_coefficient > 0.0 and observation_dim > 0
                want_belief = belief_kl_coefficient > 0.0 and int(getattr(model, "belief_dim", 0)) > 0
                belief_stats = None
                if want_dynamics or want_belief:
                    logits, values, state, hidden, belief_stats = model.forward_chunk(
                        chunk_features, state, detach_memory=True, return_hidden=True
                    )
                else:
                    logits, values, state = model.forward_chunk(
                        chunk_features, state, detach_memory=True
                    )
                distribution = torch.distributions.Categorical(logits=logits)
                actions = batch.actions[start:stop, indices].transpose(0, 1).to(device)
                new_log_probabilities = distribution.log_prob(actions)
                old_log_probabilities = (
                    batch.old_log_probabilities[start:stop, indices].transpose(0, 1).to(device)
                )
                advantages = normalized_advantages[start:stop, indices].transpose(0, 1).to(device)
                returns = batch.returns[start:stop, indices].transpose(0, 1).to(device)
                log_ratio = new_log_probabilities - old_log_probabilities
                ratio = torch.exp(log_ratio)
                policy_unclipped = -advantages * ratio
                policy_clipped = -advantages * torch.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = torch.maximum(policy_unclipped, policy_clipped)[chunk_valid].mean()
                value_loss = 0.5 * ((values - returns) ** 2)[chunk_valid].mean()
                entropy = distribution.entropy()[chunk_valid].mean()
                policy_term = policy_loss - entropy_coefficient * entropy
                if freeze_policy:
                    policy_term = policy_term.detach()
                kl_anchor = torch.zeros((), device=device)
                if reference_model is not None and kl_anchor_coefficient > 0.0:
                    with torch.no_grad():
                        reference_logits, _, reference_state = reference_model.forward_chunk(
                            chunk_features, reference_state, detach_memory=True
                        )
                    reference_distribution = torch.distributions.Categorical(
                        logits=reference_logits
                    )
                    kl_anchor = torch.distributions.kl.kl_divergence(
                        distribution, reference_distribution
                    )[chunk_valid].mean()
                dynamics_loss = torch.zeros((), device=device)
                if want_dynamics and stop - start > 1:
                    # Targets are the NEXT step's physics components, so the last row of
                    # the chunk has no target and the pair must stay inside one episode.
                    current = chunk_features[:, :-1, :observation_dim]
                    following = chunk_features[:, 1:, :observation_dim]
                    targets = (
                        following[..., list(DYNAMICS_TARGETS)]
                        - current[..., list(DYNAMICS_TARGETS)]
                    )
                    predicted = model.predict_dynamics(hidden[:, :-1], actions[:, :-1])
                    pair_valid = chunk_valid[:, :-1] & chunk_valid[:, 1:]
                    chunk_ends = (
                        batch.episode_ends[start : stop - 1, indices].transpose(0, 1).to(device)
                    )
                    pair_valid = pair_valid & ~chunk_ends
                    if bool(pair_valid.any()):
                        dynamics_loss = (
                            torch.nn.functional.smooth_l1_loss(
                                predicted, targets, reduction="none"
                            ).mean(dim=-1)[pair_valid]
                        ).mean()
                belief_kl = torch.zeros((), device=device)
                if want_belief and belief_stats is not None:
                    # Standard VariBAD regulariser: keep the posterior over "which world"
                    # close to the prior unless the history genuinely says otherwise.
                    mean, log_variance = belief_stats.chunk(2, dim=-1)
                    per_step = 0.5 * (
                        mean.pow(2) + log_variance.exp() - 1.0 - log_variance
                    ).sum(dim=-1)
                    belief_kl = per_step[chunk_valid].mean()
                loss = (
                    policy_term
                    + value_coefficient * value_loss
                    + kl_anchor_coefficient * kl_anchor
                    + dynamics_coefficient * dynamics_loss
                    + belief_kl_coefficient * belief_kl
                ) * (chunk_weight / total_valid)
                loss.backward()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio)[chunk_valid].mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > clip_range).float()[chunk_valid].mean()
                    )
                values_to_log = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "kl_anchor": kl_anchor,
                    "dynamics_loss": dynamics_loss,
                    "belief_kl": belief_kl,
                }
                for name, value in values_to_log.items():
                    metric_sums[name] += float(value.detach().cpu()) * chunk_weight
                metric_weight += chunk_weight
                epoch_kl_sum += float(approx_kl.detach().cpu()) * chunk_weight
                epoch_weight += chunk_weight
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()
        if epoch_weight > 0 and epoch_kl_sum / epoch_weight > target_kl:
            early_stopped = True
            break

    returns_valid = batch.returns[valid].numpy()
    values_valid = batch.old_values[valid].numpy()
    variance = float(np.var(returns_valid))
    explained_variance = (
        float(1.0 - np.var(returns_valid - values_valid) / variance) if variance > 1.0e-8 else 0.0
    )
    denominator = max(metric_weight, 1.0)
    return {
        **{name: value / denominator for name, value in metric_sums.items()},
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "explained_variance": explained_variance,
        "early_stop_kl": float(early_stopped),
        "final_episode_weight": float(final_episode_weight),
    }
