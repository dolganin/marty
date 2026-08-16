from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv

from .actions import ACTION_MACROS
from .rl2_model import RL2TransformerActorCritic, rl2_features
from .trial_context import EPISODE_SUMMARY_DIM, TrialContextTracker


def distill_from_ppo_teacher(
    model: RL2TransformerActorCritic,
    teacher: object,
    env: MarsRoverVecEnv,
    *,
    device: torch.device,
    transitions: int,
    chunk_length: int,
    learning_rate: float,
    teacher_temperature: float,
    seed: int,
    rollout_temperature: float = 1.0,
    metric_callback: Callable[[dict[str, float], int], None] | None = None,
) -> int:
    """GPU behavior-clone a public-observation PPO into the RL2 backbone.

    Every optimization batch is a causal sequence collected from a fresh set
    of randomized training biomes. The teacher receives exactly the same raw
    observation as the student and never receives a biome id or debug state.

    `teacher_temperature` only sharpens the cross-entropy *loss target*; it does
    not change which (state, action) pairs get visited. `rollout_temperature`
    controls the teacher's own action-sampling distribution while it drives the
    rollout that produces those states in the first place — sampling from the
    teacher's full (comparatively high-entropy) policy compounds per-step
    disagreement into rollouts the teacher's own greedy policy would never
    visit, so the student ends up cloning a noisy realization of the teacher
    rather than its peak behavior. A value near 0 makes rollout collection
    near-deterministic (close to the teacher's argmax) while still allowing a
    little exploration so distillation sees more than one fixed trajectory.
    """
    if transitions < 1 or chunk_length < 1:
        raise ValueError("distillation transitions and chunk length must be positive")
    if not 0.0 < teacher_temperature <= 1.0:
        raise ValueError("teacher temperature must be in (0, 1]")
    if not 0.0 < rollout_temperature <= 1.0:
        raise ValueError("rollout temperature must be in (0, 1]")
    if getattr(teacher, "device", torch.device("cpu")).type != "cuda":
        raise RuntimeError("the distillation teacher must run on CUDA")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1.0e-5)
    action_macros = np.asarray(ACTION_MACROS, dtype=np.int32)
    context_dim = int(getattr(model, "context_dim", 0))
    context: TrialContextTracker | None = None
    if context_dim:
        history, remainder = divmod(context_dim, EPISODE_SUMMARY_DIM + 1)
        if remainder:
            raise ValueError(f"context_dim {context_dim} is not a whole number of summaries")
        context = TrialContextTracker(env.num_envs, history=history, max_steps=env.max_steps)
    completed = 0
    update = 0
    model.train()
                                                                                     
                                                                                        
                                                                                       
                                                                                        
                                          
    observations = env.reset(seed).copy()
    if context is not None:
        context.reset_trial(np.ones(env.num_envs, dtype=bool))
    state = model.initial_state(env.num_envs, device)
    previous_actions = np.full(env.num_envs, -1, dtype=np.int64)
    previous_rewards = np.zeros(env.num_envs, dtype=np.float32)
    previous_dones = np.ones(env.num_envs, dtype=bool)
    while completed < transitions:
        trial_resets = np.zeros(env.num_envs, dtype=bool)
        features_store: list[torch.Tensor] = []
        logits_store: list[torch.Tensor] = []
        values_store: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(chunk_length):
                features_store.append(
                    rl2_features(
                        observations,
                        previous_actions,
                        previous_rewards,
                        previous_dones,
                        device=device,
                        trial_context=context.features() if context is not None else None,
                    )
                )
                step_observations = observations.copy()
                teacher_observations = torch.as_tensor(
                    observations, dtype=torch.float32, device=device
                )
                distribution = teacher.policy.get_distribution(teacher_observations)
                teacher_logits = distribution.distribution.logits
                teacher_values = teacher.policy.predict_values(teacher_observations).flatten()
                logits_store.append(teacher_logits.detach())
                values_store.append(teacher_values.detach())
                rollout_distribution = torch.distributions.Categorical(
                    logits=teacher_logits / rollout_temperature
                )
                action_indices = rollout_distribution.sample()
                action_numpy = action_indices.cpu().numpy()
                observations_view, rewards, terminated, truncated, _ = env.step_uint8(
                    action_macros[action_numpy]
                )
                observations = observations_view.copy()
                dones = np.logical_or(terminated != 0, truncated != 0)
                if context is not None:
                    context.observe(step_observations, rewards, dones)
                for env_index in np.flatnonzero(dones):
                    trial_start = bool(env.next_trial_start[env_index])
                    env.reset_at(
                        int(env_index),
                        seed + update * 1_000_003 + int(env_index),
                        trial_start=trial_start,
                    )
                    observations[env_index] = env.obs[env_index]
                    if trial_start:
                                                                                          
                                                                                      
                                                                                  
                        trial_resets[env_index] = True
                        if context is not None:
                            mask = np.zeros(env.num_envs, dtype=bool)
                            mask[env_index] = True
                            context.reset_trial(mask)
                previous_actions = action_numpy
                previous_rewards = rewards.copy()
                previous_dones = dones

        features = torch.stack(features_store, dim=1)
        target_logits = torch.stack(logits_store, dim=1)
        target_values = torch.stack(values_store, dim=1)
        student_logits, student_values, next_state = model.forward_chunk(
            features, state, detach_memory=False
        )
        target_probabilities = torch.softmax(target_logits / teacher_temperature, dim=-1)
        policy_loss = -(
            target_probabilities * torch.log_softmax(student_logits, dim=-1)
        ).sum(dim=-1).mean()
        value_loss = torch.nn.functional.smooth_l1_loss(student_values, target_values)
        loss = policy_loss + 0.1 * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

                                                                                       
                                                                                 
        state = next_state.detach()
        if trial_resets.any():
            state = state.reset_rows(
                torch.as_tensor(trial_resets, dtype=torch.bool, device=device)
            )

        batch_transitions = env.num_envs * chunk_length
        completed += batch_transitions
        update += 1
        if metric_callback is not None:
            with torch.no_grad():
                agreement = (
                    student_logits.argmax(dim=-1) == target_logits.argmax(dim=-1)
                ).float().mean()
                target_entropy = -(
                    target_probabilities
                    * torch.log(target_probabilities.clamp_min(1.0e-8))
                ).sum(dim=-1).mean()
            metric_callback(
                {
                    "distill.loss": float(loss.detach().cpu()),
                    "distill.policy_loss": float(policy_loss.detach().cpu()),
                    "distill.value_loss": float(value_loss.detach().cpu()),
                    "distill.grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "distill.argmax_agreement": float(agreement.cpu()),
                    "distill.teacher_entropy": float(target_entropy.cpu()),
                    "distill.teacher_temperature": float(teacher_temperature),
                },
                completed,
            )
    model.eval()
    return completed
