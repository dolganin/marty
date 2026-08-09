from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .actions import ACTION_MACROS


# Observation components whose evolution is governed by the biome's physics: forward and
# vertical velocity, body angle, angular velocity, and energy. Predicting these one step
# ahead is a compact proxy for "identify the mechanic you are driving on".
DYNAMICS_TARGETS = (2, 3, 4, 5, 6)


def rl2_features(
    observations: np.ndarray | torch.Tensor,
    previous_actions: np.ndarray | torch.Tensor,
    previous_rewards: np.ndarray | torch.Tensor,
    previous_dones: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    trial_context: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Canonical RL2 input, optionally widened by an explicit trial-context summary.

    `trial_context` carries compressed statistics of earlier episodes in the trial
    (see trial_context.py). It is appended last so a model trained without it keeps
    an identical feature layout.
    """
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(previous_actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(previous_rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(previous_dones, dtype=torch.float32, device=device)
    one_hot = torch.nn.functional.one_hot(
        actions.clamp(min=0), num_classes=len(ACTION_MACROS)
    ).float()
    one_hot = one_hot * (actions >= 0).unsqueeze(-1)
    reward_feature = torch.clamp(rewards, -10.0, 10.0).unsqueeze(-1) / 10.0
    parts = [obs, one_hot, reward_feature, dones.unsqueeze(-1)]
    if trial_context is not None:
        parts.append(torch.as_tensor(trial_context, dtype=torch.float32, device=device))
    return torch.cat(parts, dim=-1)


@dataclass
class TransformerXLState:
    memories: list[torch.Tensor]
    valid: torch.Tensor

    def detach(self) -> "TransformerXLState":
        return TransformerXLState(
            memories=[memory.detach() for memory in self.memories],
            valid=self.valid,
        )

    def reset_rows(self, reset_mask: torch.Tensor) -> "TransformerXLState":
        if not bool(reset_mask.any()):
            return self
        keep = (~reset_mask).to(dtype=self.memories[0].dtype).view(-1, 1, 1)
        return TransformerXLState(
            memories=[memory * keep for memory in self.memories],
            valid=self.valid & (~reset_mask).unsqueeze(-1),
        )


class RelativeCausalAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int, max_distance: int):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.output = nn.Linear(model_dim, model_dim)
        self.relative_bias = nn.Parameter(torch.zeros(heads, max_distance + 1))
        nn.init.normal_(self.relative_bias, std=0.01)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = queries.shape
        memory_length = memory.shape[1]
        keys_input = torch.cat((memory, queries), dim=1)
        query_projection = self.qkv(queries)[..., : self.model_dim]
        key_value_projection = self.qkv(keys_input)
        key_projection = key_value_projection[..., self.model_dim : self.model_dim * 2]
        value_projection = key_value_projection[..., self.model_dim * 2 :]

        def split(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, value.shape[1], self.heads, self.head_dim).transpose(1, 2)

        q = split(query_projection)
        k = split(key_projection)
        v = split(value_projection)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        query_positions = torch.arange(length, device=queries.device) + memory_length
        key_positions = torch.arange(memory_length + length, device=queries.device)
        raw_distances = query_positions[:, None] - key_positions[None, :]
        distances = raw_distances.clamp(
            min=0, max=self.max_distance
        )
        bias = self.relative_bias[:, distances].unsqueeze(0)
        scores = scores + bias

        current_valid = torch.ones((batch, length), dtype=torch.bool, device=queries.device)
        key_valid = torch.cat((memory_valid, current_valid), dim=1)
        causal = (
            (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
            & (raw_distances <= memory_length)
        )
        allowed = key_valid[:, None, None, :] & causal[None, None, :, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.model_dim)
        return self.output(attended)


class TransformerXLBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, ff_dim: int, max_distance: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = RelativeCausalAttention(model_dim, heads, max_distance)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(
        self, inputs: torch.Tensor, memory: torch.Tensor, memory_valid: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        outputs = inputs + self.attention(normalized, memory, memory_valid)
        return outputs + self.feed_forward(self.ff_norm(outputs))


class RL2TransformerActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        *,
        model_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        memory_len: int = 256,
        context_dim: int = 0,
        belief_dim: int = 0,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = len(ACTION_MACROS)
        self.context_dim = int(context_dim)
        self.feature_dim = self.observation_dim + self.action_dim + 2 + self.context_dim
        self.model_dim = int(model_dim)
        self.memory_len = int(memory_len)
        self.layers = int(layers)
        self.input_projection = nn.Sequential(
            nn.Linear(self.feature_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Tanh(),
        )
        self.blocks = nn.ModuleList(
            [TransformerXLBlock(model_dim, heads, ff_dim, memory_len * 2) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        # VariBAD: an explicit posterior over "which world am I in", inferred from the
        # trial history. belief_dim=0 keeps the pure RL2 architecture (implicit memory only),
        # so the two can be compared with everything else held constant.
        self.belief_dim = int(belief_dim)
        head_dim = model_dim + self.belief_dim
        if self.belief_dim:
            self.belief = nn.Linear(model_dim, self.belief_dim * 2)
        self.policy = nn.Linear(head_dim, self.action_dim)
        self.value = nn.Linear(head_dim, 1)
        # Self-supervised head: given the current representation and the action taken,
        # predict how the rover's dynamics respond. Getting this right REQUIRES encoding
        # the mechanic, which is the pressure the reward signal fails to supply. It is
        # trained from ordinary observations, so it transfers to held-out biomes.
        self.dynamics = nn.Linear(head_dim + self.action_dim, len(DYNAMICS_TARGETS))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> TransformerXLState:
        return TransformerXLState(
            memories=[
                torch.zeros(batch_size, self.memory_len, self.model_dim, device=device)
                for _ in range(self.layers)
            ],
            valid=torch.zeros(batch_size, self.memory_len, dtype=torch.bool, device=device),
        )

    def forward_chunk(
        self,
        features: torch.Tensor,
        state: TransformerXLState,
        *,
        reset_mask: torch.Tensor | None = None,
        detach_memory: bool = True,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature_dim]")
        if reset_mask is not None:
            state = state.reset_rows(reset_mask)
        hidden = self.input_projection(features)
        new_memories: list[torch.Tensor] = []
        new_valid = torch.cat(
            (
                state.valid,
                torch.ones(
                    features.shape[0], features.shape[1], dtype=torch.bool, device=features.device
                ),
            ),
            dim=1,
        )[:, -self.memory_len :]
        for layer, block in enumerate(self.blocks):
            layer_inputs = hidden
            hidden = block(layer_inputs, state.memories[layer], state.valid)
            # Transformer-XL caches the past inputs to this layer. Caching its
            # outputs would make one-step inference differ from causal replay.
            memory_inputs = block.attention_norm(layer_inputs)
            updated = torch.cat((state.memories[layer], memory_inputs), dim=1)[
                :, -self.memory_len :
            ]
            new_memories.append(updated.detach() if detach_memory else updated)
        hidden = self.final_norm(hidden)
        belief_stats: torch.Tensor | None = None
        if self.belief_dim:
            belief_stats = self.belief(hidden)
            mean, log_variance = belief_stats.chunk(2, dim=-1)
            log_variance = log_variance.clamp(-8.0, 8.0)
            if self.training:
                # Reparameterisation: sampling keeps the posterior honest during training,
                # while acting uses the mean so evaluation is deterministic.
                sample = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            else:
                sample = mean
            hidden = torch.cat((hidden, sample), dim=-1)
            belief_stats = torch.cat((mean, log_variance), dim=-1)
        logits = self.policy(hidden)
        values = self.value(hidden).squeeze(-1)
        state = TransformerXLState(new_memories, new_valid)
        if return_hidden:
            return logits, values, state, hidden, belief_stats
        return logits, values, state

    def predict_dynamics(self, hidden: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the next-step change in the physics-governed observation components."""
        one_hot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()
        return self.dynamics(torch.cat((hidden, one_hot), dim=-1))

    def forward_step(
        self,
        features: torch.Tensor,
        state: TransformerXLState,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TransformerXLState]:
        logits, values, state = self.forward_chunk(
            features.unsqueeze(1), state, reset_mask=reset_mask, detach_memory=True
        )
        return logits[:, 0], values[:, 0], state

    def load_compatible_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load weights, tolerating checkpoints saved before the dynamics head existed.

        Only the self-supervised head may be absent — it carries no policy behaviour, so a
        freshly initialised one is harmless. Any other missing or unexpected key is a real
        mismatch and still raises.
        """
        missing, unexpected = self.load_state_dict(state, strict=False)
        tolerated = ("dynamics.", "belief.")
        unexplained_missing = [
            key for key in missing if not key.startswith(tolerated)
        ]
        if unexplained_missing or unexpected:
            raise RuntimeError(
                f"incompatible checkpoint: missing={unexplained_missing} unexpected={list(unexpected)}"
            )

    def config(self) -> dict[str, int]:
        first_attention = self.blocks[0].attention
        return {
            "observation_dim": self.observation_dim,
            "model_dim": self.model_dim,
            "heads": first_attention.heads,
            "layers": self.layers,
            "ff_dim": self.blocks[0].feed_forward[0].out_features,
            "memory_len": self.memory_len,
            "context_dim": self.context_dim,
            "belief_dim": self.belief_dim,
        }
