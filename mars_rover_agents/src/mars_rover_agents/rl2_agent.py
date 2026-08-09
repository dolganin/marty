from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import macro_action
from .agent import Agent
from .rl2_model import RL2TransformerActorCritic, TransformerXLState, rl2_features
from .trial_context import EPISODE_SUMMARY_DIM, TrialContextTracker


class RL2TransformerAgent(Agent):
    def __init__(
        self,
        model: RL2TransformerActorCritic,
        *,
        device: torch.device,
        deterministic: bool = True,
        max_steps: int = 800,
    ) -> None:
        self.model = model
        self.device = device
        self.deterministic = bool(deterministic)
        self.state: TransformerXLState | None = None
        self.previous_action = -1
        self.previous_reward = 0.0
        self.previous_done = True
        self.last_entropy: float | None = None
        self.last_observation: np.ndarray | None = None
        context_dim = int(getattr(model, "context_dim", 0))
        if context_dim:
            history, remainder = divmod(context_dim, EPISODE_SUMMARY_DIM + 1)
            if remainder:
                raise ValueError(f"context_dim {context_dim} is not a whole number of summaries")
            self.context = TrialContextTracker(1, history=history, max_steps=max_steps)
        else:
            self.context = None

    def reset(self, trial_start: bool) -> None:
        if trial_start or self.state is None:
            self.state = self.model.initial_state(1, self.device)
            self.previous_action = -1
            self.previous_reward = 0.0
            self.previous_done = True
            if self.context is not None:
                self.context.reset_trial(np.ones(1, dtype=bool))
        # reset(False) deliberately keeps sequence state, previous feedback and the
        # accumulated trial context: that carry-over is the whole adaptation channel.

    def act(self, obs: np.ndarray) -> int:
        if self.state is None:
            raise RuntimeError("agent.reset(trial_start=True) must precede act")
        observation = np.asarray(obs, dtype=np.float32)
        self.last_observation = observation
        features = rl2_features(
            observation[None, :],
            np.asarray([self.previous_action]),
            np.asarray([self.previous_reward]),
            np.asarray([self.previous_done]),
            device=self.device,
            trial_context=self.context.features() if self.context is not None else None,
        )
        with torch.no_grad():
            logits, _, self.state = self.model.forward_step(features, self.state)
            distribution = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if self.deterministic else distribution.sample()
            self.last_entropy = float(distribution.entropy().detach().cpu())
        self.previous_action = int(action.item())
        return macro_action(self.previous_action)

    def observe(self, reward: float, done: bool, info: dict[str, Any]) -> None:
        del info
        self.previous_reward = float(reward)
        self.previous_done = bool(done)
        if self.context is not None and self.last_observation is not None:
            # The observation passed to act() is the state this reward was earned from,
            # which is what the episode summary should describe.
            self.context.observe(
                self.last_observation[None, :],
                np.asarray([self.previous_reward]),
                np.asarray([self.previous_done]),
            )

    def diagnostics(self) -> dict[str, float]:
        return {"action_entropy": self.last_entropy} if self.last_entropy is not None else {}

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        max_steps: int = 800,
    ) -> "RL2TransformerAgent":
        device = torch.device(device)
        payload = torch.load(Path(artifact_path), map_location=device, weights_only=True)
        model = RL2TransformerActorCritic(**payload["model_config"]).to(device)
        model.load_compatible_state(payload["model_state"])
        model.eval()
        return cls(model, device=device, max_steps=max_steps)

