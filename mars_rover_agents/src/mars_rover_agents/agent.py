from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class Agent(ABC):
    """The complete participant-visible interaction contract.

    Implementations receive observations plus reward/done/public info only.
    Environment instances, biome identifiers, debug state, and mechanical
    parameters are deliberately absent from this interface.
    """

    @abstractmethod
    def reset(self, trial_start: bool) -> None:
        """Reset all memory on trial start, otherwise episode-local state only."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Return one public MarsRoverEnv bit-mask action."""

    @abstractmethod
    def observe(self, reward: float, done: bool, info: dict[str, Any]) -> None:
        """Consume the only feedback channel available for within-trial memory."""

    def diagnostics(self) -> dict[str, float]:
        """Optional agent-owned diagnostics; never populated from evaluator state."""
        return {}


AgentFactory = Callable[[int], Agent]

