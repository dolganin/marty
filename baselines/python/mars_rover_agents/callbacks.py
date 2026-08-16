from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .agents import PPOAgent
from .harness import EvaluationSpec, evaluate
from .tracking import MLflowRun


class GradientNormTracker:
    """Measure the post-clipping gradient norm immediately before optimizer steps."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.values: list[float] = []
        self._original_step = optimizer.step

        def tracked_step(*args: Any, **kwargs: Any):
            squares = []
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        squares.append(torch.sum(parameter.grad.detach() ** 2))
            if squares:
                norm = torch.sqrt(torch.stack(squares).sum())
                self.values.append(float(norm.detach().cpu()))
            return self._original_step(*args, **kwargs)

        optimizer.step = tracked_step                               

    def consume_mean(self) -> float | None:
        if not self.values:
            return None
        value = float(np.mean(self.values))
        self.values.clear()
        return value

    def restore(self) -> None:
        self.optimizer.step = self._original_step                               


class MLflowTrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        tracker: MLflowRun,
        run_dir: Path,
        manifest_path: Path,
        eval_every: int,
        eval_seeds: tuple[int, ...],
        eval_episodes: int,
        eval_max_steps: int,
        config_path: str | None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.tracker = tracker
        self.run_dir = run_dir
        self.manifest_path = manifest_path
        self.eval_every = int(eval_every)
        self.eval_seeds = eval_seeds
        self.eval_episodes = int(eval_episodes)
        self.eval_max_steps = int(eval_max_steps)
        self.config_path = config_path
        self._last_eval_step = -1
        self._grad_tracker: GradientNormTracker | None = None
        self._uploaded_checkpoints: set[str] = set()
        self.best_survival_score = float("-inf")
        self.best_step = -1
        self.best_model_path = self.run_dir / "best_model.zip"

    def _on_training_start(self) -> None:
        self._grad_tracker = GradientNormTracker(self.model.policy.optimizer)
        self._run_eval(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._log_training_health()
        if self.num_timesteps - self._last_eval_step >= self.eval_every:
            self._run_eval(self.num_timesteps)

    def _log_training_health(self) -> None:
        values = self.model.logger.name_to_value
        mapping = {
            "train/policy_gradient_loss": "training.policy_loss",
            "train/value_loss": "training.value_loss",
            "train/approx_kl": "training.approx_kl",
            "train/clip_fraction": "training.clip_fraction",
            "train/explained_variance": "training.explained_variance",
            "train/learning_rate": "training.learning_rate",
        }
        metrics: dict[str, float | None] = {
            target: float(values[source]) if source in values else None
            for source, target in mapping.items()
        }
        metrics["training.entropy"] = (
            -float(values["train/entropy_loss"])
            if "train/entropy_loss" in values
            else None
        )
        metrics["training.grad_norm"] = (
            self._grad_tracker.consume_mean() if self._grad_tracker is not None else None
        )
        metrics["training.fps"] = float(values["time/fps"]) if "time/fps" in values else None
        self.tracker.log_metrics(metrics, self.num_timesteps)

    def _run_eval(self, step: int) -> None:
        eval_dir = self.run_dir / "evaluations" / f"step_{step:09d}"
        payload = evaluate(
            lambda _seed: PPOAgent(self.model, deterministic=True),
            EvaluationSpec(
                                                                                      
                                                                                     
                                                    
                split="train",
                seeds=self.eval_seeds,
                episodes_per_trial=self.eval_episodes,
                max_steps=self.eval_max_steps,
                config_path=self.config_path,
                render_trials=1,
                render_stride=6,
            ),
            manifest_path=self.manifest_path,
            output_dir=eval_dir,
        )
        summary = payload["summary"]
        metrics: dict[str, float | None] = {
            "validation.mean_raw_return": summary["mean_raw_return"],
            "validation.raw_adaptation_delta": summary["raw_adaptation_delta"],
            "validation.trial_auc": summary["trial_auc"],
            "validation.adaptation_delta": summary["adaptation_delta"],
            "validation.success_rate": summary["success_rate"],
            "validation.flip_rate": summary["flip_rate"],
            "validation.mean_battery_consumed": summary["mean_battery_consumed"],
            "validation.mean_distance": summary["mean_distance"],
            "validation.survival_auc": summary["survival_auc"],
            "validation.survival_adaptation_delta": summary["survival_adaptation_delta"],
            "validation.fatal_error_rate": summary["fatal_error_rate"],
            "validation.behavior.lidar.mean_scan_count": summary["mean_lidar_scan_count"],
            "validation.behavior.lidar.active_fraction": summary["mean_lidar_active_fraction"],
            "validation.behavior.lidar.energy_spent": summary["mean_lidar_energy_spent"],
            "validation.behavior.lidar.airborne_attempt_count": summary[
                "mean_lidar_airborne_attempt_count"
            ],
            "validation.behavior.solar.toggle_count": summary["mean_solar_toggle_count"],
            "validation.behavior.solar.active_fraction": summary[
                "mean_charging_active_fraction"
            ],
            "validation.behavior.solar.energy_gained": summary["mean_solar_energy_gained"],
            "validation.behavior.ballistic.flight_count": summary[
                "mean_ballistic_flight_count"
            ],
            "validation.behavior.ballistic.airborne_fraction": summary[
                "mean_airborne_fraction"
            ],
            "validation.behavior.ballistic.safe_landing_count": summary[
                "mean_safe_landing_count"
            ],
        }
        for index, value in enumerate(summary["raw_return_by_episode"], start=1):
            metrics[f"validation.raw_return.episode_{index}"] = value
        normalized = summary["normalized_return_by_episode"] or []
        for index, value in enumerate(normalized, start=1):
            metrics[f"validation.normalized_return.episode_{index}"] = value
        for index, value in enumerate(summary["action_entropy_by_episode"], start=1):
            metrics[f"validation.action_entropy.episode_{index}"] = value
        for index, value in enumerate(summary["survival_rate_by_episode"], start=1):
            metrics[f"validation.survival.episode_{index}"] = value
        self.tracker.log_metrics(metrics, step)
                                                                                
                                                                                   
                                                                                 
                                                                                       
        mean_distance = float(summary["mean_distance"])
        passes_distance_floor = mean_distance >= 25.0
        selection_score = (
            float(summary["survival_auc"]) + 1.0e-3 * mean_distance
            if passes_distance_floor
            else float("-inf")
        )
        self.tracker.log_metrics(
            {
                "validation.selection_score": (
                    selection_score if passes_distance_floor else None
                ),
                "validation.selection_passes_distance_floor": float(
                    passes_distance_floor
                ),
            },
            step,
        )
        if selection_score > self.best_survival_score:
            self.best_survival_score = selection_score
            self.best_step = int(step)
            self.model.save(self.best_model_path)
            self.tracker.log_artifact(self.best_model_path, artifact_path="checkpoints/best")
        local_gif = eval_dir / "behavior.gif"
        remote_gif = f"evaluations/step_{step:09d}/behavior.gif"
        try:
            self.tracker.log_artifact(eval_dir, artifact_path=f"evaluations/step_{step:09d}")
        finally:
                                                                             
                                                                              
                                                                              
                                                                              
                                          
            if local_gif.is_file() and self.tracker.artifact_exists(remote_gif):
                local_gif.unlink()
        checkpoints = sorted(
            (self.run_dir / "checkpoints").glob("robust_ppo_*_steps.zip"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if checkpoints and checkpoints[-1].name not in self._uploaded_checkpoints:
            self.tracker.log_artifact(checkpoints[-1], artifact_path="checkpoints")
            self._uploaded_checkpoints.add(checkpoints[-1].name)
        self._last_eval_step = step

    def _on_training_end(self) -> None:
        try:
            self._log_training_health()
            if self._last_eval_step != self.num_timesteps:
                self._run_eval(self.num_timesteps)
        finally:
            if self._grad_tracker is not None:
                self._grad_tracker.restore()
