from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environment" / "python"))
from mars_rover_env.actions import ACTION_MACROS as BASE_ACTION_MACROS  # noqa: E402
from mars_rover_env.config import load_env_config  # noqa: E402
from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv  # noqa: E402

EVENT_MASK = sum(1 << bit for bit in range(6, 12))
HEATER = 1 << 12
ACTION_MACROS = tuple(BASE_ACTION_MACROS) + (
    HEATER,
    1 | HEATER,
    1 | (1 << 4) | HEATER,
    1 | (1 << 5) | HEATER,
)


@dataclass
class Config:
    total_frames: int = 20_000_000
    num_envs: int = 64
    rollout_steps: int = 512
    frame_skip: int = 4
    hidden_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.9995
    gae_lambda: float = 0.995
    clip: float = 0.2
    entropy: float = 0.015
    value_coef: float = 0.5
    frontier_bonus_scale: float = 2.0
    epochs: int = 4
    envs_per_batch: int = 8
    seed: int = 1
    biome_split: int = 1
    device: str = "auto"


class RunningMeanStd:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        mean, var, count = x.mean(0), x.var(0), len(x)
        delta, total = mean - self.mean, self.count + count
        self.mean += delta * count / total
        self.var = (self.var * self.count + var * count + delta**2 * self.count * count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean, self.var = np.asarray(state["mean"]), np.asarray(state["var"])
        self.count = float(state["count"])


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim, self.action_dim, self.hidden_size = obs_dim, action_dim, hidden_size
        self.encoder = nn.Sequential(nn.Linear(obs_dim + action_dim + 3, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor, self.critic = nn.Linear(hidden_size, action_dim), nn.Linear(hidden_size, 1)
        for layer in (self.encoder[0], self.actor, self.critic):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        # Universal control prior. Uniform random actions repeatedly toggle critical
        # systems and rarely discover locomotion; PPO can overwrite these biases.
        with torch.no_grad():
            self.actor.bias.fill_(-1.0)
            preferences = {1: 2.5, 4: 1.5, 5: 1.25, 10: 0.25,
                           11: 0.25, 12: -0.25, 15: -0.25, 16: 0.5,
                           17: -0.25, 18: -0.25}
            for action, bias in preferences.items():
                if action < action_dim:
                    self.actor.bias[action] = bias

    def initial(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.hidden_size, device=device)

    def step(self, obs, prev_action, prev_reward, prev_done, episode, trial_start, hidden):
        hidden = hidden * (1 - trial_start.float().unsqueeze(-1))
        one_hot = torch.nn.functional.one_hot(prev_action, self.action_dim).float()
        features = torch.cat((obs, one_hot, torch.tanh(prev_reward[:, None] / 10),
                              prev_done[:, None], episode[:, None]), -1)
        hidden = self.gru(self.encoder(features), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden

    def sequence(self, obs, pa, pr, pd, episode, start, hidden):
        logits, values = [], []
        for t in range(len(obs)):
            logit, value, hidden = self.step(obs[t], pa[t], pr[t], pd[t], episode[t], start[t], hidden)
            logits.append(logit); values.append(value)
        return torch.stack(logits), torch.stack(values)


class Runner:
    def __init__(self, n: int, split: int, seed: int, skip: int, gamma: float,
                 frontier_bonus_scale: float = 0.0):
        self.env = MarsRoverVecEnv(n, biome_split=split)
        self.finish_x = float(load_env_config().termination.finish_x)
        self.trial_budget = int(self.env.core.trial_step_budget(0))
        self.n, self.skip, self.gamma = n, skip, gamma
        self.frontier_bonus_scale = frontier_bonus_scale
        self.rng = np.random.default_rng(seed + 17)
        self.obs = self.env.reset(seed).copy()
        self.start = np.ones(n, np.float32); self.prev_done = np.ones(n, np.float32)
        self.prev_action = np.zeros(n, np.int64); self.prev_reward = np.zeros(n, np.float32)
        self.attempt = np.zeros(n, np.int32)
        self.trial = np.zeros(n, np.int64)
        self.returns = np.zeros(n); self.lengths = np.zeros(n, np.int64)
        self.trial_steps = np.zeros(n, np.int64)
        self.trial_returns = np.zeros(n, np.float64)
        self.trial_best_distance = np.zeros(n, np.float64)
        self.trial_frontier = np.zeros(n, np.float64)
        self.trial_total_distance = np.zeros(n, np.float64)
        self.completed: list[dict[str, float]] = []
        self.completed_trials: list[dict[str, float]] = []

    @property
    def trial_fraction(self):
        if self.trial_budget <= 0:
            return np.zeros(self.n, np.float32)
        return np.clip(self.trial_steps / self.trial_budget, 0.0, 1.0).astype(np.float32)

    def step(self, actions: np.ndarray, frame_callback=None):
        macros = np.take(np.asarray(ACTION_MACROS, np.int32), actions)
        reward_sum = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool); next_start = np.zeros(self.n, bool); repeats = 0
        for repeat in range(self.skip):
            applied = macros if repeat == 0 else macros & ~EVENT_MASK
            obs, reward, terminated, truncated, _ = self.env.step(applied)
            if frame_callback is not None:
                frame_callback(self)
            current_distance = np.maximum(0.0, obs[:, 0] * self.finish_x - 1.0)
            frontier_gain = np.maximum(0.0, current_distance - self.trial_frontier)
            self.trial_frontier = np.maximum(self.trial_frontier, current_distance)
            self.trial_best_distance = np.maximum(self.trial_best_distance, current_distance)
            shaped_reward = reward + self.frontier_bonus_scale * frontier_gain
            reward_sum += self.gamma**repeat * shaped_reward
            self.returns += shaped_reward; self.trial_returns += shaped_reward
            self.lengths += 1; self.trial_steps += 1; repeats += 1
            done = terminated | truncated
            if done.any():
                terminal = obs.copy()
                for i in np.flatnonzero(done):
                    is_start = bool(self.env.next_trial_start[i]); next_start[i] = is_start
                    distance = max(0.0, float(terminal[i, 0]) * self.finish_x - 1.0)
                    self.completed.append({
                        "env": float(i), "trial": float(self.trial[i]),
                        "attempt": float(self.attempt[i] + 1), "return": float(self.returns[i]),
                        "length": float(self.lengths[i]), "progress": float(terminal[i, 0]),
                        "distance_m": distance,
                        "success": float(terminal[i, 0] >= 1.0),
                    })
                    self.trial_best_distance[i] = max(self.trial_best_distance[i], distance)
                    self.trial_total_distance[i] += distance
                    if is_start:
                        self.completed_trials.append({
                            "env": float(i), "trial": float(self.trial[i]),
                            "return": float(self.trial_returns[i]),
                            "best_distance_m": float(self.trial_best_distance[i]),
                            "total_distance_m": float(self.trial_total_distance[i]),
                            "attempts": float(self.attempt[i] + 1),
                            "success": float(self.trial_best_distance[i] >= self.finish_x - 1.0),
                        })
                    seed = int(self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                    self.env.reset_at(i, seed, trial_start=is_start)
                    self.returns[i] = 0; self.lengths[i] = 0
                    if is_start:
                        self.trial[i] += 1; self.attempt[i] = 0; self.trial_steps[i] = 0
                        self.trial_returns[i] = 0; self.trial_best_distance[i] = 0
                        self.trial_total_distance[i] = 0; self.trial_frontier[i] = 0
                    else:
                        self.attempt[i] += 1
                break
        self.obs = self.env.obs.copy(); self.start = next_start.astype(np.float32)
        self.prev_done = done.astype(np.float32); self.prev_action = actions.copy()
        self.prev_reward = reward_sum.copy()
        return reward_sum, next_start, repeats * self.n

    def pop(self):
        result, self.completed = self.completed, []
        return result

    def pop_trials(self):
        result, self.completed_trials = self.completed_trials, []
        return result


def summary(trials: list[dict[str, float]], attempts: list[dict[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {"trials": len(trials), "attempts": len(attempts)}
    if trials:
        for key in ("return", "best_distance_m", "total_distance_m", "attempts", "success"):
            values = np.asarray([row[key] for row in trials])
            result[key + "_mean"] = float(values.mean())
            result[key + "_median"] = float(np.median(values))
            result[key + "_p90"] = float(np.percentile(values, 90))
            result[key + "_max"] = float(values.max())
    grouped: dict[tuple[int, int], list[dict[str, float]]] = {}
    for row in attempts:
        grouped.setdefault((int(row["env"]), int(row["trial"])), []).append(row)
    deltas, first_distances, later_best = [], [], []
    for rows in grouped.values():
        rows.sort(key=lambda row: row["attempt"])
        first = rows[0]["distance_m"]
        later = max((row["distance_m"] for row in rows[1:]), default=first)
        first_distances.append(first); later_best.append(later); deltas.append(later - first)
    if deltas:
        result["first_attempt_distance_mean"] = float(np.mean(first_distances))
        result["later_best_distance_mean"] = float(np.mean(later_best))
        result["adaptation_distance_delta_mean"] = float(np.mean(deltas))
    return result


def flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isfinite(number):
                flat[name] = number
    return flat


class MlflowTracker:
    def __init__(self, module: Any): self.mlflow = module

    def metrics(self, data: dict[str, Any], step: int, prefix: str = "") -> None:
        values = flatten_metrics(data, prefix)
        if values: self.mlflow.log_metrics(values, step=int(step), synchronous=False)

    def artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if path.is_file(): self.mlflow.log_artifact(str(path), artifact_path=artifact_path)


@contextlib.contextmanager
def mlflow_run(tracking_uri: str | None, experiment: str, run_name: str,
               tags: dict[str, str]):
    if tracking_uri is None or tracking_uri.lower() in {"", "none", "disabled"}:
        yield None; return
    import mlflow
    import _mars_rover_cpp as native
    uri = tracking_uri; artifact_location: str | None = None
    if "://" not in uri:
        database = Path(uri).resolve()
        if database.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            database = database / "mlflow.db"
        database.parent.mkdir(parents=True, exist_ok=True)
        artifacts = database.parent / "mlflow-artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
        artifact_location = artifacts.resolve().as_uri(); uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None and artifact_location is not None:
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)
    full_tags = {"environment_version": str(native.environment_version()),
                 "biome_bank_version": str(native.biome_bank_version()), **tags}
    with mlflow.start_run(run_name=run_name, tags=full_tags):
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")
        yield MlflowTracker(mlflow)


def device_for(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def tt(x, device, dtype=None):
    return torch.as_tensor(x, device=device, dtype=dtype)


def save(path: Path, model, optimizer, rms, config, frames):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rms": rms.state_dict(), "config": asdict(config), "frames": frames,
                "obs_dim": model.obs_dim, "action_dim": model.action_dim}, temp)
    temp.replace(path)


def train(config: Config, output: Path, tracker: MlflowTracker | None = None,
          mlflow_checkpoint_every: int = 10):
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = device_for(config.device)
    runner = Runner(config.num_envs, config.biome_split, config.seed, config.frame_skip,
                    config.gamma, config.frontier_bonus_scale)
    model = Agent(runner.env.obs_dim, len(ACTION_MACROS), config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    rms = RunningMeanStd(runner.env.obs_dim); hidden = model.initial(config.num_envs, device)
    output.mkdir(parents=True, exist_ok=True)
    if tracker is not None:
        tracker.mlflow.log_params({**asdict(config), "action_macros": len(ACTION_MACROS),
                                   "train_tracks_parallel": config.num_envs})
    frames = updates = 0; attempts: list[dict[str, float]] = []
    trials: list[dict[str, float]] = []; started = time.perf_counter()
    while frames < config.total_frames:
        keys = ("obs", "pa", "pr", "pd", "ep", "start", "action", "logp", "value",
                "reward", "discount", "ldiscount", "trial_end")
        store = {k: [] for k in keys}; initial_hidden = hidden.detach().clone()
        for _ in range(config.rollout_steps):
            rms.update(runner.obs); obs = rms.normalize(runner.obs)
            current = (obs, runner.prev_action.copy(), runner.prev_reward.copy(), runner.prev_done.copy(),
                       runner.trial_fraction.copy(), runner.start.copy())
            with torch.no_grad():
                logits, value, hidden = model.step(tt(obs, device), tt(current[1], device, torch.long),
                    tt(current[2], device), tt(current[3], device), tt(current[4], device),
                    tt(current[5], device), hidden)
                dist = Categorical(logits=logits); action = dist.sample()
            reward, trial_end, used = runner.step(action.cpu().numpy()); repeats = used // config.num_envs
            for key, item in zip(("obs", "pa", "pr", "pd", "ep", "start"), current): store[key].append(item)
            for key, item in (("action", action.cpu().numpy()), ("logp", dist.log_prob(action).cpu().numpy()),
                              ("value", value.cpu().numpy()), ("reward", reward),
                              ("discount", np.full(config.num_envs, config.gamma**repeats, np.float32)),
                              ("ldiscount", np.full(config.num_envs, config.gae_lambda**repeats, np.float32)),
                              ("trial_end", trial_end.astype(np.float32))): store[key].append(item)
            frames += used; attempts.extend(runner.pop()); trials.extend(runner.pop_trials())
        with torch.no_grad():
            _, next_value, _ = model.step(tt(rms.normalize(runner.obs), device),
                tt(runner.prev_action, device, torch.long), tt(runner.prev_reward, device),
                tt(runner.prev_done, device), tt(runner.trial_fraction, device),
                tt(runner.start, device), hidden)
        batch = {k: np.stack(v) for k, v in store.items()}
        adv = np.zeros_like(batch["reward"]); last = np.zeros(config.num_envs, np.float32)
        for t in reversed(range(config.rollout_steps)):
            nv = next_value.cpu().numpy() if t == config.rollout_steps - 1 else batch["value"][t + 1]
            cont = 1 - batch["trial_end"][t]
            delta = batch["reward"][t] + batch["discount"][t] * nv * cont - batch["value"][t]
            last = delta + batch["discount"][t] * batch["ldiscount"][t] * cont * last; adv[t] = last
        returns = adv + batch["value"]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        order = np.arange(config.num_envs); losses = []
        for _ in range(config.epochs):
            np.random.shuffle(order)
            for begin in range(0, config.num_envs, config.envs_per_batch):
                ids = order[begin:begin + config.envs_per_batch]
                logits, values = model.sequence(tt(batch["obs"][:, ids], device),
                    tt(batch["pa"][:, ids], device, torch.long), tt(batch["pr"][:, ids], device),
                    tt(batch["pd"][:, ids], device), tt(batch["ep"][:, ids], device),
                    tt(batch["start"][:, ids], device), initial_hidden[ids])
                dist = Categorical(logits=logits); actions = tt(batch["action"][:, ids], device, torch.long)
                ratio = (dist.log_prob(actions) - tt(batch["logp"][:, ids], device)).exp()
                a = tt(adv[:, ids], device); policy = -torch.min(ratio * a, ratio.clamp(1-config.clip, 1+config.clip) * a).mean()
                critic = .5 * (values - tt(returns[:, ids], device)).square().mean(); entropy = dist.entropy().mean()
                loss = policy + config.value_coef * critic - config.entropy * entropy
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .5); optimizer.step()
                losses.append((float(policy.detach()), float(critic.detach()), float(entropy.detach())))
        hidden = hidden.detach(); updates += 1
        values = np.mean(losses, 0); report = {"update": updates, "frames": frames,
            "fps": int(frames / (time.perf_counter()-started)), "policy_loss": values[0],
            "value_loss": values[1], "entropy": values[2],
            "recent": summary(trials[-100:], attempts[-1000:])}
        print(json.dumps(report), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(report)+"\n")
        checkpoint = output / "checkpoint.pt"
        save(checkpoint, model, optimizer, rms, config, frames)
        if tracker is not None:
            tracker.metrics(report, frames, "train")
            if mlflow_checkpoint_every > 0 and updates % mlflow_checkpoint_every == 0:
                tracker.artifact(checkpoint, f"checkpoints/update_{updates:06d}")
    final = summary(trials, attempts)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    if tracker is not None:
        tracker.metrics(final, frames, "train.final")
        tracker.artifact(output / "checkpoint.pt", "model")
        tracker.artifact(output / "metrics.jsonl", "metrics")
        tracker.artifact(summary_path, "metrics")


@torch.no_grad()
def evaluate(path: Path, split: int, trials: int, seed: int, stochastic: bool,
             device_name: str, reset_memory_on_attempt: bool = False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device_for(device_name); ckpt = torch.load(path, map_location=device, weights_only=False)
    config = Config(**ckpt["config"]); n = min(64, trials)
    runner = Runner(n, split, seed, config.frame_skip, config.gamma,
                    config.frontier_bonus_scale)
    model = Agent(ckpt["obs_dim"], ckpt["action_dim"], config.hidden_size).to(device)
    model.load_state_dict(ckpt["model"]); model.eval(); rms = RunningMeanStd(ckpt["obs_dim"]); rms.load_state_dict(ckpt["rms"])
    hidden = model.initial(n, device); attempt_rows = []; trial_rows = []
    quotas = np.full(n, trials // n, dtype=np.int64); quotas[: trials % n] += 1
    while len(trial_rows) < trials:
        memory_reset = (np.maximum(runner.start, runner.prev_done)
                        if reset_memory_on_attempt else runner.start)
        logits, _, hidden = model.step(tt(rms.normalize(runner.obs), device), tt(runner.prev_action, device, torch.long),
            tt(runner.prev_reward, device), tt(runner.prev_done, device), tt(runner.trial_fraction, device), tt(memory_reset, device), hidden)
        actions = Categorical(logits=logits).sample() if stochastic else logits.argmax(-1)
        runner.step(actions.cpu().numpy())
        attempt_rows.extend(row for row in runner.pop()
                            if int(row["trial"]) < quotas[int(row["env"])])
        trial_rows.extend(row for row in runner.pop_trials()
                          if int(row["trial"]) < quotas[int(row["env"])])
    return summary(trial_rows, attempt_rows)


def benchmark(checkpoint: Path, split: int, trials_per_repeat: int, repeats: int,
              base_seed: int, seed_stride: int, stochastic: bool, device: str,
              reset_memory_on_attempt: bool, output: Path,
              tracker: MlflowTracker | None = None) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for repeat in range(repeats):
        seed = base_seed + repeat * seed_stride
        metrics = evaluate(checkpoint, split, trials_per_repeat, seed, stochastic,
                           device, reset_memory_on_attempt)
        runs.append({"repeat": repeat, "seed": seed, **metrics})
        print(json.dumps({"evaluation_repeat": repeat, "seed": seed, **metrics}), flush=True)
        if tracker is not None: tracker.metrics(metrics, repeat, "eval.repeat")
    numeric_keys = sorted(set.intersection(*[
        {key for key, value in run.items() if isinstance(value, (int, float))} for run in runs
    ]))
    aggregate: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        if key in {"repeat", "seed"}: continue
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        aggregate[key] = {"mean": float(values.mean()),
                          "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                          "min": float(values.min()), "max": float(values.max())}
    report = {"checkpoint": str(checkpoint.resolve()), "biome_split": split,
              "trials_per_repeat": trials_per_repeat, "repeats": repeats,
              "stochastic": stochastic, "reset_memory_on_attempt": reset_memory_on_attempt,
              "runs": runs, "aggregate": aggregate}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if tracker is not None:
        tracker.metrics(aggregate, repeats, "eval.aggregate")
        tracker.artifact(checkpoint, "model"); tracker.artifact(output, "evaluation")
    return report


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train"); p.add_argument("--output", type=Path, default=Path("codex/artifacts/meta_ppo"))
    for name, typ, default in (("total-frames", int, Config.total_frames), ("num-envs", int, Config.num_envs),
        ("rollout-steps", int, Config.rollout_steps), ("frame-skip", int, Config.frame_skip),
        ("hidden-size", int, Config.hidden_size), ("learning-rate", float, Config.learning_rate),
        ("gamma", float, Config.gamma), ("gae-lambda", float, Config.gae_lambda),
        ("frontier-bonus-scale", float, Config.frontier_bonus_scale),
        ("epochs", int, Config.epochs), ("envs-per-batch", int, Config.envs_per_batch),
        ("seed", int, Config.seed), ("biome-split", int, Config.biome_split)):
        p.add_argument("--"+name, type=typ, default=default)
    p.add_argument("--device", default="auto")
    p.add_argument("--mlflow-tracking-uri", default="codex/artifacts/mlflow.db")
    p.add_argument("--mlflow-experiment", default="mars-rover-meta-rl")
    p.add_argument("--run-name"); p.add_argument("--mlflow-checkpoint-every", type=int, default=10)
    e = sub.add_parser("evaluate"); e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--biome-split", type=int, choices=(1,2), default=2); e.add_argument("--trials", type=int, default=50)
    e.add_argument("--seed", type=int, default=10000); e.add_argument("--stochastic", action="store_true"); e.add_argument("--device", default="auto")
    e.add_argument("--reset-memory-on-attempt", action="store_true")
    b = sub.add_parser("benchmark"); b.add_argument("--checkpoint", type=Path, required=True)
    b.add_argument("--biome-split", type=int, choices=(1, 2), default=2)
    b.add_argument("--trials-per-repeat", type=int, default=20); b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--base-seed", type=int, default=10000); b.add_argument("--seed-stride", type=int, default=1000)
    b.add_argument("--stochastic", action="store_true"); b.add_argument("--reset-memory-on-attempt", action="store_true")
    b.add_argument("--device", default="auto"); b.add_argument("--output", type=Path, default=Path("codex/artifacts/benchmark.json"))
    b.add_argument("--mlflow-tracking-uri", default="codex/artifacts/mlflow.db")
    b.add_argument("--mlflow-experiment", default="mars-rover-meta-rl"); b.add_argument("--run-name")
    args = parser.parse_args()
    if args.command == "train":
        config = Config(total_frames=args.total_frames, num_envs=args.num_envs, rollout_steps=args.rollout_steps,
            frame_skip=args.frame_skip, hidden_size=args.hidden_size, learning_rate=args.learning_rate,
            gamma=args.gamma, gae_lambda=args.gae_lambda, frontier_bonus_scale=args.frontier_bonus_scale,
            epochs=args.epochs, envs_per_batch=args.envs_per_batch,
            seed=args.seed, biome_split=args.biome_split, device=args.device)
        run_name = args.run_name or f"train-v12-seed-{args.seed}"
        with mlflow_run(args.mlflow_tracking_uri, args.mlflow_experiment, run_name,
                        {"run_type": "train", "biome_split": str(args.biome_split)}) as tracker:
            train(config, args.output, tracker, args.mlflow_checkpoint_every)
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.checkpoint, args.biome_split, args.trials, args.seed,
                                  args.stochastic, args.device, args.reset_memory_on_attempt), indent=2))
    else:
        run_name = args.run_name or f"benchmark-split-{args.biome_split}"
        with mlflow_run(args.mlflow_tracking_uri, args.mlflow_experiment, run_name,
                        {"run_type": "benchmark", "biome_split": str(args.biome_split)}) as tracker:
            if tracker is not None:
                tracker.mlflow.log_params({"checkpoint": str(args.checkpoint.resolve()),
                    "trials_per_repeat": args.trials_per_repeat, "repeats": args.repeats,
                    "base_seed": args.base_seed, "seed_stride": args.seed_stride,
                    "stochastic": args.stochastic,
                    "reset_memory_on_attempt": args.reset_memory_on_attempt})
            result = benchmark(args.checkpoint, args.biome_split, args.trials_per_repeat,
                args.repeats, args.base_seed, args.seed_stride, args.stochastic, args.device,
                args.reset_memory_on_attempt, args.output, tracker)
            print(json.dumps({"aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__": main()
