from __future__ import annotations

import argparse
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
from mars_rover_env.actions import ACTION_MACROS  # noqa: E402
from mars_rover_env.config import load_env_config  # noqa: E402
from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv  # noqa: E402

EVENT_MASK = sum(1 << bit for bit in range(6, 12))


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
                           11: 0.25, 12: -0.25}
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
    def __init__(self, n: int, split: int, seed: int, skip: int, gamma: float):
        self.env = MarsRoverVecEnv(n, biome_split=split)
        self.finish_x = float(load_env_config().termination.finish_x)
        self.trial_budget = int(self.env.core.trial_step_budget(0))
        self.n, self.skip, self.gamma = n, skip, gamma
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
        self.trial_total_distance = np.zeros(n, np.float64)
        self.completed: list[dict[str, float]] = []
        self.completed_trials: list[dict[str, float]] = []

    @property
    def trial_fraction(self):
        if self.trial_budget <= 0:
            return np.zeros(self.n, np.float32)
        return np.clip(self.trial_steps / self.trial_budget, 0.0, 1.0).astype(np.float32)

    def step(self, actions: np.ndarray):
        macros = np.take(np.asarray(ACTION_MACROS, np.int32), actions)
        reward_sum = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool); next_start = np.zeros(self.n, bool); repeats = 0
        for repeat in range(self.skip):
            applied = macros if repeat == 0 else macros & ~EVENT_MASK
            obs, reward, terminated, truncated, _ = self.env.step(applied)
            reward_sum += self.gamma**repeat * reward
            self.returns += reward; self.trial_returns += reward
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
                        self.trial_total_distance[i] = 0
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


def train(config: Config, output: Path):
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = device_for(config.device)
    runner = Runner(config.num_envs, config.biome_split, config.seed, config.frame_skip, config.gamma)
    model = Agent(runner.env.obs_dim, len(ACTION_MACROS), config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    rms = RunningMeanStd(runner.env.obs_dim); hidden = model.initial(config.num_envs, device)
    output.mkdir(parents=True, exist_ok=True)
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
        save(output / "checkpoint.pt", model, optimizer, rms, config, frames)
    final = summary(trials, attempts)
    (output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(path: Path, split: int, trials: int, seed: int, stochastic: bool,
             device_name: str, reset_memory_on_attempt: bool = False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device_for(device_name); ckpt = torch.load(path, map_location=device, weights_only=False)
    config = Config(**ckpt["config"]); n = min(64, trials)
    runner = Runner(n, split, seed, config.frame_skip, config.gamma)
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


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train"); p.add_argument("--output", type=Path, default=Path("codex/artifacts/meta_ppo"))
    for name, typ, default in (("total-frames", int, Config.total_frames), ("num-envs", int, Config.num_envs),
        ("rollout-steps", int, Config.rollout_steps), ("frame-skip", int, Config.frame_skip),
        ("hidden-size", int, Config.hidden_size), ("learning-rate", float, Config.learning_rate),
        ("gamma", float, Config.gamma), ("gae-lambda", float, Config.gae_lambda),
        ("seed", int, Config.seed), ("biome-split", int, Config.biome_split)):
        p.add_argument("--"+name, type=typ, default=default)
    p.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate"); e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--biome-split", type=int, choices=(1,2), default=2); e.add_argument("--trials", type=int, default=50)
    e.add_argument("--seed", type=int, default=10000); e.add_argument("--stochastic", action="store_true"); e.add_argument("--device", default="auto")
    e.add_argument("--reset-memory-on-attempt", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train(Config(total_frames=args.total_frames, num_envs=args.num_envs, rollout_steps=args.rollout_steps,
            frame_skip=args.frame_skip, hidden_size=args.hidden_size, learning_rate=args.learning_rate,
            gamma=args.gamma, gae_lambda=args.gae_lambda, seed=args.seed, biome_split=args.biome_split, device=args.device), args.output)
    else:
        print(json.dumps(evaluate(args.checkpoint, args.biome_split, args.trials, args.seed,
                                  args.stochastic, args.device, args.reset_memory_on_attempt), indent=2))


if __name__ == "__main__": main()
