"""Оценка на среде v12: как далеко ровер уезжает за 120-секундный бюджет.

Оценка всегда идёт по НЕСКОЛЬКИМ трассам (сидам) и усредняется — отдельно на
train-биомах (`--biome-split 1`) и на невиданных test-биомах (`--biome-split 2`).

Попыток внутри трайла не ограничено, поэтому главная метрика — не «выжил», а
скорость: сколько метров получилось за отведённое время.

    .venv/bin/python claude/evaluate.py --ckpt claude/runs/rl2_v12/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meta_env import MAX_ATTEMPT_BUCKETS, MetaVecEnv
from train_rl2 import RecurrentActorCritic, RunningNorm

FINISH_X = 800.0


@torch.no_grad()
def evaluate_seed(
    net: "RecurrentActorCritic",
    obs_norm: "RunningNorm",
    *,
    biome_split: int,
    seed: int,
    num_envs: int,
    trials: int,
    device: torch.device,
    greedy: bool = False,
) -> dict:
    """Один прогон на одной трассе (сиде). Агрегаты по завершённым трайлам."""
    env = MetaVecEnv(num_envs, seed=seed, biome_split=biome_split)
    prog = [[] for _ in range(MAX_ATTEMPT_BUCKETS)]
    speed = [[] for _ in range(MAX_ATTEMPT_BUCKETS)]
    best, total, rets, atts, succ = [], [], [], [], []

    obs_np, ts_np = env.reset()
    obs = torch.as_tensor(obs_np, device=device)
    ts = torch.as_tensor(ts_np.astype(np.float32), device=device)
    h = torch.zeros(1, env.num_envs, net.gru_size, device=device)
    was_training = net.training
    net.eval()

    while len(best) < trials:
        logits, _, h = net(obs_norm(obs).unsqueeze(0), h, ts.unsqueeze(0))
        a = (
            logits[0].argmax(-1)
            if greedy
            else torch.distributions.Categorical(logits=logits[0]).sample()
        )
        nobs, _, _, ts_np, infos = env.step(a.cpu().numpy())
        obs = torch.as_tensor(nobs, device=device)
        ts = torch.as_tensor(ts_np.astype(np.float32), device=device)
        for info in infos:
            b = min(info["attempt"], MAX_ATTEMPT_BUCKETS - 1)
            prog[b].append(info["progress"])
            speed[b].append(info["speed_mps"])
            succ.append(float(info["success"]))
            if "trial" in info:
                best.append(info["trial"]["best_progress"])
                total.append(info["trial"]["sum_progress"])
                rets.append(info["trial"]["return"])
                atts.append(info["trial"]["attempts"])
    if was_training:
        net.train()

    speed_by_attempt = [float(np.mean(r)) if r else None for r in speed]
    known = [v for v in speed_by_attempt if v is not None]
    return {
        "seed": seed,
        "trials": len(best),
        # главное: лучший заезд за 120 с и суммарно пройденное
        "best_run_m": float(np.mean(best)) * FINISH_X,
        "total_distance_m": float(np.mean(total)) * FINISH_X,
        "trial_return": float(np.mean(rets)),
        "attempts_per_trial": float(np.mean(atts)),
        "finish_rate": float(np.mean(succ)),
        "meters_by_attempt": [float(np.mean(r)) * FINISH_X if r else None for r in prog],
        "speed_mps_by_attempt": speed_by_attempt,
        "adaptation_gain_mps": (known[-1] - known[0]) if len(known) > 1 else 0.0,
    }


def evaluate_many(
    net,
    obs_norm,
    *,
    biome_split: int,
    seeds: list[int],
    num_envs: int,
    trials: int,
    device: torch.device,
    greedy: bool = False,
) -> dict:
    """Несколько трасс -> среднее и разброс между трассами."""
    per_seed = [
        evaluate_seed(
            net,
            obs_norm,
            biome_split=biome_split,
            seed=s,
            num_envs=num_envs,
            trials=trials,
            device=device,
            greedy=greedy,
        )
        for s in seeds
    ]
    out: dict = {"per_seed": per_seed, "seeds": seeds}
    for k in (
        "best_run_m",
        "total_distance_m",
        "trial_return",
        "attempts_per_trial",
        "finish_rate",
        "adaptation_gain_mps",
    ):
        vals = [r[k] for r in per_seed]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    for name in ("meters_by_attempt", "speed_mps_by_attempt"):
        stacked = []
        for b in range(MAX_ATTEMPT_BUCKETS):
            vals = [r[name][b] for r in per_seed if r[name][b] is not None]
            stacked.append(float(np.mean(vals)) if vals else None)
        out[name] = stacked
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "runs/rl2_v12/ckpt.pt"))
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--trials", type=int, default=64, help="трайлов на одну трассу")
    p.add_argument("--train-seeds", type=str, default="101,202,303")
    p.add_argument("--test-seeds", type=str, default="1001,2002,3003")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--json-out", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    probe = MetaVecEnv(2, seed=0)
    net = RecurrentActorCritic(probe.obs_dim, probe.num_actions).to(device)
    net.load_state_dict(ckpt["net"])
    obs_norm = RunningNorm(probe.obs_dim, device)
    obs_norm.load_state_dict(ckpt["obs_norm"])

    report = {"ckpt": args.ckpt, "update": int(ckpt.get("update", -1))}
    for name, split, seeds in (("train", 1, args.train_seeds), ("test", 2, args.test_seeds)):
        report[name] = evaluate_many(
            net,
            obs_norm,
            biome_split=split,
            seeds=[int(s) for s in seeds.split(",")],
            num_envs=args.num_envs,
            trials=args.trials,
            device=device,
            greedy=args.greedy,
        )
    text = json.dumps(report, indent=2, default=lambda o: round(o, 3))
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
