"""Оценка на среде v11: сколько метров за 120 секунд и как быстро идёт адаптация.

    .venv/bin/python claude/evaluate.py --ckpt claude/runs/rl2_v11/ckpt.pt --trials 200
    .venv/bin/python claude/evaluate.py --ckpt ... --biome-split 2   # невиданные биомы
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "runs/rl2_v11/ckpt.pt"))
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--biome-split", type=int, default=1, help="1=train, 2=test, 0=все")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    env = MetaVecEnv(args.num_envs, seed=args.seed, biome_split=args.biome_split)
    net = RecurrentActorCritic(env.obs_dim, env.num_actions).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    obs_norm = RunningNorm(env.obs_dim, device)
    obs_norm.load_state_dict(ckpt["obs_norm"])

    prog = [[] for _ in range(MAX_ATTEMPT_BUCKETS)]
    speed = [[] for _ in range(MAX_ATTEMPT_BUCKETS)]
    best, total, rets, atts, succ = [], [], [], [], []
    obs_np, ts_np = env.reset()
    obs = torch.as_tensor(obs_np, device=device)
    ts = torch.as_tensor(ts_np.astype(np.float32), device=device)
    h = torch.zeros(1, env.num_envs, net.gru_size, device=device)

    with torch.no_grad():
        while len(best) < args.trials:
            logits, _, h = net(obs_norm(obs).unsqueeze(0), h, ts.unsqueeze(0))
            a = (
                logits[0].argmax(-1)
                if args.greedy
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

    by_attempt = [round(float(np.mean(r)) * FINISH_X, 1) if r else None for r in prog]
    speed_by_attempt = [round(float(np.mean(r)), 2) if r else None for r in speed]
    known = [v for v in speed_by_attempt if v is not None]
    report = {
        "ckpt": args.ckpt,
        "biome_split": args.biome_split,
        "trials": len(best),
        "best_run_m": round(float(np.mean(best)) * FINISH_X, 1),
        "total_distance_m": round(float(np.mean(total)) * FINISH_X, 1),
        "trial_return": round(float(np.mean(rets)), 1),
        "attempts_per_trial": round(float(np.mean(atts)), 2),
        "finish_rate": round(float(np.mean(succ)), 4),
        "meters_by_attempt": by_attempt,
        "speed_mps_by_attempt": speed_by_attempt,
        # прирост от первой попытки к последней — прямая мера скорости адаптации
        "adaptation_gain_mps": round(known[-1] - known[0], 1) if len(known) > 1 else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
