"""Оценка адаптации: средняя награда/прогресс по номеру эпизода внутри трайла.

Главная метрика meta-RL — прирост от эпизода 1 к эпизоду 4 внутри одного трайла
(одна и та же скрытая задача). Также умеет оценивать на test-биомах (--biome-split 2).

    .venv/bin/python claude/evaluate.py --ckpt claude/runs/rl2/ckpt.pt --trials 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meta_env import MetaVecEnv
from train_rl2 import RecurrentActorCritic, RunningNorm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "runs/rl2/ckpt.pt"))
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--trials", type=int, default=200, help="сколько трайлов набрать")
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

    K = env.episodes_per_trial
    rets = [[] for _ in range(K)]
    progs = [[] for _ in range(K)]
    succs = [[] for _ in range(K)]
    obs_np, ts_np = env.reset()
    obs = torch.as_tensor(obs_np, device=device)
    ts = torch.as_tensor(ts_np.astype(np.float32), device=device)
    h = torch.zeros(1, env.num_envs, net.gru_size, device=device)
    done_trials = 0

    with torch.no_grad():
        while done_trials < args.trials:
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
                e = info["episode_in_trial"]
                rets[e].append(info["return"])
                progs[e].append(info["progress"])
                succs[e].append(float(info["success"]))
                if e == K - 1:
                    done_trials += 1

    report = {
        "ckpt": args.ckpt,
        "biome_split": args.biome_split,
        "trials": done_trials,
        "return_by_episode": [round(float(np.mean(r)), 2) for r in rets],
        "progress_by_episode": [round(float(np.mean(r)), 4) for r in progs],
        "success_by_episode": [round(float(np.mean(r)), 4) for r in succs],
    }
    report["adaptation_gain"] = round(
        report["progress_by_episode"][-1] - report["progress_by_episode"][0], 4
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
