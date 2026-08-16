"""PPO + GRU (RL^2) для Mars Rover.

Запуск:
    .venv/bin/python claude/train_rl2.py --steps 20_000_000 --num-envs 128

Идея решения — см. docstring в claude/meta_env.py и claude/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_env import MetaVecEnv

HERE = Path(__file__).resolve().parent


class RunningNorm:
    """Welford-нормализация наблюдений (среда выдаёт очень разные масштабы)."""

    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta.pow(2) * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256, gru: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, gru)
        self.pi = nn.Linear(gru, num_actions)
        self.vf = nn.Linear(gru, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.vf.weight, 1.0)
        nn.init.zeros_(self.vf.bias)
        self.gru_size = gru

    def forward(self, obs_seq: torch.Tensor, h: torch.Tensor, trial_start: torch.Tensor):
        """obs_seq: (T, N, obs), trial_start: (T, N) — 1.0 там, где сброс памяти."""
        x = self.encoder(obs_seq)
        # Сбросы памяти редки (раз в трайл), поэтому гоняем GRU крупными кусками
        # между точками сброса вместо шага-за-шагом — это в разы быстрее.
        cuts = (trial_start.any(dim=1).nonzero().flatten().tolist() + [x.shape[0]])
        outs, start = [], 0
        for cut in cuts:
            if cut > start:
                out, h = self.gru(x[start:cut], h)
                outs.append(out)
                start = cut
            if cut < x.shape[0]:
                h = h * (1.0 - trial_start[cut]).view(1, -1, 1)
        z = torch.cat(outs, 0)
        return self.pi(z), self.vf(z).squeeze(-1), h


def compute_gae(rew, val, done, last_val, gamma, lam):
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    last = torch.zeros(N, device=rew.device)
    nxt = last_val
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
        nxt = val[t]
    return adv, adv + val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatches", type=int, default=4, help="разбиение по средам")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--ent-final", type=float, default=0.002)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--biome-split", type=int, default=1)
    p.add_argument("--run", type=str, default="rl2")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = HERE / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "log.jsonl").open("a")

    env = MetaVecEnv(args.num_envs, seed=args.seed, biome_split=args.biome_split)
    N, T = env.num_envs, args.rollout
    net = RecurrentActorCritic(env.obs_dim, env.num_actions).to(device)
    obs_norm = RunningNorm(env.obs_dim, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        obs_norm.load_state_dict(ckpt["obs_norm"])
        start_update = ckpt["update"]

    obs_np, trial_start_np = env.reset()
    obs = torch.as_tensor(obs_np, device=device)
    trial_start = torch.as_tensor(trial_start_np.astype(np.float32), device=device)
    h = torch.zeros(1, N, net.gru_size, device=device)

    b_obs = torch.zeros(T, N, env.obs_dim, device=device)
    b_act = torch.zeros(T, N, dtype=torch.long, device=device)
    b_logp = torch.zeros(T, N, device=device)
    b_val = torch.zeros(T, N, device=device)
    b_rew = torch.zeros(T, N, device=device)
    b_done = torch.zeros(T, N, device=device)
    b_ts = torch.zeros(T, N, device=device)

    from meta_env import MAX_ATTEMPT_BUCKETS

    prog_hist = [deque(maxlen=400) for _ in range(MAX_ATTEMPT_BUCKETS)]
    trial_best = deque(maxlen=200)
    trial_sum = deque(maxlen=200)
    trial_ret = deque(maxlen=200)
    trial_att = deque(maxlen=200)
    n_updates = args.steps // (N * T)
    global_step = start_update * N * T
    t0 = time.time()

    for update in range(start_update, n_updates):
        frac = update / max(1, n_updates)
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - frac)
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        h_start = h.detach().clone()

        net.eval()
        with torch.no_grad():
            for t in range(T):
                b_obs[t] = obs
                b_ts[t] = trial_start
                logits, value, h = net(obs_norm(obs).unsqueeze(0), h, trial_start.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits[0])
                a = dist.sample()
                b_act[t], b_logp[t], b_val[t] = a, dist.log_prob(a), value[0]

                nobs, rew, done, ts, infos = env.step(a.cpu().numpy())
                obs = torch.as_tensor(nobs, device=device)
                trial_start = torch.as_tensor(ts.astype(np.float32), device=device)
                b_rew[t] = torch.as_tensor(rew, device=device)
                b_done[t] = torch.as_tensor(done.astype(np.float32), device=device)
                for info in infos:
                    prog_hist[min(info["attempt"], MAX_ATTEMPT_BUCKETS - 1)].append(
                        info["progress"]
                    )
                    if "trial" in info:
                        tr = info["trial"]
                        trial_best.append(tr["best_progress"])
                        trial_sum.append(tr["sum_progress"])
                        trial_ret.append(tr["return"])
                        trial_att.append(tr["attempts"])
            _, last_val, _ = net(obs_norm(obs).unsqueeze(0), h, trial_start.unsqueeze(0))
            last_val = last_val[0]

        obs_norm.update(b_obs.reshape(-1, env.obs_dim))
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, args.gamma, args.lam)
        norm_obs = obs_norm(b_obs)
        global_step += N * T

        # PPO по последовательностям: минибатч = подмножество сред, целиком по T
        net.train()
        env_ids = np.arange(N)
        mb = N // args.minibatches
        stats = {}
        for _ in range(args.epochs):
            np.random.shuffle(env_ids)
            for s in range(0, N, mb):
                idx = torch.as_tensor(env_ids[s : s + mb], device=device)
                logits, value, _ = net(norm_obs[:, idx], h_start[:, idx], b_ts[:, idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[:, idx])
                ratio = (logp - b_logp[:, idx]).exp()
                a_mb = adv[:, idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - args.clip, 1 + args.clip))
                v_loss = F.mse_loss(value, ret[:, idx])
                entropy = dist.entropy().mean()
                loss = pg.mean() + args.vf * v_loss - ent_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                opt.step()
                stats = {
                    "pg": pg.mean().item(),
                    "vf": v_loss.item(),
                    "entropy": entropy.item(),
                    "kl": (b_logp[:, idx] - logp).detach().mean().item(),
                }
        h = h.detach()

        if update % 10 == 0:
            row = {
                "update": update,
                "step": global_step,
                "sps": int(global_step / max(1e-6, time.time() - t0)),
                # прогресс по номеру попытки внутри трайла: должен расти слева направо
                "prog_by_attempt": [
                    round(float(np.mean(r)), 3) if r else None for r in prog_hist
                ],
                "trial_best_m": round(float(np.mean(trial_best)) * 800, 1) if trial_best else None,
                "trial_sum_m": round(float(np.mean(trial_sum)) * 800, 1) if trial_sum else None,
                "trial_return": round(float(np.mean(trial_ret)), 1) if trial_ret else None,
                "attempts": round(float(np.mean(trial_att)), 2) if trial_att else None,
                **{k: round(v, 4) for k, v in stats.items()},
            }
            print(json.dumps(row), flush=True)
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
        if update % 100 == 0 or update == n_updates - 1:
            torch.save(
                {
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "obs_norm": obs_norm.state_dict(),
                    "update": update,
                    "args": vars(args),
                },
                out_dir / "ckpt.pt",
            )


if __name__ == "__main__":
    main()
