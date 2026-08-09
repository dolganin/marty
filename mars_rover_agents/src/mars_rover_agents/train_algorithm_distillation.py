"""Train the Algorithm Distillation model on collected learning histories.

Pure sequence modelling: predict the action taken, given everything seen so far in the
context. No RL, no reward maximisation, no recurrent state carried by the training loop —
if adaptation appears at evaluation time it can only have come from reading the context.

Context length is the whole point. The histories are five improving stages concatenated,
so the memory window must span more than one stage or the model can never observe that
improvement is happening; memory_len is therefore set far above the RL2 default.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from .rl2_model import RL2TransformerActorCritic, rl2_features


def load_histories(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(directory / "*.npz"))):
        payload = np.load(path)
        rows.append(
            {
                "observations": payload["observations"],
                "actions": payload["actions"],
                "rewards": payload["rewards"],
            }
        )
    if not rows:
        raise SystemExit(f"no histories found in {directory}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Algorithm Distillation")
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--memory-len", type=int, default=2048)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    histories = load_histories(args.histories)
    observation_dim = int(histories[0]["observations"].shape[1])
    model = RL2TransformerActorCritic(
        observation_dim=observation_dim,
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        memory_len=args.memory_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)

    lengths = [len(row["actions"]) for row in histories]
    horizon = min(lengths)
    print(
        f"{len(histories)} histories, horizon {horizon}, memory_len {args.memory_len}",
        flush=True,
    )

    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(histories))
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_count = 0.0
        for begin in range(0, len(order), args.batch_size):
            batch_indices = order[begin : begin + args.batch_size]
            if len(batch_indices) < 2:
                continue
            observations = np.stack(
                [histories[i]["observations"][:horizon] for i in batch_indices], axis=0
            )
            actions = np.stack(
                [histories[i]["actions"][:horizon] for i in batch_indices], axis=0
            )
            rewards = np.stack(
                [histories[i]["rewards"][:horizon] for i in batch_indices], axis=0
            )
            previous_actions = np.concatenate(
                [np.full((len(batch_indices), 1), -1, dtype=np.int64), actions[:, :-1]], axis=1
            )
            previous_rewards = np.concatenate(
                [np.zeros((len(batch_indices), 1), dtype=np.float32), rewards[:, :-1]], axis=1
            )
            previous_dones = np.zeros_like(previous_rewards, dtype=bool)
            previous_dones[:, 0] = True

            state = model.initial_state(len(batch_indices), device)
            optimizer.zero_grad(set_to_none=True)
            total_weight = float(horizon)
            for start in range(0, horizon, args.chunk_length):
                stop = min(horizon, start + args.chunk_length)
                features = rl2_features(
                    observations[:, start:stop].reshape(-1, observation_dim),
                    previous_actions[:, start:stop].reshape(-1),
                    previous_rewards[:, start:stop].reshape(-1),
                    previous_dones[:, start:stop].reshape(-1),
                    device=device,
                ).view(len(batch_indices), stop - start, -1)
                logits, _, state = model.forward_chunk(features, state, detach_memory=True)
                targets = torch.as_tensor(
                    actions[:, start:stop], dtype=torch.long, device=device
                )
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
                (loss * ((stop - start) / total_weight)).backward()
                with torch.no_grad():
                    epoch_correct += float(
                        (logits.argmax(dim=-1) == targets).float().sum()
                    )
                    epoch_count += targets.numel()
                    epoch_loss += float(loss) * (stop - start)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss / max(horizon, 1):.4f} "
            f"action_accuracy={epoch_correct / max(epoch_count, 1.0):.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "artifact_type": "algorithm_distillation",
            "histories": len(histories),
            "horizon": horizon,
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
