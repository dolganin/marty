from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from meta_ppo import Agent, Config, RunningMeanStd, Runner, device_for, summary, tt


@torch.no_grad()
def record(checkpoint: Path, output: Path, seed: int, split: int, device_name: str) -> dict:
    device = device_for(device_name)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = Config(**ckpt["config"])
    runner = Runner(1, split, seed, config.frame_skip, config.gamma,
                    config.frontier_bonus_scale)
    model = Agent(ckpt["obs_dim"], ckpt["action_dim"], config.hidden_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rms = RunningMeanStd(ckpt["obs_dim"])
    rms.load_state_dict(ckpt["rms"])
    hidden = model.initial(1, device)
    attempts, trials = [], []
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    physics_frames = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        output,
        fps=30,
        codec="libx264",
        quality=7,
        macro_block_size=None,
    ) as writer:
        runner.env.core.render_rgb(0, rgb, 640, 360, True)
        writer.append_data(rgb)

        def capture(active_runner: Runner) -> None:
            nonlocal physics_frames
            physics_frames += 1
            if physics_frames % 2 == 0:
                active_runner.env.core.render_rgb(0, rgb, 640, 360, True)
                writer.append_data(rgb)

        while not trials:
            logits, _, hidden = model.step(
                tt(rms.normalize(runner.obs), device),
                tt(runner.prev_action, device, torch.long),
                tt(runner.prev_reward, device),
                tt(runner.prev_done, device),
                tt(runner.trial_fraction, device),
                tt(runner.start, device),
                hidden,
            )
            runner.step(logits.argmax(-1).cpu().numpy(), frame_callback=capture)
            attempts.extend(runner.pop())
            trials.extend(runner.pop_trials())

    report = {"seed": seed, "split": split, "physics_frames": physics_frames,
              "video": str(output.resolve()), **summary(trials, attempts)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--biome-split", type=int, default=2, choices=(1, 2))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(record(args.checkpoint, args.output, args.seed,
                            args.biome_split, args.device), indent=2))


if __name__ == "__main__":
    main()
