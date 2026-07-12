from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

from mars_rover_env import MarsRoverEnv


def _ppm_bytes(rgb) -> bytes:
    h, w, _ = rgb.shape
    return f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()


class ManualPlayer:
    def __init__(self, args: argparse.Namespace):
        self.env = MarsRoverEnv(
            config_path=args.config,
            rig_path=args.rig,
            render_mode="debug_rgb_array" if args.debug else "rgb_array",
            render_width=args.width,
            render_height=args.height,
        )
        self.seed = args.seed
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.keys: set[str] = set()
        self.last_time = time.perf_counter()
        self.frame_ms = max(1, int(1000 / args.fps))

        self.root = tk.Tk()
        self.root.title("Mars Rover Manual Control")
        self.canvas = tk.Canvas(self.root, width=args.width, height=args.height, highlightthickness=0)
        self.canvas.pack()
        self.status = tk.Label(self.root, anchor="w", justify="left")
        self.status.pack(fill="x")
        self.photo = None
        self.image_id = None

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

    def on_key_press(self, event) -> None:
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "r":
            self.seed += 1
            self.obs, self.info = self.env.reset(seed=self.seed, options={"trial_start": True})
        elif key == "n":
            self.seed += 1
            self.obs, self.info = self.env.reset(seed=self.seed, options={"trial_start": False})
        elif key in {"escape", "q"}:
            self.root.destroy()

    def on_key_release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def action(self) -> int:
        gas = "right" in self.keys or "d" in self.keys
        reverse = "left" in self.keys or "a" in self.keys
        brake = "down" in self.keys or "s" in self.keys or "space" in self.keys
        tilt_left = "j" in self.keys
        tilt_right = "l" in self.keys
        if brake:
            return 2
        if reverse:
            return 3
        if gas and tilt_left:
            return 4
        if gas and tilt_right:
            return 5
        if gas:
            return 1
        return 0

    def update(self) -> None:
        obs, reward, terminated, truncated, _ = self.env.step(self.action())
        self.obs = obs
        if terminated or truncated:
            self.seed += 1
            self.obs, self.info = self.env.reset(seed=self.seed)

        rgb = self.env.render()
        self.photo = tk.PhotoImage(data=_ppm_bytes(rgb), format="PPM")
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self.image_id, image=self.photo)

        debug = self.env.debug_info()
        self.status.configure(
            text=(
                "Controls: Right/D gas, Left/A reverse, Down/S/Space brake, J/L tilt, "
                "R new trial, N same mechanics, Q/Esc quit\n"
                f"mechanic={debug['mechanic']} x={debug['x']:.2f} vx={debug['vx']:.2f} "
                f"energy={debug['energy']:.3f} damage={debug['damage']:.3f} reward={reward:.3f}"
            )
        )
        self.root.after(self.frame_ms, self.update)

    def run(self) -> None:
        self.update()
        self.root.mainloop()


def main() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_dir / "play.yaml"))
    parser.add_argument("--rig", default=str(config_dir / "rover_rig.yaml"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ManualPlayer(args).run()


if __name__ == "__main__":
    main()
