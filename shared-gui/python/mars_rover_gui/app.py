from __future__ import annotations

import argparse
import random
import tkinter as tk

from PIL import Image, ImageTk
from _mars_rover_cpp import biome_catalog
from mars_rover_env import MarsRoverEnv


KEY_BITS = {
    "d": 1 << 0,
    "a": (1 << 0) | (1 << 2),
    "space": 1 << 1,
    "c": 1 << 3,
    "j": 1 << 4,
    "l": 1 << 5,
    "x": 1 << 6,
    "z": 1 << 7,
    "v": 1 << 8,
    "e": 1 << 9,
    "f": 1 << 10,
    "g": 1 << 11,
    "h": 1 << 12,
    "k": 1 << 13,
    "b": 1 << 14,
    "p": 1 << 15,
    "u": 1 << 21,
    "i": 1 << 22,
    "r": 1 << 23,
}


class Player:
    def __init__(self, args: argparse.Namespace):
        self.env = MarsRoverEnv(
            config_path=args.config,
            rig_path=args.rig,
            fixed_biome_id=args.biome_index,
            render_mode="debug_rgb_array" if args.debug else "rgb_array",
            render_width=args.width,
            render_height=args.height,
        )
        self.seed = args.seed
        self.env.reset(seed=self.seed)
        self.root = tk.Tk()
        self.root.title("Mars Rover")
        self.canvas = tk.Label(self.root)
        self.canvas.pack(side="left")
        self.telemetry = tk.Label(
            self.root, width=42, justify="left", anchor="nw", font=("Consolas", 10)
        )
        self.telemetry.pack(side="right", fill="y")
        self.keys: set[str] = set()
        self.photo = None
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.after(0, self._tick)

    def _press(self, event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self.root.destroy()
        elif key == "t":
            self.seed = random.randrange(2**31)
            self.env.reset(seed=self.seed)
        else:
            self.keys.add(key)

    def _release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def _tick(self) -> None:
        action = 0
        for key in self.keys:
            action |= KEY_BITS.get(key, 0)
        _, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.seed = random.randrange(2**31)
            self.env.reset(seed=self.seed)
        frame = self.env.render()
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=self.photo)
        debug = self.env.debug_info()
        fields = (
            ("seed", self.seed),
            ("biome", debug.get("mechanic")),
            ("distance", f"{debug.get('x', 0.0):.1f} m"),
            ("speed", f"{debug.get('speed_kmh', 0.0):.1f} km/h"),
            ("energy", f"{debug.get('energy', 0.0):.1f}"),
            ("gear", debug.get("gear")),
            ("rpm", f"{debug.get('engine_rpm', 0.0):.0f}"),
            ("slip", f"{debug.get('drivetrain_slip', 0.0):.2f}"),
            ("lidar", f"{debug.get('lidar_range', 0.0):.1f} m"),
        )
        self.telemetry.configure(text="\n".join(f"{key:>10}: {value}" for key, value in fields))
        self.root.after(16, self._tick)

    def run(self) -> None:
        self.root.mainloop()
        self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--rig")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--biome")
    parser.add_argument("--list-biomes", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    catalog = [dict(item) for item in biome_catalog()]
    if args.list_biomes:
        for item in catalog:
            print(item["id"])
        return
    args.biome_index = None
    if args.biome:
        match = next((item for item in catalog if item["id"] == args.biome), None)
        if match is None:
            parser.error(f"unknown biome: {args.biome}")
        args.biome_index = int(match["index"])
    Player(args).run()


if __name__ == "__main__":
    main()
