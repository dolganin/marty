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

BUTTONS = (
    ("Gas", 1 << 0), ("Reverse", (1 << 0) | (1 << 2)), ("Brake", 1 << 1),
    ("Clutch", 1 << 3), ("Tilt left", 1 << 4), ("Tilt right", 1 << 5),
    ("Gear +", 1 << 6), ("Gear -", 1 << 7), ("Drive", 1 << 8),
    ("Ignition", 1 << 9), ("Charge", 1 << 10), ("Lidar", 1 << 11),
    ("Heater", 1 << 12), ("Jump", 1 << 13), ("Climb", 1 << 14),
    ("Propeller", 1 << 15), ("Blow", 1 << 21), ("Flood", 1 << 22),
    ("Thruster", 1 << 23),
)

GAS = 1 << 0
REVERSE = (1 << 0) | (1 << 2)
LATCHED_BITS = {GAS, REVERSE, 1 << 1, 1 << 3, 1 << 4, 1 << 5, 1 << 12, 1 << 21, 1 << 22, 1 << 23}


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
        self.sidebar = tk.Frame(self.root)
        self.sidebar.pack(side="right", fill="y")
        self.telemetry = tk.Label(
            self.sidebar, width=44, justify="left", anchor="nw", font=("Consolas", 10)
        )
        self.telemetry.pack(fill="x")
        self.controls = tk.Frame(self.sidebar)
        self.controls.pack(fill="x", padx=6, pady=6)
        self.keys: set[str] = set()
        self.latched_action = 0
        self.pulse_action = 0
        self.buttons: dict[int, tk.Button] = {}
        for index, (label, bit) in enumerate(BUTTONS):
            button = tk.Button(
                self.controls, text=label, width=12,
                command=lambda control_bit=bit: self._toggle(control_bit),
            )
            button.grid(row=index // 2, column=index % 2, padx=2, pady=2, sticky="ew")
            self.buttons[bit] = button
        tk.Button(self.controls, text="Reset", width=12, command=self._reset).grid(
            row=10, column=0, padx=2, pady=2, sticky="ew"
        )
        tk.Button(self.controls, text="New track", width=12, command=self._new_track).grid(
            row=10, column=1, padx=2, pady=2, sticky="ew"
        )
        self.photo = None
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.after(0, self._tick)

    def _press(self, event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self.root.destroy()
        elif key == "t":
            self._new_track()
        else:
            self.keys.add(key)

    def _release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def _toggle(self, bit: int) -> None:
        if bit not in LATCHED_BITS:
            self.pulse_action |= bit
        elif bit == GAS and (self.latched_action & REVERSE) == REVERSE:
            self.latched_action &= ~(1 << 2)
        elif bit == REVERSE:
            if (self.latched_action & REVERSE) == REVERSE:
                self.latched_action &= ~REVERSE
            else:
                self.latched_action = (self.latched_action & ~REVERSE) | REVERSE
        else:
            self.latched_action ^= bit
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        for bit, button in self.buttons.items():
            active = bit in LATCHED_BITS and (self.latched_action & bit) == bit
            button.configure(
                relief="sunken" if active else "raised",
                bg="#8ccf7e" if active else "#f0f0f0",
            )

    def _reset(self) -> None:
        self.latched_action = 0
        self.pulse_action = 0
        self._refresh_buttons()
        self.env.reset(seed=self.seed)

    def _new_track(self) -> None:
        self.seed = random.randrange(2**31)
        self._reset()

    def _tick(self) -> None:
        action = self.latched_action | self.pulse_action
        for key in self.keys:
            action |= KEY_BITS.get(key, 0)
        _, _, terminated, truncated, _ = self.env.step(action)
        self.pulse_action = 0
        if terminated or truncated:
            self._new_track()
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
            ("charge", f"{debug.get('solar_charge_rate', 0.0):.2f}/s"),
            ("panel", f"{debug.get('solar_panel_deployment', 0.0) * 100:.0f}%"),
            ("current", f"{debug.get('water_current_x', 0.0):.1f} m/s"),
            ("gear", debug.get("gear")),
            ("rpm", f"{debug.get('engine_rpm', 0.0):.0f}"),
            ("slip", f"{debug.get('drivetrain_slip', 0.0):.2f}"),
            ("lidar", f"{debug.get('lidar_range', 0.0):.1f} m"),
            ("drive", debug.get("drive_layout")),
            ("propeller", f"{debug.get('propeller_thrust', 0.0):.1f} N"),
            ("thruster", f"{debug.get('thruster_thrust', 0.0):.1f} N"),
            ("climb", "on" if debug.get("climb_mode") else "off"),
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
