from __future__ import annotations

import argparse
import random
import time
from _mars_rover_cpp import biome_catalog


GAS = 1 << 0
BRAKE = 1 << 1
REVERSE = 1 << 2
CLUTCH = 1 << 3
TILT_LEFT = 1 << 4
TILT_RIGHT = 1 << 5
GEAR_UP = 1 << 6
GEAR_DOWN = 1 << 7
DRIVE = 1 << 8
IGNITION = 1 << 9
CHARGE = 1 << 10
LIDAR = 1 << 11
HEATER = 1 << 12
JUMP = 1 << 13
CLIMB = 1 << 14
PROPELLER = 1 << 15
JUMP_FRONT = 1 << 17
JUMP_REAR = 1 << 18
PISTON_FRONT = 1 << 19
PISTON_REAR = 1 << 20
BALLAST_BLOW = 1 << 21
BALLAST_FLOOD = 1 << 22
THRUSTER = 1 << 23

HELD_CONTROLS = (
    ("gas", "D / →\nGAS", GAS),
    ("reverse", "A / ←\nREVERSE", GAS | REVERSE),
    ("brake", "S / ↓\nBRAKE", BRAKE),
    ("clutch", "C / SHIFT\nCLUTCH", CLUTCH),
    ("tilt_left", "J\nTILT LEFT", TILT_LEFT),
    ("tilt_right", "L\nTILT RIGHT", TILT_RIGHT),
    ("heater", "H\nENGINE HEAT", HEATER),
    ("jump", "K\nJUMP BOTH", JUMP),
    ("jump_front", "CTRL+K\nJUMP FRONT", JUMP_FRONT),
    ("jump_rear", "ALT+K\nJUMP REAR", JUMP_REAR),
    ("piston", "I\nPUSH BOTH", PISTON_FRONT | PISTON_REAR),
    ("piston_front", "CTRL+I\nPUSH FRONT", PISTON_FRONT),
    ("piston_rear", "ALT+I\nPUSH REAR", PISTON_REAR),
    ("blow", "Y\nBALLAST BLOW", BALLAST_BLOW),
    ("flood", "U\nBALLAST FLOOD", BALLAST_FLOOD),
    ("thruster", "SPACE\nROCKET", THRUSTER),
)

PULSE_CONTROLS = (
    ("gear_down", "Z\nGEAR DOWN", GEAR_DOWN),
    ("gear_up", "X\nGEAR UP", GEAR_UP),
    ("drive", "V\nRWD/FWD/AWD", DRIVE),
    ("ignition", "E\nIGNITION", IGNITION),
    ("charge", "F\nSOLAR PANEL", CHARGE),
    ("lidar", "G\nLIDAR", LIDAR),
    ("climb", "B\nCLIMB MODE", CLIMB),
    ("propeller", "P\nPROPELLER", PROPELLER),
)


class OrganizerPlayer:
    def __init__(self, args: argparse.Namespace):
        global tk, Image, ImageTk
        import tkinter as tk
        from PIL import Image, ImageTk
        from mars_rover_env import MarsRoverEnv

        self.root = tk.Tk()
        self.root.title("Mars Rover Manual Control")
        self.fullscreen = bool(args.fullscreen)
        self.sidebar_width = 500
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
            render_width = max(640, self.root.winfo_screenwidth() - self.sidebar_width - 16)
            render_height = max(360, self.root.winfo_screenheight() - 16)
        else:
            render_width = args.width
            render_height = args.height
        self.env = MarsRoverEnv(
            config_path=args.config,
            rig_path=args.rig,
            fixed_biome_id=args.biome_index,
            render_mode="debug_rgb_array" if args.debug else "rgb_array",
            render_width=render_width,
            render_height=render_height,
        )
        self.seed = args.seed
        self.env.reset(seed=self.seed)
        self.frame_ms = max(1, round(1000 / max(1, args.fps)))
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.last_reward = 0.0
        self.notice = ""
        self.keys: set[str] = set()
        self.mouse_action = 0
        self.pulse_action = 0
        self.current_action = 0
        self.photo = None
        self.canvas = tk.Label(self.root, bg="#000000", borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sidebar = tk.Frame(
            self.root, width=self.sidebar_width, bg="#171717", padx=10, pady=10
        )
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(False)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        tk.Label(
            self.sidebar,
            text="ROVER TELEMETRY",
            bg="#171717",
            fg="#ffffff",
            font=("Consolas", 14, "bold"),
        ).pack(anchor="w", pady=(0, 6))
        self.telemetry = tk.Label(
            self.sidebar,
            justify="left",
            anchor="nw",
            bg="#171717",
            fg="#f4f4f4",
            font=("Consolas", 9, "bold"),
        )
        self.telemetry.pack(fill="x")
        tk.Label(
            self.sidebar,
            text="CONTROLS",
            bg="#171717",
            fg="#ffffff",
            font=("Consolas", 12, "bold"),
        ).pack(anchor="w", pady=(8, 4))
        controls = tk.Frame(self.sidebar, bg="#171717")
        controls.pack(fill="x")
        self.control_buttons: dict[str, tuple[tk.Button, int]] = {}
        for index, (name, label, bit) in enumerate(HELD_CONTROLS + PULSE_CONTROLS):
            button = tk.Button(
                controls,
                text=label,
                bg="#252525",
                fg="#eeeeee",
                activebackground="#287a3d",
                activeforeground="#ffffff",
                font=("Consolas", 8, "bold"),
                relief="flat",
                borderwidth=1,
                height=2,
                takefocus=False,
            )
            button.grid(row=index // 3, column=index % 3, sticky="nsew", padx=1, pady=1)
            if (name, label, bit) in HELD_CONTROLS:
                button.bind("<ButtonPress-1>", lambda event, value=bit: self._hold(value, True))
                button.bind("<ButtonRelease-1>", lambda event, value=bit: self._hold(value, False))
                button.bind("<Leave>", lambda event, value=bit: self._hold(value, False))
            else:
                button.configure(command=lambda value=bit: self._pulse(value))
            self.control_buttons[name] = (button, bit)
        for column in range(3):
            controls.grid_columnconfigure(column, weight=1)
        session = tk.Frame(self.sidebar, bg="#171717")
        session.pack(fill="x", pady=(5, 0))
        tk.Button(
            session,
            text="R  RESET SAME WORLD",
            command=self._reset,
            bg="#3b3030",
            fg="#ffffff",
            font=("Consolas", 9, "bold"),
            takefocus=False,
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tk.Button(
            session,
            text="T  RANDOM NEW WORLD",
            command=self._new_track,
            bg="#303b46",
            fg="#ffffff",
            font=("Consolas", 9, "bold"),
            takefocus=False,
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.status = tk.Label(
            self.sidebar,
            justify="left",
            anchor="sw",
            wraplength=self.sidebar_width - 20,
            bg="#171717",
            fg="#aaaaaa",
            font=("Consolas", 8),
        )
        self.status.pack(side="bottom", fill="x")
        self.root.bind_all("<KeyPress>", self._press)
        self.root.bind_all("<KeyRelease>", self._release)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after_idle(self.root.focus_force)
        self.root.after(0, self._tick)

    def _hold(self, bit: int, active: bool) -> None:
        if active:
            self.mouse_action |= bit
        else:
            self.mouse_action &= ~bit

    def _pulse(self, bit: int) -> None:
        self.pulse_action |= bit

    def _press(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys:
            return
        self.keys.add(key)
        pulses = {
            "z": GEAR_DOWN,
            "x": GEAR_UP,
            "v": DRIVE,
            "e": IGNITION,
            "f": CHARGE,
            "g": LIDAR,
            "b": CLIMB,
            "p": PROPELLER,
        }
        if key in pulses:
            self._pulse(pulses[key])
        elif key == "r":
            self._reset()
        elif key == "t":
            self._new_track()
        elif key == "f11":
            self.fullscreen = not self.fullscreen
            self.root.attributes("-fullscreen", self.fullscreen)
        elif key in {"escape", "q"}:
            self._close()

    def _release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def _keyboard_action(self) -> int:
        action = 0
        if {"d", "right"} & self.keys:
            action |= GAS
        if {"a", "left"} & self.keys:
            action |= GAS | REVERSE
        if {"s", "down"} & self.keys:
            action |= BRAKE
        if {"c", "shift_l", "shift_r"} & self.keys:
            action |= CLUTCH
        if "j" in self.keys:
            action |= TILT_LEFT
        if "l" in self.keys:
            action |= TILT_RIGHT
        if "h" in self.keys:
            action |= HEATER
        ctrl = bool({"control_l", "control_r"} & self.keys)
        alt = bool({"alt_l", "alt_r"} & self.keys)
        if "k" in self.keys:
            action |= JUMP_FRONT if ctrl else JUMP_REAR if alt else JUMP
        if "i" in self.keys:
            action |= PISTON_FRONT if ctrl else PISTON_REAR if alt else PISTON_FRONT | PISTON_REAR
        if "y" in self.keys:
            action |= BALLAST_BLOW
        if "u" in self.keys:
            action |= BALLAST_FLOOD
        if "space" in self.keys:
            action |= THRUSTER
        return action

    def _reset(self) -> None:
        self.keys.clear()
        self.mouse_action = 0
        self.pulse_action = 0
        self.env.reset(seed=self.seed, options={"trial_start": False})
        self.notice = f"RESET seed={self.seed}"

    def _new_track(self) -> None:
        self.seed = random.randrange(2**31)
        self.keys.clear()
        self.mouse_action = 0
        self.pulse_action = 0
        self.env.reset(seed=self.seed, options={"trial_start": True})
        self.notice = f"NEW WORLD seed={self.seed}"

    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        if elapsed > 0.0:
            instant = 1.0 / elapsed
            self.fps = instant if self.fps == 0.0 else self.fps * 0.9 + instant * 0.1
        self.current_action = self._keyboard_action() | self.mouse_action | self.pulse_action
        _, reward, terminated, truncated, _ = self.env.step(self.current_action)
        self.last_reward = reward
        self.pulse_action = 0
        debug = self.env.debug_info()
        if terminated or truncated:
            self.notice = f"RUN ENDED reason={debug.get('termination_reason', 0)}  R OR T TO CONTINUE"
        frame = self.env.render()
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=self.photo)
        self._update_hud(debug)
        self._update_buttons(debug)
        self.root.after(self.frame_ms, self._tick)

    def _update_hud(self, debug: dict) -> None:
        energy = debug.get("energy", 0.0)
        capacity = max(1.0, debug.get("energy_capacity", 1.0))
        engine = "RUNNING" if debug.get("engine_running") else "OFF"
        if debug.get("engine_overheated"):
            engine = "OVERHEATED"
        elif debug.get("engine_cold_locked"):
            engine = "TOO COLD"
        jump_phase = {0: "READY", 1: "PRELOAD", 2: "LAUNCH"}.get(
            debug.get("suspension_jump_phase", 0), "READY"
        )
        contacts = "".join(
            name if debug.get(field) else "-"
            for name, field in (
                ("F", "body_contact_front"),
                ("B", "body_contact_belly"),
                ("R", "body_contact_rear"),
                ("T", "body_contact_roof"),
            )
        )
        if debug.get("lidar_landing_valid"):
            drop = debug.get("lidar_landing_x", 0.0) - debug.get("x", 0.0)
            landing = f"{drop:+6.1f}m   h {debug.get('lidar_landing_y', 0.0):+6.1f}m"
        else:
            landing = "   --"
        lines = (
            f"FPS {self.fps:5.1f}   SEED {self.seed}",
            f"TIME {debug.get('trial_time_left', 0.0):6.1f}s   BIOME {debug.get('mechanic', '-')}",
            f"DIST {debug.get('x', 0.0):7.1f}m   SPEED {debug.get('speed_kmh', 0.0):5.1f} km/h",
            f"BATTERY {energy / capacity * 100:5.1f}%   NET {debug.get('energy_gain_rate', 0.0) - debug.get('energy_cost_rate', 0.0):+.2f}/s",
            f"ENGINE {engine}   RPM {debug.get('engine_rpm', 0.0):5.0f}",
            f"TEMP {debug.get('engine_temperature', 0.0):5.1f} C   ENV {debug.get('ambient_temperature', 0.0):5.1f} C",
            f"GEAR {debug.get('gear', '-')} / {debug.get('gear_count', '-')}   DRIVE {debug.get('drive_layout', '-')}",
            f"CLUTCH {debug.get('clutch_engagement', 0.0) * 100:3.0f}%   SLIP {debug.get('drivetrain_slip', 0.0):.2f}",
            f"SPRING {jump_phase} {debug.get('suspension_jump_charge', 0.0) * 100:3.0f}%",
            f"PISTON {debug.get('roof_piston_extension', 0.0) * 100:3.0f}%   CONTACT {contacts}",
            f"SOLAR {debug.get('solar_panel_deployment', 0.0) * 100:3.0f}%   +{debug.get('solar_charge_rate', 0.0):.2f}/s",
            f"LIDAR {debug.get('lidar_range', 0.0):5.1f}m   CURRENT {debug.get('water_current_x', 0.0):+.2f}m/s",
            f"LANDING {landing}",
            f"PROPELLER {debug.get('propeller_thrust', 0.0):6.1f}N   ROCKET {debug.get('thruster_thrust', 0.0):6.1f}N",
            f"GRAVITY {debug.get('gravity', -3.71):+.2f}m/s²   DAMAGE {debug.get('damage', 0.0):.3f}",
        )
        self.telemetry.configure(text="\n".join(lines))
        self.status.configure(text=f"reward={self.last_reward:.3f}   {self.notice}")

    def _update_buttons(self, debug: dict) -> None:
        for name, (button, bit) in self.control_buttons.items():
            active = (self.current_action & bit) == bit
            if name == "climb":
                active = bool(debug.get("climb_mode"))
            elif name == "propeller":
                active = bool(debug.get("propeller_mode"))
            elif name == "charge":
                active = bool(debug.get("solar_panel_requested"))
            elif name == "drive":
                layout = debug.get("drive_layout", "RWD")
                button.configure(text=f"V\n{layout}")
                active = layout != "RWD"
            elif name == "ignition":
                active = bool(debug.get("engine_running"))
            elif name == "heater":
                active = bool(debug.get("heater_active"))
            elif name == "lidar":
                active = bool(debug.get("lidar_active"))
            button.configure(bg="#287a3d" if active else "#252525")

    def _close(self) -> None:
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--rig")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--biome")
    parser.add_argument("--list-biomes", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    catalog = [dict(item) for item in biome_catalog()]
    if args.list_biomes:
        for item in catalog:
            print(item["id"])
        return
    if args.seed is None:
        args.seed = random.randrange(2**31)
    args.biome_index = None
    if args.biome:
        match = next((item for item in catalog if item["id"] == args.biome), None)
        if match is None:
            parser.error(f"unknown biome: {args.biome}")
        args.biome_index = int(match["index"])
    OrganizerPlayer(args).run()


if __name__ == "__main__":
    main()
