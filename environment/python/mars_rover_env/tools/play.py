from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from mars_rover_env import MarsRoverEnv


def _ppm_bytes(rgb) -> bytes:
    h, w, _ = rgb.shape
    return f"P6 {w} {h} 255\n".encode("ascii") + rgb.tobytes()


class ManualPlayer:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Mars Rover Manual Control")
        self._fullscreen = bool(args.fullscreen)
        self._sidebar_width = 500
        if self._fullscreen:
            self.root.attributes("-fullscreen", True)
            render_width = max(640, self.root.winfo_screenwidth() - self._sidebar_width - 16)
            render_height = max(360, self.root.winfo_screenheight() - 16)
        else:
            render_width = int(args.width)
            render_height = int(args.height)

        self.env = MarsRoverEnv(
            config_path=args.config,
            rig_path=args.rig,
            biome_split={"train": 1, "test": 2}[args.split],
            render_mode="debug_rgb_array" if args.debug else "rgb_array",
            render_width=render_width,
            render_height=render_height,
        )
        self.seed = args.seed
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.keys: set[str] = set()
        self.last_time = time.perf_counter()
        self.fps_value = 0.0
        self.frame_ms = max(1, int(1000 / args.fps))
        self.restart_notice = ""
        self.last_reward = 0.0

        self.canvas = tk.Canvas(self.root, width=render_width, height=render_height,
                                highlightthickness=0, bg="#000000")
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar = tk.Frame(self.root, width=self._sidebar_width, bg="#171717", padx=10, pady=10)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        tk.Label(sidebar, text="ROVER TELEMETRY", bg="#171717", fg="#ffffff",
                 font=("Consolas", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.hud_labels: list[tk.Label] = []
        for _ in range(24):
            label = tk.Label(sidebar, anchor="w", justify="left", bg="#171717", fg="#f4f4f4",
                             font=("Consolas", 9, "bold"))
            label.pack(fill="x", pady=0)
            self.hud_labels.append(label)

        tk.Label(sidebar, text="CONTROLS", bg="#171717", fg="#ffffff",
                 font=("Consolas", 12, "bold")).pack(anchor="w", pady=(12, 5))
        controls = tk.Frame(sidebar, bg="#171717")
        controls.pack(fill="x")
        self.control_labels: dict[str, tk.Label] = {}
        for index, (name, text) in enumerate((
            ("gas", "D / →  GAS"),
            ("brake", "S / ↓  BRAKE"),
            ("clutch", "C / SHIFT  CLUTCH"),
            ("down", "Z  GEAR DOWN"),
            ("up", "X  GEAR UP"),
            ("ignition", "E  IGNITION"),
            ("heater", "H  ENGINE HEAT"),
            ("solar", "F  SOLAR PANEL"),
            ("lidar", "G  FORWARD LONG-RANGE LIDAR"),
            ("drive", "V  RWD / FWD / AWD"),
            ("jump", "K  SPRING JUMP (+CTRL FRONT, +ALT REAR)"),
            ("piston", "I  ROOF PISTON (+CTRL FRONT, +ALT REAR)"),
            ("climb", "B  CLIMB MODE"),
            ("propeller", "P  PROPELLER"),
            ("ballast_up", "Y  BLOW BALLAST / FLOAT"),
            ("ballast_down", "U  FLOOD BALLAST / SINK"),
            ("restart", "R  RESET SAME WORLD"),
            ("randomize", "T  RANDOM NEW WORLD"),
        )):
            label = tk.Label(controls, text=text, bg="#252525", fg="#eeeeee",
                             font=("Consolas", 8, "bold"), padx=3, pady=2)
            label.grid(row=index // 3, column=index % 3, sticky="ew", padx=1, pady=1)
            self.control_labels[name] = label
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(2, weight=1)
        self.status = tk.Label(sidebar, anchor="w", justify="left", wraplength=self._sidebar_width - 20,
                               bg="#171717", fg="#aaaaaa", font=("Consolas", 9))
        self.status.pack(side="bottom", fill="x", pady=(4, 0))
        self.photo = None
        self.image_id = None

        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)
        self.root.after_idle(self.root.focus_force)

    def on_key_press(self, event) -> None:
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "f11":
            self.toggle_fullscreen()
        elif key == "r":
            self.restart_notice = "RESTART: MANUAL RESET; SAME TRIAL"
            self.obs, self.info = self.env.reset(seed=self.seed, options={"trial_start": False})
            self.last_reward = 0.0
        elif key == "t":
            # Reroll into a brand new random world (fresh seed, trial boundary
            # so hidden mechanics resample too) without leaving the debugger.
            self.seed = random.randint(0, 2**31 - 1)
            self.restart_notice = f"RANDOMIZED WORLD: seed={self.seed}"
            self.obs, self.info = self.env.reset(seed=self.seed, options={"trial_start": True})
            self.last_reward = 0.0
        elif key in {"escape", "q"}:
            self.root.destroy()

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def on_key_release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def action(self) -> int:
        gas = "right" in self.keys or "d" in self.keys
        reverse = "left" in self.keys or "a" in self.keys
        brake = "down" in self.keys or "s" in self.keys or "space" in self.keys
        clutch = "c" in self.keys or "shift_l" in self.keys or "shift_r" in self.keys
        tilt_left = "j" in self.keys
        tilt_right = "l" in self.keys
        shift_down = "z" in self.keys
        shift_up = "x" in self.keys
        toggle_drive = "v" in self.keys
        ignition = "e" in self.keys
        heater = "h" in self.keys
        toggle_charge = "f" in self.keys
        climb = "b" in self.keys
        propeller = "p" in self.keys
        ballast_up = "y" in self.keys
        ballast_down = "u" in self.keys

        # Jump and piston use modifiers to select an edge. Lidar deliberately
        # has one direction only: forward along the course.
        ctrl_held = "control_l" in self.keys or "control_r" in self.keys
        alt_held = "alt_l" in self.keys or "alt_r" in self.keys

        lidar = "g" in self.keys

        jump_held = "k" in self.keys
        jump = jump_held and not ctrl_held and not alt_held
        jump_front = jump_held and ctrl_held
        jump_rear = jump_held and alt_held and not ctrl_held

        piston_held = "i" in self.keys
        roof_piston = piston_held and not ctrl_held and not alt_held
        roof_piston_front = piston_held and ctrl_held
        roof_piston_rear = piston_held and alt_held and not ctrl_held
        action = 0
        if gas or reverse:
            action |= 1
        if brake:
            action |= 2
        if reverse:
            action |= 4
        if clutch:
            action |= 8
        if tilt_left:
            action |= 16
        if tilt_right:
            action |= 32
        if shift_up:
            action |= 64
        if shift_down:
            action |= 128
        if toggle_drive:
            action |= 256
        if ignition:
            action |= 512
        if toggle_charge:
            action |= 1024
        if lidar:
            action |= 2048
        if heater:
            action |= 4096
        if jump:
            action |= 8192
        if jump_front:
            action |= 131072
        if jump_rear:
            action |= 262144
        if roof_piston:
            action |= 65536
        if roof_piston_front:
            action |= 524288
        if roof_piston_rear:
            action |= 1048576
        if climb:
            action |= 16384
        if propeller:
            action |= 32768
        if ballast_up:
            action |= 2097152
        if ballast_down:
            action |= 4194304
        return action

    def update(self) -> None:
        now = time.perf_counter()
        frame_dt = now - self.last_time
        self.last_time = now
        if frame_dt > 0.0:
            instant_fps = 1.0 / frame_dt
            self.fps_value = instant_fps if self.fps_value <= 0.0 else self.fps_value * 0.9 + instant_fps * 0.1

        obs, reward, terminated, truncated, _ = self.env.step(self.action())
        self.obs = obs
        self.last_reward = reward
        if terminated or truncated:
            reason = self.env.debug_info().get("termination_reason", "EPISODE ENDED")
            self.restart_notice = f"EVENT: {reason}; MANUAL PLAY CONTINUES (R RESETS SAME WORLD)"

        rgb = self.env.render()
        self.photo = self._tk.PhotoImage(data=_ppm_bytes(rgb), format="PPM")
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self.image_id, image=self.photo)

        debug = self.env.debug_info()
        self.update_hud(debug)
        self.update_control_hints(debug)
        self.status.configure(
            text=(
                f"DIST {debug.get('distance_m', 0.0):.1f}m  BEST {debug.get('best_distance_m', 0.0):.1f}m\n"
                f"gear={debug['gear']} x={debug['x']:.2f} vx={debug['vx']:.2f} "
                f"energy={debug['energy']:.3f} damage={debug['damage']:.3f} reward={self.last_reward:.3f}  "
                f"{self.restart_notice}"
            )
        )
        self.root.after(self.frame_ms, self.update)

    def update_hud(self, debug: dict) -> None:
        solar_mode = debug["solar_panel_requested"] or debug["solar_panel_deployment"] > 0.001
        if debug["engine_overheated"]:
            engine_text = "ENGINE OVERHEAT - COOLING"
            engine_color = "#ff5555"
        elif debug.get("heater_active", False):
            engine_text = "ENGINE PREHEAT - H"
            engine_color = "#ffb84d"
        elif debug["engine_cold_locked"]:
            engine_text = "ENGINE TOO COLD"
            engine_color = "#65bfff"
        elif not debug["engine_running"]:
            engine_text = "ENGINE OFF - PRESS E"
            engine_color = "#ffb84d"
        else:
            engine_text = f"RPM {debug['engine_rpm']:4.0f}"
            engine_color = "#f4f4f4"

        if debug["engine_overheated"]:
            temperature_color = "#ff5555"
        elif debug["engine_temperature"] >= debug["overheat_restart_temperature"]:
            temperature_color = "#ffb84d"
        elif debug["engine_cold_locked"]:
            temperature_color = "#65bfff"
        else:
            temperature_color = "#f4f4f4"

        if debug["top_gear"]:
            upshift_text = "X UP: MAX"
            upshift_color = "#888888"
            after_shift_text = "NEXT GEAR RPM: --"
        elif debug["shift_cooldown"] > 0.0:
            queued = " / QUEUED" if debug["shift_buffered"] else ""
            upshift_text = f"X UP: SYNC {debug['shift_cooldown']:.1f}s{queued}"
            upshift_color = "#ffb84d"
            after_shift_text = f"NEXT GEAR RPM: {debug['next_gear_rpm']:.0f}"
        elif not debug["upshift_speed_ok"]:
            queued = " / QUEUED" if debug["shift_buffered"] else ""
            upshift_text = (
                f"X UP: TOO SLOW  MIN {debug['minimum_shift_up_rpm']:.0f}{queued}"
            )
            upshift_color = "#ff5555"
            after_shift_text = f"NEXT GEAR RPM: {debug['next_gear_rpm']:.0f}"
        elif debug["clutch_down"]:
            upshift_text = "X UP: READY"
            upshift_color = "#52e06f"
            after_shift_text = f"NEXT GEAR RPM: {debug['next_gear_rpm']:.0f}"
        elif debug["upshift_recommended"]:
            suffix = " / QUEUED" if debug["shift_buffered"] else " + CLUTCH"
            upshift_text = f"X UP: SHIFT NOW{suffix}"
            upshift_color = "#52e06f"
            after_shift_text = f"NEXT GEAR RPM: {debug['next_gear_rpm']:.0f}"
        else:
            suffix = " / QUEUED" if debug["shift_buffered"] else ""
            upshift_text = f"X UP: WAIT {debug['shift_up_rpm']:.0f} RPM{suffix}"
            upshift_color = "#ffb84d"
            after_shift_text = f"NEXT GEAR RPM: {debug['next_gear_rpm']:.0f}"

        regen_text = f"MOTION RECHARGE +{debug.get('passive_charge_rate', 0.0):.2f}/s"
        regen_color = "#52e06f" if debug.get("passive_charge_rate", 0.0) > 0.0 else "#888888"
        panel_deploy = debug.get("solar_panel_deployment", 0.0)
        if not debug.get("solar_panel_requested", False):
            panel_text = "SOLAR PANEL STOWED - F"
        elif panel_deploy < 0.99:
            panel_text = f"SOLAR PANEL DEPLOYING {panel_deploy * 100:3.0f}%"
        elif debug.get("charging_active", False):
            panel_text = f"SOLAR CHARGING +{debug.get('solar_charge_rate', 0.0):.2f}/s"
        else:
            panel_text = "SOLAR PANEL READY - STOP TO CHARGE"
        battery_percent = debug['energy'] / max(1.0, debug['energy_capacity']) * 100
        net_rate = debug.get('energy_gain_rate', 0.0) - debug.get('energy_cost_rate', 0.0)
        battery_text = f"BATTERY {battery_percent:3.0f}%  NET {net_rate:+.2f}/s"

        piston_text = (
            "CONTACT" if debug.get("roof_piston_contact")
            else f"{debug.get('roof_piston_extension', 0.0) * 100:.0f}%"
        )
        jump_edge_text = {1: "FRONT", 2: "REAR", 3: "BOTH"}.get(
            debug.get("suspension_jump_mask", 3), "BOTH"
        )
        piston_edge_text = {1: "FRONT", 2: "REAR", 3: "BOTH"}.get(
            debug.get("roof_piston_mask", 3), "BOTH"
        )

        lines = [
            (f"FPS {self.fps_value:5.1f}", "#f4f4f4"),
            (f"SEED {debug.get('seed', 0)}  TIME {debug.get('trial_time_left', 0.0):5.1f}s", "#f4f4f4"),
            (f"SPD {debug['speed_kmh']:5.1f} km/h", "#f4f4f4"),
            (engine_text, engine_color),
            (f"TEMP {debug['engine_temperature']:5.1f} C  "
             f"ENV {debug['ambient_temperature']:5.1f} C  "
             f"PWR {debug['cold_power_factor'] * 100:3.0f}%", temperature_color),
            (f"GRAVITY {debug.get('gravity', -3.71):5.2f} m/s²  "
             f"x{debug.get('gravity_multiplier', 1.0):.2f}", "#d5c6ff"),
            (battery_text, "#52e06f" if net_rate > 0.01 else "#ffb84d" if net_rate < -0.01 else "#f4f4f4"),
            (f"LAYERS {debug.get('active_layers', 0)}  W {debug.get('layer_weight', 0.0):.2f}  "
             f"TRAC {debug.get('latent_traction', 1.0):.2f}  VISC {debug.get('latent_viscosity', 0.0):.2f}", "#d5c6ff"),
            (f"MOIST {debug.get('latent_moisture', 0.0):.2f}  "
             f"RESERVE {debug.get('latent_charge_reserve', 0.0) * 100:.0f}%", "#d5c6ff"),
            (f"SPRING {jump_edge_text} {'PRELOAD' if debug.get('suspension_jump_phase') == 1 else 'LAUNCH' if debug.get('suspension_jump_phase') == 2 else 'READY'} "
             f"{debug.get('suspension_jump_charge', 0.0) * 100:.0f}%  PISTON {piston_edge_text} "
             f"{piston_text} "
             f"B CLIMB {'ON' if debug.get('climb_mode') else 'OFF'}  "
             f"P PROP {debug.get('propeller_deployment', 0.0) * 100:.0f}%", "#52e06f"),
            (f"{'TEST ENDGAME' if debug.get('endgame_test_world') else 'TRAIN CURRICULUM'}  "
             f"BRANCH {debug.get('route_branch', 'terrain').upper()}  "
             f"BALLAST {debug.get('ballast_air', 0.55) * 100:.0f}%  "
             f"DIFF {debug.get('course_difficulty', 0.0) * 100:.0f}%", "#6fd3ff"),
            (regen_text, regen_color),
            (panel_text, "#52e06f" if debug.get("charging_active", False) else "#9fd8ff"),
            ((f"LIDAR ACTIVE {debug.get('lidar_range', 0.0):.1f} m  "
              f"COST {debug.get('lidar_energy_cost', 0.0):.2f}")
             if debug.get("lidar_active", False) else
             (f"LIDAR COOLDOWN {debug.get('lidar_cooldown', 0.0):.1f}s"
              if debug.get("lidar_cooldown", 0.0) > 0.0 else "LIDAR READY - G TO SCAN"),
             "#52ff8a" if debug.get("lidar_active", False) else
             ("#ffb84d" if debug.get("lidar_cooldown", 0.0) > 0.0 else "#888888")),
            (f"DRIVE {debug['drive_layout']}", "#f4f4f4"),
            (f"LIGHT {debug.get('screen_brightness', 1.0) * 100:3.0f}%", "#f4f4f4"),
            (f"GEAR {debug['gear']} / {debug['gear_count']}  "
             f"ENERGY x{debug['gear_energy_multiplier']:.2f}", "#f4f4f4"),
            (("CLUTCH OPEN" if debug["clutch_engagement"] <= 0.01 else
              "CLUTCH ENGAGED" if debug["clutch_engagement"] >= 0.99 else
              (f"CLUTCH DISENGAGING {debug.get('clutch_time_remaining', 0.0):.2f}s"
               if debug["clutch_down"] else
               f"CLUTCH ENGAGING {debug.get('clutch_time_remaining', 0.0):.2f}s")),
             "#52e06f" if debug["clutch_down"] else "#f4f4f4"),
            (upshift_text, upshift_color),
            (after_shift_text, "#9fd8ff" if not debug["top_gear"] else "#888888"),
            ((f"Z DOWN: SYNC {debug['shift_cooldown']:.1f}s" if debug["shift_cooldown"] > 0.0 else
              ("Z DOWN: READY" if debug["can_shift_down"] else
               ("Z DOWN: RECOMMENDED" if debug["should_shift_down"] else "Z DOWN: --"))),
             "#52e06f" if debug["can_shift_down"] else
             ("#ffb84d" if debug["should_shift_down"] else "#888888")),
        ]
        for label, (text, color) in zip(self.hud_labels, lines):
            label.configure(text=text, fg=color)

    def update_control_hints(self, debug: dict) -> None:
        active = "#287a3d"
        ready = "#9a651f"
        idle = "#252525"
        self.control_labels["gas"].configure(bg=active if ({"d", "right"} & self.keys) else idle)
        self.control_labels["brake"].configure(
            bg=active if ({"s", "down", "space"} & self.keys) else idle
        )
        self.control_labels["clutch"].configure(bg=active if debug["clutch_down"] else idle)
        self.control_labels["ignition"].configure(
            bg=active if "e" in self.keys else
            (ready if not debug["engine_running"] and not debug["engine_cold_locked"] and
             not debug["engine_overheated"] else idle)
        )
        self.control_labels["heater"].configure(
            bg=active if debug.get("heater_active", False) else
            (ready if debug["engine_cold_locked"] else idle)
        )
        self.control_labels["lidar"].configure(
            bg=active if debug.get("lidar_active", False) else
            (ready if debug.get("lidar_cooldown", 0.0) <= 0.0 else idle)
        )
        self.control_labels["down"].configure(
            bg=active if debug["can_shift_down"] else
            (ready if debug["should_shift_down"] else idle)
        )
        self.control_labels["up"].configure(
            bg="#555555" if debug["top_gear"] else
            (active if debug["upshift_recommended"] or debug["can_shift_up"] else
             (ready if debug["upshift_speed_ok"] else idle))
        )
        self.control_labels["drive"].configure(
            text=f"V  {debug['drive_layout']}",
            bg=("#315d83" if debug["drive_layout"] == "AWD" else
                ("#6a4d86" if debug["drive_layout"] == "FWD" else idle)),
        )
        self.control_labels["jump"].configure(
            bg=active if debug.get("suspension_jump_phase") == 1 or
                         debug.get("jump_cooldown", 0.0) > 0.0 else idle
        )
        self.control_labels["piston"].configure(
            bg=active if debug.get("roof_piston_extension", 0.0) > 0.0 else idle
        )
        self.control_labels["climb"].configure(
            bg=active if debug.get("climb_mode", False) else idle
        )
        self.control_labels["propeller"].configure(
            bg=active if debug.get("propeller_mode", False) else idle
        )
        self.control_labels["ballast_up"].configure(
            bg=active if debug.get("ballast_blowing", False) else idle
        )
        self.control_labels["ballast_down"].configure(
            bg=active if debug.get("ballast_flooding", False) else idle
        )

    def run(self) -> None:
        self.update()
        self.root.mainloop()


def main() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_dir / "play.yaml"))
    parser.add_argument("--rig", default=str(config_dir / "rover_rig.yaml"))
    parser.add_argument(
        "--seed", type=int, default=None,
        help="fixed world seed; omit for a fresh random seed every launch (press T in-session to reroll)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true",
                        help="use the entire screen; press F11 to toggle during play")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--split", choices=("train", "test"), default="train",
        help="train uses the progressive course; test starts in the held-out endgame world",
    )
    parser.add_argument(
        "--request-rules", type=int, default=0, metavar="N",
        help="request N structured rule candidates through the configured OpenAI endpoint before opening HUD",
    )
    parser.add_argument(
        "--rules-output", default="artifacts/candidates.json",
        help="path for candidates requested by --request-rules",
    )
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"No --seed given; using random seed={args.seed} (pass --seed to pin a world)")
    if args.request_rules:
        from mars_rover_env.rule_request import request_rules

        candidates = request_rules(args.request_rules)
        output = Path(args.rules_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(candidates)} rule candidates to {output}")
    ManualPlayer(args).run()


def _print_biome_catalog() -> None:
    from _mars_rover_cpp import biome_catalog

    split_names = {0: "builtin", 1: "train", 2: "test"}
    print("\nCOMPILED BIOME BANK")
    print("-" * 177)
    print(f"{'ID':<22} {'SPLIT':<8} {'COLOR':<11} {'FRIC':>6} {'SINK':>6} {'VISC':>6} "
          f"{'WIND':>6} {'GRAV':>6} {'ENERGY':>7} {'TEMP':>7} {'THERM':>6} {'SOLAR':>6} "
          f"{'P.RATE':>7} {'STORM':>6} {'LIGHT':>6} {'L.COST':>7} {'L.RANGE':>8}")
    print("-" * 177)
    for biome in biome_catalog():
        p = biome["parameters"]



        v = biome.get("visuals", {
            "ground_rgb": (42, 35, 30),
            "particle_rate": 0.0,
            "ambient_particles": 0,
        })
        color = "#" + "".join(f"{channel:02X}" for channel in v["ground_rgb"])
        print(
            f"{biome['id']:<22} {split_names.get(biome['split'], '?'):<8} {color:<11} "
            f"{p['friction']:>6.2f} {p['sink']:>6.3f} {p['viscosity']:>6.2f} "
            f"{p['wind']:>6.2f} {p['gravity']:>6.2f} {p['energy']:>7.2f} "
            f"{p['temperature']:>7.1f} {p['thermal']:>6.2f} {p['solar']:>6.2f} "
            f"{v['particle_rate']:>7.1f} {v['ambient_particles']:>6} "
            f"{v.get('screen_brightness', 1.0):>6.2f} "
            f"{p.get('lidar_energy', 1.0):>7.2f} {p.get('lidar_range', 1.0):>8.2f}"
        )
    print("-" * 177)
    print("Parameters are deterministic reference samples; generated code is frozen in this binary.\n")


if __name__ == "__main__":
    main()
