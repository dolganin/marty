from __future__ import annotations

import argparse
import json
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
        self.fps_value = 0.0
        self.frame_ms = max(1, int(1000 / args.fps))
        self.restart_notice = ""
        self.last_reward = 0.0

        self.root = tk.Tk()
        self.root.title("Mars Rover Manual Control")
        self.canvas = tk.Canvas(self.root, width=args.width, height=args.height, highlightthickness=0)
        self.canvas.pack()
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        self.control_labels: dict[str, tk.Label] = {}
        for name, text in (
            ("gas", "D / →  GAS"),
            ("brake", "S / ↓  BRAKE"),
            ("clutch", "C / SHIFT  CLUTCH"),
            ("down", "Z  GEAR DOWN"),
            ("up", "X  GEAR UP"),
            ("ignition", "E  IGNITION"),
            ("heater", "H  ENGINE HEAT"),
            ("solar", "F  SOLAR CHARGE"),
            ("lidar", "G  LIDAR SCAN"),
            ("drive", "V  RWD / FWD / AWD"),
            ("jump", "K  SUSPENSION JUMP"),
            ("climb", "B  CLIMB MODE"),
            ("propeller", "P  PROPELLER"),
            ("restart", "R  RESET SAME WORLD"),
        ):
            label = tk.Label(controls, text=text, bg="#252525", fg="#eeeeee",
                             font=("Consolas", 10, "bold"), padx=7, pady=4)
            label.pack(side="left", padx=2, pady=3)
            self.control_labels[name] = label
        self.status = tk.Label(self.root, anchor="w", justify="left")
        self.status.pack(fill="x")
        self.photo = None
        self.image_id = None
        self.hud_bg_id = None
        self.hud_text_ids: list[int] = []

        self.root.bind_all("<KeyPress>", self.on_key_press)
        self.root.bind_all("<KeyRelease>", self.on_key_release)
        self.root.after_idle(self.root.focus_force)

    def on_key_press(self, event) -> None:
        key = event.keysym.lower()
        self.keys.add(key)
        if key == "r":
            self.restart_notice = "RESTART: MANUAL RESET; SAME TRIAL"
            self.obs, self.info = self.env.reset(seed=self.seed, options={"trial_start": False})
            self.last_reward = 0.0
        elif key in {"escape", "q"}:
            self.root.destroy()

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
        toggle_charge = "f" in self.keys
        lidar = "g" in self.keys
        heater = "h" in self.keys
        jump = "k" in self.keys
        climb = "b" in self.keys
        propeller = "p" in self.keys
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
        if climb:
            action |= 16384
        if propeller:
            action |= 32768
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
                f"gear={debug['gear']} mechanic={debug['mechanic']} x={debug['x']:.2f} vx={debug['vx']:.2f} "
                f"energy={debug['energy']:.3f} damage={debug['damage']:.3f} reward={self.last_reward:.3f}  "
                f"{self.restart_notice}"
            )
        )
        self.root.after(self.frame_ms, self.update)

    def update_hud(self, debug: dict) -> None:
        solar_mode = debug["solar_panel_requested"] or debug["solar_panel_deployment"] > 0.001
        if solar_mode:
            engine_text = "ENGINE OFF - SOLAR MODE"
            engine_color = "#78c8ff"
        elif debug["engine_overheated"]:
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

        panel = debug["solar_panel_deployment"] * 100.0
        if debug["charging_active"]:
            solar_text = f"SOLAR CHARGING +{debug['solar_charge_rate']:.2f}/s"
            solar_color = "#52e06f"
        elif debug["solar_panel_requested"] and panel <= 0.1:
            solar_text = "SOLAR PARKING..."
            solar_color = "#ffb84d"
        elif debug["solar_panel_requested"]:
            solar_text = f"SOLAR DEPLOYING {panel:3.0f}%"
            solar_color = "#78c8ff"
        elif panel > 0.1:
            solar_text = f"SOLAR STOWING {panel:3.0f}%"
            solar_color = "#78c8ff"
        else:
            solar_text = "SOLAR STOWED - F TO DEPLOY"
            solar_color = "#888888"

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
            (f"ENERGY {debug['energy']:6.2f} / {debug['energy_capacity']:.0f}  "
             f"HEATER {'ON' if debug.get('heater_active', False) else 'OFF'}", "#f4f4f4"),
            (f"LAYERS {' + '.join(debug.get('active_layer_names', [])) or 'NONE'}  W {debug.get('layer_weight', 0.0):.2f}  "
             f"TRAC {debug.get('latent_traction', 1.0):.2f}  VISC {debug.get('latent_viscosity', 0.0):.2f}", "#d5c6ff"),
            (f"MOIST {debug.get('latent_moisture', 0.0):.2f}  PRESS {debug.get('latent_tire_pressure', 1.0):.2f}  "
             f"RESERVE {debug.get('latent_charge_reserve', 0.0) * 100:.0f}%", "#d5c6ff"),
            (f"K JUMP {debug.get('jump_cooldown', 0.0):.1f}s  "
             f"B CLIMB {'ON' if debug.get('climb_mode') else 'OFF'}  "
             f"P PROP {'ON' if debug.get('propeller_mode') else 'OFF'}", "#52e06f"),
            (f"BRANCH {debug.get('route_branch', 'terrain').upper()}", "#6fd3ff"),
            (solar_text, solar_color),
            ((f"LIDAR ACTIVE {debug.get('lidar_range', 0.0):.1f} m  "
              f"COST {debug.get('lidar_energy_cost', 0.0):.2f}")
             if debug.get("lidar_active", False) else
             (f"LIDAR COOLDOWN {debug.get('lidar_cooldown', 0.0):.1f}s"
              if debug.get("lidar_cooldown", 0.0) > 0.0 else "LIDAR READY - G TO SCAN"),
             "#52ff8a" if debug.get("lidar_active", False) else
             ("#ffb84d" if debug.get("lidar_cooldown", 0.0) > 0.0 else "#888888")),
            (f"DRIVE {debug['drive_layout']}", "#f4f4f4"),
            ((f"WATER {debug['water_depth']:.2f} m  "
              f"{max(0.0, debug['zone_end_x'] - debug['x']):.1f} m LEFT")
             if debug["mechanic"] == "Liquid" else
             f"SURFACE {debug['mechanic']}  {max(0.0, debug['zone_end_x'] - debug['x']):.1f} m",
             "#6fd3ff" if debug["mechanic"] == "Liquid" else "#f4f4f4"),
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
        if self.hud_bg_id is None:
            self.hud_bg_id = self.canvas.create_rectangle(
                6, 6, 430, 16 + len(lines) * 18, fill="#111111", outline=""
            )
            self.hud_text_ids = [
                self.canvas.create_text(12, 12 + i * 18, anchor="nw", font=("Consolas", 12, "bold"))
                for i in range(len(lines))
            ]
        for item_id, (text, color) in zip(self.hud_text_ids, lines):
            self.canvas.itemconfigure(item_id, text=text, fill=color)
            self.canvas.tag_raise(item_id)
        self.canvas.tag_raise(self.hud_bg_id)
        for item_id in self.hud_text_ids:
            self.canvas.tag_raise(item_id)

    def update_control_hints(self, debug: dict) -> None:
        active = "#287a3d"
        ready = "#9a651f"
        idle = "#252525"
        solar_mode = (
            debug["solar_panel_requested"] or debug["solar_panel_deployment"] > 0.001
        )
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
        self.control_labels["solar"].configure(
            bg=active if debug["charging_active"] else
            (ready if solar_mode else idle)
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
            bg=active if debug.get("jump_cooldown", 0.0) > 0.0 else idle
        )
        self.control_labels["climb"].configure(
            bg=active if debug.get("climb_mode", False) else idle
        )
        self.control_labels["propeller"].configure(
            bg=active if debug.get("propeller_mode", False) else idle
        )

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
    parser.add_argument(
        "--request-rules", type=int, default=0, metavar="N",
        help="request N structured rule candidates through the configured OpenAI endpoint before opening HUD",
    )
    parser.add_argument(
        "--rules-output", default="artifacts/candidates.json",
        help="path for candidates requested by --request-rules",
    )
    args = parser.parse_args()
    if args.request_rules:
        from mars_rover_env.rule_request import request_rules

        candidates = request_rules(args.request_rules)
        output = Path(args.rules_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(candidates)} rule candidates to {output}")
    _print_biome_catalog()
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
