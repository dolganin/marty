from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE_HEADER = ROOT / "cpp" / "include" / "mars" / "biome_bank.hpp"
SOURCE_MANIFEST = ROOT / "python" / "mars_rover_env" / "configs" / "biome_bank.json"
HEADER = SOURCE_HEADER
MANIFEST = SOURCE_MANIFEST
CHECKPOINT = ROOT / "python" / "mars_rover_env" / "configs" / "biome_generation_checkpoint.json"
START = "// <MARS_GENERATED_BIOMES>"
END = "// </MARS_GENERATED_BIOMES>"
VERSION_RE = re.compile(
    r'inline constexpr std::string_view kBiomeBankVersion = "[^"]*";'
)

SKILL_STRATA = (
    "traction_loss",
    "lateral_force",
    "inertia_hysteresis",
    "gravity_change",
    "energy_mode",
    "dynamic_obstacle",
)
DEFAULT_STRATUM_QUOTA = 2


def _configure_bank_dir(bank_dir: Path | None) -> None:
    """Route mutable generation state to one run artifact, never to source."""
    global HEADER, MANIFEST, CHECKPOINT
    if bank_dir is None:
        return
    bank_dir = bank_dir.resolve()
    header = bank_dir / "include" / "mars" / "biome_bank.hpp"
    manifest = bank_dir / "biome_bank.json"
    header.parent.mkdir(parents=True, exist_ok=True)
    if not header.exists():
        shutil.copy2(SOURCE_HEADER, header)
    if not manifest.exists():
        shutil.copy2(SOURCE_MANIFEST, manifest)
    HEADER = header
    MANIFEST = manifest
    CHECKPOINT = bank_dir / "biome_generation_checkpoint.json"


class CandidateRejected(RuntimeError):
    """The model answered, but the candidate cannot enter the bank."""


class FatalApiError(RuntimeError):
    """Configuration/authentication error that retries cannot repair."""


class TransientApiError(RuntimeError):
    """Temporary service/network error that should be retried."""


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = data or {}
    local_path = path.with_name(path.stem + ".local" + path.suffix)
    if local_path.is_file():
        local = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        config = _merge(config, local)
    return config


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _endpoint(address: str, api_mode: str = "responses") -> str:
    address = address.rstrip("/")
    suffix = "/responses" if api_mode == "responses" else "/chat/completions"
    return address if address.endswith(suffix) else address + suffix


def _output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    choices = response.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
    if not chunks:
        raise CandidateRejected("API response contains no model output text")
    return "".join(chunks)


def _request_candidate(config: dict[str, Any], prompt: str) -> dict[str, str]:
    api = config["openai"]
    token = os.environ.get("OPENAI_API_KEY") or str(api.get("token", ""))
    if not token:
        raise FatalApiError("Set OPENAI_API_KEY or openai.token in the generator config")
    schema = {
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$"},
            "skill_stratum": {"type": "string", "enum": list(SKILL_STRATA)},
            "source": {"type": "string"},
        },
        "required": ["class_name", "skill_stratum", "source"],
        "additionalProperties": False,
    }
    api_mode = str(api.get("api_mode", "responses"))
    if api_mode == "responses":
        payload = {
            "model": api["model"],
            "instructions": config["generation"]["system_prompt"],
            "input": prompt,
            "max_output_tokens": int(api.get("max_output_tokens", 5000)),
            "temperature": float(api.get("temperature", 0.9)),
            "text": {"format": {"type": "json_schema", "name": "mars_biome", "strict": True, "schema": schema}},
        }
    elif api_mode == "chat_completions":
        payload = {
            "model": api["model"],
            "messages": [
                {"role": "system", "content": config["generation"]["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(api.get("max_output_tokens", 5000)),
            "temperature": float(api.get("temperature", 0.9)),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "mars_biome", "strict": True, "schema": schema},
            },
        }
    else:
        raise FatalApiError(f"Unsupported openai.api_mode: {api_mode!r}")
    body = json.dumps(payload).encode("utf-8")
    endpoint = _endpoint(str(api["address"]), api_mode)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        for _ in range(6):
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with opener.open(request, timeout=float(api.get("timeout_seconds", 180))) as reply:
                    response = json.loads(reply.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                location = exc.headers.get("Location")
                if exc.code not in {301, 302, 307, 308} or not location:
                    raise
                redirected = urllib.parse.urljoin(endpoint, location)
                old_url, new_url = urllib.parse.urlparse(endpoint), urllib.parse.urlparse(redirected)
                if old_url.hostname != new_url.hostname:
                    raise FatalApiError(
                        f"Refusing to forward the API token to redirect host {new_url.hostname!r}"
                    ) from exc
                if old_url.scheme == "https" and new_url.scheme != "https":
                    raise FatalApiError("Refusing an HTTPS to HTTP API redirect") from exc
                endpoint = redirected
        else:
            raise FatalApiError("OpenAI-compatible API returned too many redirects")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        message = f"OpenAI-compatible API returned HTTP {exc.code}: {detail}"
        if exc.code == 429 or exc.code >= 500:
            raise TransientApiError(message) from exc
        raise FatalApiError(message) from exc
    except urllib.error.URLError as exc:
        raise TransientApiError(f"OpenAI-compatible API network error: {exc}") from exc
    try:
        result = json.loads(_strip_json_fence(_output_text(response)))
    except json.JSONDecodeError as exc:
        raise CandidateRejected(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise CandidateRejected("Model output must be a JSON object")
    if "source" not in result:
        keys = ", ".join(sorted(str(key) for key in result))
        raise CandidateRejected(
            "Model output is missing the required source field "
            f"(returned keys: {keys or '<none>'})"
        )
    source = _strip_code_fence(str(result["source"]).strip())
    class_name = str(result.get("class_name", "")).strip()
    if not class_name:
        match = re.search(
            r"class\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:final\s*)?:\s*public\s+(?:mars::)?Biome\b",
            source,
        )
        if not match:
            raise CandidateRejected("Model omitted class_name and no Biome class could be inferred from source")
        class_name = match.group(1)
    return {
        "class_name": class_name,
        "skill_stratum": str(result.get("skill_stratum", "")).strip(),
        "source": source,
    }


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            return text[first_newline + 1 : -3].strip()
    return text


def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            return text[first_newline + 1 : -3].strip()
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):                                
        return None


def _compiler(config: dict[str, Any]) -> str:
    requested = os.environ.get("CXX") or str(config["generation"].get("compiler", "g++"))
    found = shutil.which(requested)
    if not found:
        raise RuntimeError(f"C++ compiler not found: {requested}")
    return found


def _syntax_check(candidate: dict[str, str], compiler: str, split: str) -> None:
    name, source = candidate["class_name"], candidate["source"]
    forbidden = {
        r"\breinterpret_cast\b": "pointer/address-derived randomness",
        r"\bconst_cast\b": "mutation through const-cast",
        r"\b(?:rand|srand)\s*\(": "process-global pseudo-randomness",
        r"\b(?:new|delete)\b": "dynamic allocation",
                                                                                    
                                                                                       
        r"\bstatic\s+(?!constexpr\b)": "static storage",
        r"\b(?:mutable|thread_local|volatile)\b": "mutable hidden state",
        r"\b(?:uintptr_t|addressof|random_device)\b": "address or process-derived randomness",
        r"\b(?:chrono|time|clock|gettimeofday)\b": "wall-clock input",
        r"#\s*(?:include|define)\b": "preprocessor directives",
        r"\b(?:std::)?(?:cout|cerr|clog|cin)\b": "stream I/O",
        r"\b(?:fopen|freopen|system|fork)\s*\(": "external process or file I/O",
    }
    for pattern, description in forbidden.items():
        if re.search(pattern, source):
            raise CandidateRejected(
                f"{name} uses forbidden {description}; biome mechanics must be deterministic "
                "pure functions of their seed and mechanic contexts"
            )
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise CandidateRejected(f"Invalid C++ class name: {name}")
    if not re.search(
        rf"class\s+{re.escape(name)}\s+(?:final\s*)?:\s*public\s+(?:mars::)?Biome\b",
        source,
    ):
        raise CandidateRejected(f"{name} must be declared as a public Biome subclass")
    split_enum = "Train" if split == "train" else "Test"
    if f"BiomeSplit::{split_enum}" not in source:
        raise CandidateRejected(f"{name} must return BiomeSplit::{split_enum} from split()")
    stratum = _candidate_stratum(candidate)
    if stratum not in SKILL_STRATA:
        raise CandidateRejected(f"Unknown skill_stratum: {stratum!r}")
    unit = (
        '#include "mars/biome_bank.hpp"\n'
        "namespace mars::generated_biomes::candidate_check {\n"
        f"{source}\n"
        f"static const {name} instance{{}};\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory(prefix="mars-biome-") as temp:
        path = Path(temp) / "candidate.cpp"
        path.write_text(unit, encoding="utf-8")
        command = [compiler, "-std=c++20", "-fsyntax-only", "-I", str(ROOT / "cpp" / "include"), str(path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise CandidateRejected(f"C++ interpretation failed for {name}:\n{result.stderr}")


def _behavioral_fingerprint(candidate: dict[str, Any], compiler: str) -> list[float]:
    """Compile a candidate and measure deterministic state deltas on a fixed action trace."""
    return _fingerprint_trace(
        candidate["class_name"],
        declarations=(
            "namespace mars::generated_biomes::fingerprint_check {\n"
            + candidate["source"]
            + "\n}"
        ),
        construction="generated_biomes::fingerprint_check::" + candidate["class_name"],
        compiler=compiler,
    )


def fingerprint_of_compiled_biome(qualified_class: str, compiler: str = "g++") -> list[float]:
    """Fingerprint a biome that is already part of the compiled bank.

    Hand-written biomes have no `source` field in the manifest - they live in the header
    rather than in the generated block - so the candidate path cannot reach them, and the
    audit rejects a bank whose rows carry no fingerprint. This runs the SAME trace against
    the compiled class, so hand-written and generated biomes stay directly comparable and
    the de-duplication distance keeps meaning what it meant.
    """
    return _fingerprint_trace(
        qualified_class.rsplit("::", 1)[-1],
        declarations="",
        construction=qualified_class,
        compiler=compiler,
    )


def _fingerprint_trace(
    name: str, *, declarations: str, construction: str, compiler: str
) -> list[float]:
    unit = f'''#include <cmath>
#include <iomanip>
#include <iostream>
#include "mars/biome_bank.hpp"
{declarations}
int main() {{
  using namespace mars;
  {construction} biome;
  std::cout << std::setprecision(9);
  for (uint64_t seed : {{11ULL, 29ULL, 47ULL}}) {{
    MechanicParams p = biome.sample_params(seed);
    std::cout << static_cast<int>(biome.visual_type()) << ' '
              << p.friction_mul << ' ' << p.sink_rate << ' ' << p.viscosity << ' '
              << p.wind_force << ' ' << p.gravity_mul << ' ' << p.energy_drain_mul << ' '
              << p.crust_deform << ' ' << p.ambient_temperature / 100.0f << ' '
              << p.thermal_transfer << ' ' << p.solar_charge_rate << ' '
              << p.lidar_energy_mul << ' ' << p.lidar_range_mul << ' '
              << p.terrain_amplitude_mul << ' ' << p.terrain_roughness_mul << ' '
              << p.terrain_crater_mul << ' ' << p.terrain_step_mul << ' '
              << p.ledge_gap_width << ' ' << p.ledge_spacing << ' '
              << p.ledge_ramp_length << ' ' << p.ledge_ramp_height << ' '
              << p.ledge_start_x << ' ' << biome.friction_scale(p) << ' ';
    for (float terrain_x : {{0.0f, 1.7f, 4.3f, 9.1f, 17.0f, 31.0f}}) {{
      std::cout << biome.terrain_height_delta(terrain_x, seed ^ 0x6a09e667ULL) << ' ';
    }}
    for (int hazard_step = 0; hazard_step <= 1260; hazard_step += 21) {{
      std::cout << biome.hazard_at(hazard_step) << ' ';
    }}
    for (int probe_step : {{0, 137, 389, 701, 1031}}) {{
      for (float probe_speed : {{0.15f, 0.85f, 2.20f, 4.00f}}) {{
        Vec2 probe_force{{0.0f, 0.0f}};
        float probe_torque = 0.0f, probe_cost = 0.0f;
        MechanicBodyContext probe;
        probe.body_force = &probe_force; probe.body_torque = &probe_torque;
        probe.energy_cost = &probe_cost; probe.velocity = Vec2{{probe_speed, 0.18f}};
        probe.angular_velocity = 0.24f; probe.mass = 12.0f;
        probe.gravity = -3.71f * p.gravity_mul; probe.dt = 1.0f / 60.0f;
        probe.step_index = probe_step;
        biome.apply_body_effects(p, probe);
        std::cout << probe_force.x << ' ' << probe_force.y << ' '
                  << probe_torque << ' ' << probe_cost << ' ';
      }}
    }}
    Vec2 velocity{{0.15f, 0.0f}};
    float angular_velocity = 0.0f, energy = 25.0f;
    for (int step = 0; step < 96; ++step) {{
      const float drive = step < 24 ? 70.0f : (step < 48 ? -35.0f : (step < 72 ? 0.0f : 95.0f));
      WheelContact contact;
      contact.active = true;
      contact.normal_force = 80.0f + static_cast<float>((step * 17) % 43);
      contact.penetration = 0.006f + static_cast<float>(step % 9) * 0.0015f;
      contact.normal = normalized(Vec2{{-0.08f * std::sin(step * 0.13f), 1.0f}});
      contact.tangent = normalized(Vec2{{contact.normal.y, -contact.normal.x}});
      Vec2 wheel_force{{0.0f, 0.0f}}, body_force{{0.0f, 0.0f}};
      float torque = 0.0f, cost = 0.0f;
      MechanicContext wheel;
      wheel.contact = &contact; wheel.wheel_force = &wheel_force; wheel.body_force = &body_force;
      wheel.energy_cost = &cost; wheel.dt = 1.0f / 60.0f; wheel.wheel_radius = 0.24f;
      wheel.base_friction = 1.2f; wheel.drive_force = drive; wheel.minimum_drive_limit = 0.0f;
      wheel.wheel_speed = velocity.x + std::sin(step * 0.19f); wheel.immersion = (step % 32) / 31.0f;
      const int biome_step = step * 13;
      wheel.step_index = biome_step;
      biome.apply(p, wheel);
      MechanicBodyContext body;
      body.body_force = &body_force; body.body_torque = &torque; body.energy_cost = &cost;
      body.velocity = velocity; body.angular_velocity = angular_velocity; body.mass = 12.0f;
      body.gravity = -3.71f * p.gravity_mul; body.dt = 1.0f / 60.0f; body.step_index = biome_step;
      biome.apply_body_effects(p, body);
      const Vec2 previous = velocity;
      velocity += (wheel_force + body_force) * (body.dt / 12.0f);
      angular_velocity += torque * body.dt / 4.0f;
      energy -= cost;
      if (step % 12 == 11) {{
        std::cout << velocity.x - previous.x << ' ' << velocity.y - previous.y << ' '
                  << angular_velocity << ' ' << energy << ' ' << contact.slip << ' ';
      }}
    }}
  }}
}}
'''
    with tempfile.TemporaryDirectory(prefix="mars-biome-fingerprint-") as temp:
        source_path = Path(temp) / "fingerprint.cpp"
        executable = Path(temp) / ("fingerprint.exe" if os.name == "nt" else "fingerprint")
        source_path.write_text(unit, encoding="utf-8")
        build = subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-I",
                str(ROOT / "cpp" / "include"),
                str(source_path),
                "-o",
                str(executable),
            ],
            capture_output=True,
            text=True,
        )
        if build.returncode:
            raise CandidateRejected(f"Behavior fingerprint build failed for {name}:\n{build.stderr}")
        run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
        if run.returncode:
            raise CandidateRejected(f"Behavior fingerprint failed for {name}:\n{run.stderr}")
    try:
        values = [float(item) for item in run.stdout.split()]
    except ValueError as exc:
        raise CandidateRejected(f"Non-numeric behavior fingerprint for {name}") from exc
    if not values or not all(math.isfinite(item) for item in values):
        raise CandidateRejected(f"Non-finite behavior fingerprint for {name}")
    return values


def _fingerprint_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    return math.sqrt(
        sum(((a - b) / (1.0 + abs(a) + abs(b))) ** 2 for a, b in zip(left, right))
        / len(left)
    )


def _render_block(candidates: list[dict[str, str]]) -> str:
    sources = "\n\n".join(item["source"] for item in candidates)
    instances = "\n".join(f"  static const {item['class_name']} biome_{i}; out.push_back(&biome_{i});" for i, item in enumerate(candidates))
    return f"\n{sources}\n\ninline void append(std::vector<const Biome*>& out) {{\n{instances}\n}}\n"


def _generated_sources_and_names(text: str) -> tuple[str, list[str]]:
    body = text.split(START, 1)[1].split(END, 1)[0]
    sources = body.split("inline void append", 1)[0].strip()
    names = re.findall(
        r"class\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:final\s*)?:\s*public\s+(?:mars::)?Biome\b",
        sources,
    )
    return sources, names


def _class_source(text: str, start: int) -> tuple[str, int]:
    """Return one generated class using brace matching rather than a fragile regex."""
    brace = text.find("{", start)
    if brace < 0:
        raise CandidateRejected("Generated class has no body")
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                if end < len(text) and text[end] == ";":
                    end += 1
                return text[start:end].strip(), end
    raise CandidateRejected("Generated class has unbalanced braces")


def _generated_candidates(text: str) -> list[dict[str, str]]:
    body = text.split(START, 1)[1].split(END, 1)[0].split("inline void append", 1)[0]
    pattern = re.compile(
        r"class\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:final\s*)?:\s*public\s+(?:mars::)?Biome\b"
    )
    candidates: list[dict[str, Any]] = []
    cursor = 0
    while match := pattern.search(body, cursor):
        source, cursor = _class_source(body, match.start())
        split_match = re.search(r"BiomeSplit::(Train|Test)", source)
        if not split_match:
            raise CandidateRejected(f"{match.group(1)} has no generated split")
        candidate = {"class_name": match.group(1), "source": source, "skill_stratum": ""}
        candidate["skill_stratum"] = _candidate_stratum(candidate)
        candidate["split"] = split_match.group(1).lower()
        candidates.append(candidate)
    return candidates


def _write_bank(
    candidates: list[dict[str, str]],
    replace: bool,
    split: str,
    dedup_epsilon: float | None = None,
) -> None:
    text = HEADER.read_text(encoding="utf-8")
    existing = _generated_candidates(text)
    if replace:
        existing = [item for item in existing if item["split"] != split]
    combined = existing + [{**item, "split": split} for item in candidates]
    _write_full_bank(combined, dedup_epsilon)


def _write_full_bank(
    candidates: list[dict[str, str]], dedup_epsilon: float | None = None
) -> None:
    text = HEADER.read_text(encoding="utf-8")
    instances = "\n".join(
        f"  static const {item['class_name']} biome_{i}; out.push_back(&biome_{i});"
        for i, item in enumerate(candidates)
    )
    new_sources = "\n\n".join(item["source"] for item in candidates)
    block = f"\n{new_sources}\n\ninline void append(std::vector<const Biome*>& out) {{\n{instances}\n}}\n"
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    HEADER.write_text(before + START + block + END + after, encoding="utf-8")
    _write_manifest(candidates, dedup_epsilon)


def _candidate_stratum(candidate: dict[str, str]) -> str:
    match = re.search(
        r'std::string_view\s+skill_stratum\s*\(\s*\)[^{]*\{.*?return\s+"([a-z_]+)";',
        candidate["source"],
        flags=re.DOTALL,
    )
    if not match:
        raise CandidateRejected(
            f"{candidate['class_name']} must return a literal from skill_stratum()"
        )
    source_stratum = match.group(1)
    declared = candidate.get("skill_stratum", "")
    if declared and declared != source_stratum:
        raise CandidateRejected(
            f"JSON skill_stratum={declared!r} disagrees with source value {source_stratum!r}"
        )
    return source_stratum


def _candidate_id_unchecked(source: str) -> str:
    match = re.search(
        r'std::string_view\s+id\s*\(\s*\)[^{]*\{.*?return\s+"([a-z0-9_]+)";',
        source,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def _write_manifest(
    candidates: list[dict[str, str]], dedup_epsilon: float | None = None
) -> None:
    previous: dict[str, Any] = {}
    old_manifest: dict[str, Any] = {}
    if MANIFEST.is_file():
        old_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        previous = {
            item.get("id", ""): item for item in old_manifest.get("biomes", [])
        }
    entries = []
    for item in candidates:
        biome_id = _candidate_id_unchecked(item["source"])
        source_hash = hashlib.sha256(item["source"].encode()).hexdigest()
        old_item = previous.get(biome_id, {})
        unchanged = old_item.get("source_sha256") == source_hash
        entries.append(
            {
                "id": biome_id,
                "class_name": item["class_name"],
                "split": item["split"],
                "skill_stratum": item["skill_stratum"],
                "source_sha256": source_hash,
                "status": old_item.get("status", "pending_difficulty_gate") if unchanged else "pending_difficulty_gate",
                "r_random": old_item.get("r_random") if unchanged else None,
                "r_solve": old_item.get("r_solve") if unchanged else None,
                "r_robust": old_item.get("r_robust") if unchanged else None,
                "behavior_fingerprint": item.get("behavior_fingerprint") or (
                    old_item.get("behavior_fingerprint") if unchanged else None
                ),
            }
        )
                                                                           
                                                                               
                                                                              
                                          
    handwritten_entries = [
        dict(item)
        for item in old_manifest.get("biomes", [])
        if item.get("origin") == "handwritten"
    ]
    anchors = ["normal", "sand", "ice", "mud", "wind", "low_gravity", "crust", "liquid"]
    anchor_source = HEADER.read_text(encoding="utf-8").split(START, 1)[0]
    anchor_source_sha256 = hashlib.sha256(anchor_source.encode()).hexdigest()
    version_entries = sorted([
        {key: item[key] for key in ("id", "split", "skill_stratum", "source_sha256")}
        for item in entries
    ], key=lambda item: (item["split"], item["id"]))
    canonical = json.dumps(
        {
            "anchors": anchors,
            "anchor_source_sha256": anchor_source_sha256,
            "quota_per_stratum": DEFAULT_STRATUM_QUOTA,
            "generated": version_entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    split_versions = {}
    for split in ("train", "test"):
        split_canonical = json.dumps(
            {
                "anchors": anchors,
                "anchor_source_sha256": anchor_source_sha256,
                "quota_per_stratum": DEFAULT_STRATUM_QUOTA,
                "generated": [item for item in version_entries if item["split"] == split],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        split_versions[f"{split}_version"] = (
            "sha256:" + hashlib.sha256(split_canonical.encode()).hexdigest()
        )
    payload = {
        "schema_version": 2,
        "bank_version": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        **split_versions,
        "reference_version": (
            old_manifest.get("reference_version")
            if old_manifest.get("train_version") == split_versions["train_version"]
            else None
        ),
        "anchors": anchors,
        "anchor_source_sha256": anchor_source_sha256,
        "quota_per_stratum": DEFAULT_STRATUM_QUOTA,
        "behavioral_dedup_epsilon": (
            float(dedup_epsilon)
            if dedup_epsilon is not None
            else float(old_manifest.get("behavioral_dedup_epsilon", 0.035))
        ),
        "difficulty_gates": {
            split: old_manifest.get("difficulty_gates", {}).get(split)
            for split in ("train", "test")
            if old_manifest.get(f"{split}_version") == split_versions[f"{split}_version"]
            and old_manifest.get("difficulty_gates", {}).get(split) is not None
        },
        "biomes": entries + handwritten_entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    declaration = (
        f'inline constexpr std::string_view kBiomeBankVersion = '
        f'"{payload["bank_version"]}";'
    )
    if VERSION_RE.search(header):
        header = VERSION_RE.sub(declaration, header)
    else:
        needle = END + "\n}"
        if needle not in header:
            raise RuntimeError("Cannot place compiled bank version in biome_bank.hpp")
        header = header.replace(needle, needle + "\n\n" + declaration, 1)
    HEADER.write_text(header, encoding="utf-8")


def _load_checkpoint(split: str, count: int, replace: bool) -> list[dict[str, Any]]:
    if not CHECKPOINT.is_file():
        return []
    payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    expected = {"split": split, "count": count, "replace": replace}
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise SystemExit(
            f"Generation checkpoint belongs to {actual}; finish it or remove {CHECKPOINT}"
        )
    candidates = list(payload.get("candidates", []))
    print(f"Resuming {split} generation with {len(candidates)}/{count} accepted candidates")
    return candidates


def _save_checkpoint(
    split: str, count: int, replace: bool, candidates: list[dict[str, Any]]
) -> None:
    payload = {
        "schema_version": 1,
        "split": split,
        "count": count,
        "replace": replace,
        "candidates": candidates,
    }
    temporary = CHECKPOINT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CHECKPOINT)


TRAIN_BRIEFS = {
    "traction_loss": "deformable dry terrain with a throttle/gear traction strategy",
    "lateral_force": "steady or slowly varying lateral force requiring balance and drive-layout control",
    "inertia_hysteresis": "momentum-dependent resistance with deterministic hysteresis or delayed consequences",
    "gravity_change": "a coherent nonstandard gravity regime coupled to suspension and gearing",
    "energy_mode": "thermal survival and energy routing under sustained engine load",
    "dynamic_obstacle": "either a deterministic time-varying hazard or a ballistic ledge where survival requires committing at speed",
}

                                                                                                 
TEST_BRIEFS = {
    "traction_loss": "traction inversion coupled to darkness or scan timing, without copying dry sink/deformation",
    "lateral_force": "intermittent cross-force coupled to thermal load, with calm windows for progress",
    "inertia_hysteresis": "reversible memory-like momentum effects coupled to buoyancy or drivetrain state",
    "gravity_change": "spatially or temporally modulated effective gravity coupled to energy scarcity",
    "energy_mode": "energy harvesting windows coupled to hazardous motion or visibility constraints",
    "dynamic_obstacle": "a held-out moving hazard or ballistic ledge whose required commitment speed must be inferred",
}


def _candidate_id(candidate: dict[str, str]) -> str:
    match = re.search(
        r'std::string_view\s+id\s*\(\s*\)[^{]*\{.*?return\s+"([a-z0-9_]+)";',
        candidate["source"],
        flags=re.DOTALL,
    )
    if not match:
        raise CandidateRejected(f"{candidate['class_name']} must return a literal snake_case id()")
    return match.group(1)


def _normalize_candidate(candidate: dict[str, str]) -> dict[str, str]:
    """Repair only harmless wrapper mistakes; never change biome behavior."""
    source = candidate["source"].strip()
    declaration = re.search(
        r"class\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:final\s*)?:\s*public\s+(?:mars::)?Biome\b",
        source,
    )
    if not declaration:
        raise CandidateRejected("Source does not contain a public Biome subclass")
    if source[: declaration.start()].strip():
        raise CandidateRejected("Source must contain only the Biome class declaration")
    class_source, end = _class_source(source, declaration.start())
    if source[end:].strip():
        raise CandidateRejected("Source must not declare globals or trailing helper code")
                                                                          
                                                   
    candidate = {
        "class_name": declaration.group(1),
        "skill_stratum": candidate.get("skill_stratum", ""),
        "source": class_source,
    }
                                                                            
                                                                
    if class_source.endswith("}"):
        candidate["source"] = class_source + ";"
    return candidate


def _prompt(
    header: str,
    split: str,
    accepted: int,
    total: int,
    request_number: int,
    existing_ids: set[str],
    feedback: str,
    target_stratum: str,
) -> str:
    briefs = TRAIN_BRIEFS if split == "train" else TEST_BRIEFS
    brief = briefs[target_stratum]
    correction = ""
    if feedback:
        correction = (
            "\n\nYOUR PREVIOUS CANDIDATE WAS REJECTED. Repair the concrete problem below; do not "
            "repeat the same mistake. You may redesign the biome if that produces cleaner code.\n"
            "<validator_feedback>\n" + feedback[-6000:] + "\n</validator_feedback>"
        )
    return (
        f"The bank currently has {accepted} accepted candidates and needs exactly {total}. "
        f"Generate the next candidate for the {split} bank (request #{request_number}). "
        f"Override split() to return BiomeSplit::{split.title()}.\n"
        f"The required skill_stratum for this candidate is {target_stratum!r}. Return it in the "
        "JSON field and override skill_stratum() with exactly the same literal.\n"
        "Already used biome ids (never reuse them): " + ", ".join(sorted(existing_ids)) + ".\n"
        f"Held-{ 'in' if split == 'train' else 'out' } creative direction: explore {brief}.\n"
        "Create a challenging but learnable location that requires a recognizably different policy, "
        "not a cosmetic reskin or a small coefficient change. Couple several mechanics coherently. "
        "It may be harsh and complicated, but it must leave the rover some adaptation strategy. "
        "Study every existing implementation below and avoid repeating its physical signature, visual "
        "identity, dominant hazard, or solution. Experiment aggressively within the contract. "
        "The bank must contain BOTH worlds where driving fast is dangerous and worlds where "
        "driving slowly or cautiously is fatal (commitment ledges). For a ledge, set the ledge_* "
        "MechanicParams so Env creates a physical gap and launch ramp; never simulate a gap with "
        "an arbitrary force. Scanner observations are raw physical echoes only: never encode or "
        "reveal the mechanic class, hazard phase, safe speed, or correct action. A surface-break "
        "precursor must remain ambiguous: it may require braking before a pit in one world and "
        "accelerating over a ramp in another."
        + correction + "\n\n"
        "Complete contract and existing implementation bank:\n```cpp\n" + header + "\n```"
    )


def _catalog_from_header() -> list[str]:
    text = HEADER.read_text(encoding="utf-8")
    return re.findall(
        r'std::string_view\s+id\s*\(\s*\)[^{]*\{.*?return\s+"([a-z0-9_]+)";',
        text,
        flags=re.DOTALL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and compile-check an immutable C++ biome bank")
    parser.add_argument("--config", type=Path, default=ROOT / "python" / "mars_rover_env" / "configs" / "biome_generator.yaml")
    parser.add_argument("--count", type=int)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument(
        "--target-strata",
        help=(
            "Comma-separated skill strata for the new slots, in generation order. "
            "Use this when filling an exact stratified quota."
        ),
    )
    parser.add_argument("--list", action="store_true", help="list biome ids already present in the C++ bank")
    parser.add_argument("--refresh-manifest", action="store_true", help="rebuild bank metadata without API calls")
    parser.add_argument("--prune-rejected", action="store_true", help="remove difficulty-rejected generated biomes")
    parser.add_argument("--dry-run", action="store_true", help="check config/compiler and print request summary without API calls")
    parser.add_argument("--replace", action="store_true", help="discard the previous generated bank instead of appending")
    parser.add_argument(
        "--bank-dir", type=Path,
        help="artifact directory for this run's mutable C++ bank and manifest",
    )
    args = parser.parse_args()
    _configure_bank_dir(args.bank_dir)
    if args.list:
        print("\n".join(_catalog_from_header()))
        return
    if args.refresh_manifest:
        config = _load(args.config)
        compiler = _compiler(config)
        candidates = _generated_candidates(HEADER.read_text(encoding="utf-8"))
        print("Refreshing deterministic fingerprints for every generated biome...")
        for item in candidates:
            _syntax_check(item, compiler, item["split"])
            item["behavior_fingerprint"] = _behavioral_fingerprint(item, compiler)
        _write_manifest(
            candidates,
            float(config["generation"].get("behavioral_dedup_epsilon", 0.035)),
        )
                                                                        
                                                                                
                                                                       
        from mars_rover_env.tools.evaluate_biomes import (
            _catalog_by_id,
            _handwritten_class_name,
            fingerprint_of_compiled_biome,
        )

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        catalog = _catalog_by_id()
        active_handwritten_ids = {
            biome_id
            for biome_id, biome in catalog.items()
            if int(biome["split"]) in (1, 2)
        }
        manifest["biomes"] = [
            row
            for row in manifest["biomes"]
            if row.get("origin") != "handwritten"
            or str(row.get("id")) in active_handwritten_ids
        ]
        by_id = {str(item["id"]): item for item in manifest["biomes"]}
        for biome_id, biome in catalog.items():
            if int(biome["split"]) not in (1, 2):
                continue
            row = by_id.get(biome_id)
            if row is None:
                row = {
                    "id": biome_id,
                    "split": "train" if int(biome["split"]) == 1 else "test",
                    "origin": "handwritten",
                    "display_name": biome.get("name", biome_id),
                    "skill_stratum": biome.get("skill_stratum", ""),
                }
                manifest["biomes"].append(row)
                by_id[biome_id] = row
            if row.get("origin") == "handwritten":
                row.update(
                    split="train" if int(biome["split"]) == 1 else "test",
                    display_name=biome.get("name", biome_id),
                    skill_stratum=biome.get("skill_stratum", ""),
                )
                row["behavior_fingerprint"] = fingerprint_of_compiled_biome(
                    "handcrafted_biomes::" + _handwritten_class_name(biome_id)
                )
            row.update(
                status="pending_difficulty_gate",
                r_random=None,
                r_solve=None,
                r_robust=None,
            )
            for key in (
                "random_score", "solve_score", "robust_score", "rejection_reasons",
                "strategy_disagreement", "strategy_profile", "strategy_regret",
                "strategy_spread", "strategy_winner",
            ):
                row.pop(key, None)
                                                                                       
                                                                                      
        manifest["difficulty_gates"] = {}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Refreshed {MANIFEST}")
        return
    if args.prune_rejected:
        if not MANIFEST.is_file():
            raise SystemExit(f"Missing manifest: {MANIFEST}")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rejected = {
            item["id"] for item in manifest.get("biomes", [])
            if str(item.get("status", "")).startswith("rejected_")
        }
        existing = _generated_candidates(HEADER.read_text(encoding="utf-8"))
        retained = [item for item in existing if _candidate_id(item) not in rejected]
        _write_full_bank(retained)
        print(f"Pruned {len(existing) - len(retained)} rejected biomes: {', '.join(sorted(rejected))}")
        return
    config = _load(args.config)
    count = args.count if args.count is not None else int(config["generation"].get("count", 12))
    split = args.split or str(config["generation"].get("split", "train"))
    forced_strata = tuple(
        item.strip() for item in (args.target_strata or "").split(",") if item.strip()
    )
    unknown_forced = sorted(set(forced_strata) - set(SKILL_STRATA))
    if unknown_forced:
        raise SystemExit("Unknown target strata: " + ", ".join(unknown_forced))
    compiler = _compiler(config)
    if args.dry_run:
        api_mode = str(config["openai"].get("api_mode", "responses"))
        print(f"endpoint={_endpoint(str(config['openai']['address']), api_mode)}")
        print(f"api_mode={api_mode}")
        print(f"model={config['openai']['model']} split={split} count={count} compiler={compiler}")
        return
    header = HEADER.read_text(encoding="utf-8")
    _, existing_names = _generated_sources_and_names(header)
    existing_candidates = _generated_candidates(header)
    retained_for_floor = [] if args.replace else [
        item for item in existing_candidates if item["split"] == split
    ]
    needed = count - len(retained_for_floor)
    if needed < 0:
        raise SystemExit(
            f"{split} bank already has {len(retained_for_floor)} items, above target count={count}; "
            "use --replace for a new bank"
        )
    present_strata = {item["skill_stratum"] for item in retained_for_floor}
    missing_strata = [item for item in SKILL_STRATA if item not in present_strata]
    if forced_strata and len(forced_strata) != needed:
        raise SystemExit(
            f"target-strata provides {len(forced_strata)} slots, but count={count} "
            f"requires exactly {needed} new candidates"
        )
    if not forced_strata and needed < len(missing_strata):
        raise SystemExit(
            f"Only {needed} open slots cannot satisfy the skill floor; need at least "
            f"{len(missing_strata)} "
            f"candidates for: {', '.join(missing_strata)}"
        )
    if needed == 0:
        print(f"{split} bank already has target count={count}; nothing to generate")
        return
    opposite_split = "train" if split == "test" else "test"
    print(f"Building fixed-trajectory fingerprints for the frozen {opposite_split} bank...")
    opposite_fingerprints: list[tuple[str, list[float]]] = []
    if MANIFEST.is_file():
        current_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        opposite_fingerprints.extend(
            (str(item["id"]), [float(value) for value in item["behavior_fingerprint"]])
            for item in current_manifest.get("biomes", [])
            if item.get("split") == opposite_split and item.get("behavior_fingerprint")
        )
    known_opposite_ids = {biome_id for biome_id, _ in opposite_fingerprints}
    for item in existing_candidates:
        if item["split"] != opposite_split or _candidate_id(item) in known_opposite_ids:
            continue
        opposite_fingerprints.append(
            (_candidate_id(item), _behavioral_fingerprint(item, compiler))
        )
    if split == "test" and not opposite_fingerprints:
        raise SystemExit("Cannot generate a test bank before the train bank is frozen")
    dedup_epsilon = float(config["generation"].get("behavioral_dedup_epsilon", 0.035))
    candidates: list[dict[str, Any]] = _load_checkpoint(split, count, args.replace)
    if len(candidates) > needed:
        raise SystemExit(
            "Generation checkpoint contains more candidates than the current target requires"
        )
    if candidates:
        print("Recomputing checkpoint fingerprints with the current validator...")
        for item in candidates:
            _syntax_check(item, compiler, split)
            item["behavior_fingerprint"] = _behavioral_fingerprint(item, compiler)
        _save_checkpoint(split, count, args.replace, candidates)
    peer_fingerprints: list[tuple[str, list[float]]] = []
    for item in retained_for_floor:
        peer_fingerprints.append(
            (_candidate_id(item), _behavioral_fingerprint(item, compiler))
        )
    known_peer_ids = {biome_id for biome_id, _ in peer_fingerprints}
    if MANIFEST.is_file():
        current_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        peer_fingerprints.extend(
            (str(item["id"]), [float(value) for value in item["behavior_fingerprint"]])
            for item in current_manifest.get("biomes", [])
            if item.get("split") == split
            and str(item.get("id")) not in known_peer_ids
            and item.get("behavior_fingerprint")
        )
                                                                               
                                                                    
    names: set[str] = set(existing_names)
    ids: set[str] = set(_catalog_from_header())
    names.update(item["class_name"] for item in candidates)
    ids.update(_candidate_id(item) for item in candidates)
    request_number = 0
    feedback = ""
    transient_failures = 0
    target_stratum: str | None = None
    free_stratum_rng = random.SystemRandom()
    while len(candidates) < needed:
        if target_stratum is None:
            if forced_strata:
                target_stratum = forced_strata[len(candidates)]
            else:
                generated_strata = {item["skill_stratum"] for item in candidates}
                still_missing = [
                    item for item in missing_strata if item not in generated_strata
                ]
                target_stratum = (
                    still_missing[0]
                    if still_missing
                    else free_stratum_rng.choice(SKILL_STRATA)
                )
        request_number += 1
        accepted_context = "\n\n".join(item["source"] for item in candidates)
        context = header + ("\n\nAlready accepted in this run:\n" + accepted_context if accepted_context else "")
        candidate: dict[str, str] | None = None
        try:
            candidate = _normalize_candidate(_request_candidate(
                config,
                _prompt(
                    context,
                    split,
                    len(retained_for_floor) + len(candidates),
                    count,
                    request_number,
                    ids,
                    feedback,
                    target_stratum,
                ),
            ))
            if candidate["class_name"] in names:
                raise CandidateRejected(f"Duplicate generated class: {candidate['class_name']}")
            candidate_id = _candidate_id(candidate)
            if candidate_id in ids:
                raise CandidateRejected(f"Duplicate biome id: {candidate_id}")
            if _candidate_stratum(candidate) != target_stratum:
                raise CandidateRejected(
                    f"Expected skill_stratum={target_stratum!r}, got "
                    f"{_candidate_stratum(candidate)!r}"
                )
            _syntax_check(candidate, compiler, split)
            candidate["behavior_fingerprint"] = _behavioral_fingerprint(candidate, compiler)
            if opposite_fingerprints:
                nearest_id, nearest_distance = min(
                    (
                        (biome_id, _fingerprint_distance(candidate["behavior_fingerprint"], fingerprint))
                        for biome_id, fingerprint in opposite_fingerprints
                    ),
                    key=lambda item: item[1],
                )
                if nearest_distance < dedup_epsilon:
                    raise CandidateRejected(
                        f"Behavior duplicates {opposite_split} biome {nearest_id!r}: "
                        f"distance={nearest_distance:.6f} < epsilon={dedup_epsilon:.6f}"
                    )
            peers = peer_fingerprints + [
                (_candidate_id(item), item["behavior_fingerprint"])
                for item in candidates
            ]
            if peers:
                nearest_peer = min(
                    (
                        (biome_id, _fingerprint_distance(
                            candidate["behavior_fingerprint"], fingerprint
                        ))
                        for biome_id, fingerprint in peers
                    ),
                    key=lambda item: item[1],
                )
                if nearest_peer[1] < dedup_epsilon:
                    raise CandidateRejected(
                        f"Behavior duplicates accepted {split} biome {nearest_peer[0]!r}: "
                        f"distance={nearest_peer[1]:.6f} < epsilon={dedup_epsilon:.6f}"
                    )
        except CandidateRejected as exc:
            rejection = str(exc)
            feedback = rejection
            if candidate is not None and candidate.get("source"):
                feedback += "\n\nRejected source:\n" + candidate["source"]
            print(
                f"[{len(retained_for_floor) + len(candidates)}/{count}] "
                f"request {request_number} rejected; "
                f"feeding error back to model: {rejection}"
            )
            continue
        except TransientApiError as exc:
            transient_failures += 1
            delay = min(30.0, 2.0 ** min(transient_failures, 5))
            print(f"Temporary API failure: {exc}\nRetrying in {delay:.0f}s...")
            time.sleep(delay)
            continue
        candidates.append(candidate)
        names.add(candidate["class_name"])
        ids.add(candidate_id)
        feedback = ""
        target_stratum = None
        transient_failures = 0
        _save_checkpoint(split, count, args.replace, candidates)
        print(
            f"[{len(retained_for_floor) + len(candidates)}/{count}] "
            f"accepted {candidate['class_name']} "
            f"(id={candidate_id}, request {request_number}, syntax only)"
        )
    _write_bank(candidates, args.replace, split, dedup_epsilon)
    CHECKPOINT.unlink(missing_ok=True)
    print(
        f"Wrote {len(candidates)} new {split} implementations; "
        f"split total={len(retained_for_floor) + len(candidates)} in {HEADER}"
    )
    print("Rebuild the native extension before starting training; the bank is then frozen in the binary.")


if __name__ == "__main__":
    main()
