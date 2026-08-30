"""Request bounded coupling-rule candidates using the existing OpenAI config."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from mars_rover_env.rules import validate_rule


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "biome_generator.yaml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_openai_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    local = path.with_name(path.stem + ".local" + path.suffix)
    if local.is_file():
        config = _merge(config, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return config


def _output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    choices = response.get("choices", [])
    if choices and isinstance(choices[0].get("message", {}).get("content"), str):
        return choices[0]["message"]["content"]
    raise RuntimeError("LLM response contains no text")


def request_rules(count: int, config_path: str | Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    if count < 1 or count > 16:
        raise ValueError("count must be from 1 to 16")
    config = load_openai_config(config_path)
    api = config.get("openai", {})
    token = os.environ.get("OPENAI_API_KEY") or str(api.get("token", ""))
    if not token:
        raise RuntimeError("Set OPENAI_API_KEY or openai.token in biome_generator.local.yaml")
    schema = {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array", "minItems": count, "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "inputs": {"type": "array", "minItems": 1, "maxItems": 2,
                                   "items": {"type": "string"}},
                        "target": {"type": "string"},
                        "coefficients": {"type": "array", "minItems": 1, "maxItems": 2,
                                         "items": {"type": "number"}},
                        "bias": {"type": "number"},
                    },
                    "required": ["inputs", "target", "coefficients", "bias"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rules"], "additionalProperties": False,
    }
    instruction = (
        "Propose distinct bounded coupling rules for a Mars rover. Return only the JSON schema. "
        "Inputs allowed: moisture, viscosity, heat, tire_pressure, slip, speed, slope, immersion, throttle. "
        "Targets allowed: traction, moisture, heat, charge_reserve, tire_pressure, viscosity, sink, suspension. "
        "Use one or two inputs, matching coefficient count, coefficients and bias within [-2.5, 2.5]. "
        "Rules must be deterministic affine formulas and mechanically plausible."
    )
    mode = str(api.get("api_mode", "responses"))
    address = str(api.get("address", "https://api.openai.com/v1")).rstrip("/")
    if mode == "responses":
        endpoint = address if address.endswith("/responses") else address + "/responses"
        payload = {"model": api["model"], "instructions": instruction,
                   "input": f"Generate exactly {count} candidates.",
                   "max_output_tokens": int(api.get("max_output_tokens", 5000)),
                   "temperature": float(api.get("temperature", 0.7)),
                   "text": {"format": {"type": "json_schema", "name": "coupling_rules",
                                       "strict": True, "schema": schema}}}
    elif mode == "chat_completions":
        endpoint = address if address.endswith("/chat/completions") else address + "/chat/completions"
        payload = {"model": api["model"], "messages": [{"role": "system", "content": instruction},
                   {"role": "user", "content": f"Generate exactly {count} candidates."}],
                   "max_tokens": int(api.get("max_output_tokens", 5000)),
                   "temperature": float(api.get("temperature", 0.7)),
                   "response_format": {"type": "json_schema", "json_schema":
                       {"name": "coupling_rules", "strict": True, "schema": schema}}}
    else:
        raise RuntimeError(f"Unsupported openai.api_mode: {mode!r}")
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=float(api.get("timeout_seconds", 180))) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI request failed: HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    rules = json.loads(_output_text(result)).get("rules", [])
    if len(rules) != count:
        raise RuntimeError("LLM did not return the requested number of rules")
    return [validate_rule(rule).to_dict() for rule in rules]
