from __future__ import annotations

import argparse
import json
import os
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
HEADER = ROOT / "cpp" / "include" / "mars" / "biome_bank.hpp"
START = "// <MARS_GENERATED_BIOMES>"
END = "// </MARS_GENERATED_BIOMES>"


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
            "source": {"type": "string"},
        },
        "required": ["class_name", "source"],
        "additionalProperties": False,
    }
    api_mode = str(api.get("api_mode", "responses"))
    if api_mode == "responses":
        payload = {
            "model": api["model"],
            "instructions": config["generation"]["system_prompt"],
            "input": prompt,
            "max_output_tokens": int(api.get("max_output_tokens", 5000)),
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
    return {"class_name": class_name, "source": source}


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
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _compiler(config: dict[str, Any]) -> str:
    requested = os.environ.get("CXX") or str(config["generation"].get("compiler", "g++"))
    found = shutil.which(requested)
    if not found:
        raise RuntimeError(f"C++ compiler not found: {requested}")
    return found


def _syntax_check(candidate: dict[str, str], compiler: str, split: str) -> None:
    name, source = candidate["class_name"], candidate["source"]
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


def _write_bank(candidates: list[dict[str, str]], replace: bool) -> None:
    text = HEADER.read_text(encoding="utf-8")
    old_sources, old_names = _generated_sources_and_names(text)
    combined: list[dict[str, str]] = [] if replace else [
        {"class_name": name, "source": ""} for name in old_names
    ]
    combined.extend(candidates)
    new_sources = "\n\n".join(item["source"] for item in candidates)
    if not replace and old_sources:
        new_sources = old_sources + ("\n\n" + new_sources if new_sources else "")
    instances = "\n".join(
        f"  static const {item['class_name']} biome_{i}; out.push_back(&biome_{i});"
        for i, item in enumerate(combined)
    )
    block = f"\n{new_sources}\n\ninline void append(std::vector<const Biome*>& out) {{\n{instances}\n}}\n"
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    HEADER.write_text(before + START + block + END + after, encoding="utf-8")


CREATIVE_BRIEFS = (
    "traction, deformable terrain and a non-obvious throttle/gear strategy",
    "gravity, wind or body torque that demands active balance and drive-layout adaptation",
    "thermal survival, engine load and energy routing under environmental pressure",
    "darkness, misleading visibility and deliberate lidar budgeting",
    "liquid-like resistance, buoyancy, sinking or momentum management",
    "time-varying forces or oscillatory hazards derived deterministically from step_index",
    "a difficult start/escape condition followed by a different high-speed risk",
    "an unusual coupling of particles/visibility with physical and energy consequences",
)


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
    # The class name in JSON is occasionally stale while the C++ source is
    # otherwise valid. The source is authoritative.
    candidate = {"class_name": declaration.group(1), "source": source}
    # The overwhelmingly common `expected initializer before instance` error
    # is simply a missing semicolon after the final class brace.
    if source.endswith("}"):
        candidate["source"] = source + ";"
    return candidate


def _prompt(
    header: str,
    split: str,
    accepted: int,
    total: int,
    request_number: int,
    existing_ids: set[str],
    feedback: str,
) -> str:
    brief = CREATIVE_BRIEFS[(request_number - 1) % len(CREATIVE_BRIEFS)]
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
        "Already used biome ids (never reuse them): " + ", ".join(sorted(existing_ids)) + ".\n"
        f"Creative direction for this request: explore {brief}. This is inspiration, not a cage.\n"
        "Create a challenging but learnable location that requires a recognizably different policy, "
        "not a cosmetic reskin or a small coefficient change. Couple several mechanics coherently. "
        "It may be harsh and complicated, but it must leave the rover some adaptation strategy. "
        "Study every existing implementation below and avoid repeating its physical signature, visual "
        "identity, dominant hazard, or solution. Experiment aggressively within the contract."
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
    parser.add_argument("--list", action="store_true", help="list biome ids already present in the C++ bank")
    parser.add_argument("--dry-run", action="store_true", help="check config/compiler and print request summary without API calls")
    parser.add_argument("--replace", action="store_true", help="discard the previous generated bank instead of appending")
    args = parser.parse_args()
    if args.list:
        print("\n".join(_catalog_from_header()))
        return
    config = _load(args.config)
    count = args.count if args.count is not None else int(config["generation"].get("count", 12))
    split = args.split or str(config["generation"].get("split", "train"))
    compiler = _compiler(config)
    if args.dry_run:
        api_mode = str(config["openai"].get("api_mode", "responses"))
        print(f"endpoint={_endpoint(str(config['openai']['address']), api_mode)}")
        print(f"api_mode={api_mode}")
        print(f"model={config['openai']['model']} split={split} count={count} compiler={compiler}")
        return
    header = HEADER.read_text(encoding="utf-8")
    _, existing_names = _generated_sources_and_names(header)
    candidates: list[dict[str, str]] = []
    # Even --replace sees the previous bank as a diversity blacklist: replacing
    # a bank should produce genuinely new tasks, not renamed copies.
    names: set[str] = set(existing_names)
    ids: set[str] = set(_catalog_from_header())
    request_number = 0
    feedback = ""
    transient_failures = 0
    while len(candidates) < count:
        request_number += 1
        accepted_context = "\n\n".join(item["source"] for item in candidates)
        context = header + ("\n\nAlready accepted in this run:\n" + accepted_context if accepted_context else "")
        candidate: dict[str, str] | None = None
        try:
            candidate = _normalize_candidate(_request_candidate(
                config,
                _prompt(context, split, len(candidates), count, request_number, ids, feedback),
            ))
            if candidate["class_name"] in names:
                raise CandidateRejected(f"Duplicate generated class: {candidate['class_name']}")
            candidate_id = _candidate_id(candidate)
            if candidate_id in ids:
                raise CandidateRejected(f"Duplicate biome id: {candidate_id}")
            _syntax_check(candidate, compiler, split)
        except CandidateRejected as exc:
            feedback = str(exc)
            if candidate is not None and candidate.get("source"):
                feedback += "\n\nRejected source:\n" + candidate["source"]
            print(
                f"[{len(candidates)}/{count}] request {request_number} rejected; "
                f"feeding error back to model:\n{feedback}"
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
        transient_failures = 0
        print(
            f"[{len(candidates)}/{count}] accepted {candidate['class_name']} "
            f"(id={candidate_id}, request {request_number}, syntax only)"
        )
    _write_bank(candidates, args.replace)
    print(f"Wrote {len(candidates)} {split} biome implementations to {HEADER}")
    print("Rebuild the native extension before starting training; the bank is then frozen in the binary.")


if __name__ == "__main__":
    main()
