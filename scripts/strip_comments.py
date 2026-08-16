from __future__ import annotations

import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TREES = (ROOT / "environment", ROOT / "baselines", ROOT / "scripts")


def strip_python(text: str) -> str:
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            tokens.append(token)
    return tokenize.untokenize(tokens)


def strip_cpp(text: str) -> str:
    output = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == '"':
                state = "string"
                output.append(char)
            elif char == "'":
                state = "char"
                output.append(char)
            elif char == "/" and following == "/":
                state = "line"
                index += 1
            elif char == "/" and following == "*":
                state = "block"
                index += 1
            else:
                output.append(char)
        elif state == "string":
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 1
            elif char == '"':
                state = "code"
        elif state == "char":
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 1
            elif char == "'":
                state = "code"
        elif state == "line":
            if char == "\n":
                output.append(char)
                state = "code"
        elif state == "block":
            if char == "*" and following == "/":
                state = "code"
                index += 1
            elif char == "\n":
                output.append(char)
        index += 1
    return "".join(output)


def strip_hash(text: str) -> str:
    lines = []
    for line_number, line in enumerate(text.splitlines(keepends=True)):
        if line_number == 0 and line.startswith("#!"):
            lines.append(line)
            continue
        single = False
        double = False
        escaped = False
        cut = len(line)
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\" and double:
                escaped = True
            elif char == "'" and not double:
                single = not single
            elif char == '"' and not single:
                double = not double
            elif char == "#" and not single and not double:
                cut = index
                break
        prefix = line[:cut].rstrip()
        if line.endswith("\n"):
            prefix += "\n"
        lines.append(prefix)
    return "".join(lines)


def main() -> None:
    python_suffixes = {".py"}
    cpp_suffixes = {".cpp", ".hpp", ".cc", ".h"}
    hash_suffixes = {".yaml", ".yml", ".toml", ".sh", ".ps1"}
    for tree in TREES:
        for path in tree.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix in python_suffixes:
                result = strip_python(text)
            elif path.suffix in cpp_suffixes:
                result = strip_cpp(text)
            elif path.suffix in hash_suffixes or path.name in {"CMakeLists.txt", "Makefile"}:
                result = strip_hash(text)
            else:
                continue
            path.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
