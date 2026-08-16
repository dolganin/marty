from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TREES = (ROOT / "environment", ROOT / "scripts")


def strip_docstrings(text: str) -> str:
    tree = ast.parse(text)
    ranges = []
    nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, nodes) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            replacement = None
            if not isinstance(node, ast.Module) and len(node.body) == 1:
                source = text.splitlines()[first.lineno - 1]
                replacement = source[: len(source) - len(source.lstrip())] + "pass\n"
            ranges.append((first.lineno, first.end_lineno or first.lineno, replacement))
    lines = text.splitlines(keepends=True)
    for start, end, replacement in ranges:
        for index in range(start - 1, end):
            lines[index] = "\n" if lines[index].endswith("\n") else ""
        if replacement is not None:
            lines[start - 1] = replacement
    return "".join(lines)


def strip_python(text: str) -> str:
    text = strip_docstrings(text)
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
            if path.suffix in python_suffixes:
                transform = strip_python
            elif path.suffix in cpp_suffixes:
                transform = strip_cpp
            elif path.suffix in hash_suffixes or path.name in {"CMakeLists.txt", "Makefile"}:
                transform = strip_hash
            else:
                continue
            text = path.read_text(encoding="utf-8-sig")
            result = transform(text)
            path.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
