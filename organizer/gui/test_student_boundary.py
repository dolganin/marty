from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOTS = (ROOT / "student",)
TERMS_PATH = Path(__file__).with_name("private_terms.txt")
TEXT_SUFFIXES = {".py", ".toml", ".md", ".txt", ".yaml", ".yml", ".json"}


def main() -> None:
    terms = tuple(line.strip() for line in TERMS_PATH.read_text().splitlines() if line.strip())
    leaks = []
    for public_root in PUBLIC_ROOTS:
        for path in public_root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if any(part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info") for part in path.parts):
                continue
            text = path.read_text(errors="replace")
            for term in terms:
                if term.casefold() in text.casefold():
                    leaks.append(f"{path.relative_to(ROOT)}: {term}")
    profile = (ROOT / "student/gui/python/mars_rover_gui/student_profile.py").read_text()
    if "VISIBLE_SPLITS = frozenset({0, 1})" not in profile:
        leaks.append("student profile does not restrict the visible catalog")
    if "biome_split=1" not in profile:
        leaks.append("student profile does not force the training split")
    if "else -1" not in profile:
        leaks.append("student profile allows a config file to retain a private fixed id")
    if leaks:
        raise SystemExit("student GUI boundary failed:\n" + "\n".join(leaks))
    print(f"student delivery boundary passed: {len(terms)} private terms absent")


if __name__ == "__main__":
    main()
