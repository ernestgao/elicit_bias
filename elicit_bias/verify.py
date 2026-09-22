"""Offline syntax, local-import, secret-pattern and submission-inventory checks."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .common import PACKAGE

PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "token-pattern": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|AKIA[A-Z0-9]{16})\b"),
    "email": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "private-absolute-path": re.compile(r"/(?:home|Users|net|scratch)/[A-Za-z0-9_.-]+"),
    "credential-assignment": re.compile(r'''(?i)(?:api[_-]?key|password|secret|(?:access[_-]?|auth[_-]?|hf[_-]?)?token|authorization)["']?\s*[:=]\s*["'](?!TEST_ONLY_PLACEHOLDER["']|YOUR_[A-Z_]+["']|<[A-Z_]+>["'])([^\s"']{8,})["']'''),
}

MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 5_000_000
PUBLIC_MODULES = {
    "__init__", "analyze", "build_preferences", "cloud", "common", "generate",
    "metrics", "neutralize", "prepare_sft", "rubrics", "score", "train_dpo",
    "train_sft", "verify",
}
PUBLIC_PROMPTS = {
    "sft_system.txt", "target.txt", "target_synthesis.txt", "judge/original.txt",
    "strategies/S1_persona.txt", "strategies/S2_hypothetical.txt",
    "strategies/S3_moral.txt", "strategies/S4_authority.txt",
    "strategies/S5_contradiction.txt", "strategies/S6_refusal_suppression.txt",
}
RELEASE_FILES = (
    {".gitignore", "README.md", "MODEL_CARD.md", "LICENSE", "LICENSE.md", "pyproject.toml",
     "elicit_bias/taxonomy.yaml", "tests/test_method.py"}
    | {f"elicit_bias/{name}.py" for name in PUBLIC_MODULES}
    | {f"elicit_bias/prompts/{name}" for name in PUBLIC_PROMPTS}
)


def release_file(rel):
    """Only the reviewed source files, documentation and package prompts."""
    path = Path(rel)
    if path.parts[0] == ".git":
        return True  # Metadata is still scanned for credentials and binary files.
    return rel in RELEASE_FILES


def inspect(root, forbidden=()):
    findings, files, imports = [], [], set()
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"file": rel, "issue": "symlink"})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append({"file": rel, "bytes": size})
        if not release_file(rel):
            findings.append({"file": rel, "issue": "excluded-artifact"})
        if size > MAX_FILE_BYTES:
            findings.append({"file": rel, "issue": "file-size-limit"})
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"file": rel, "issue": "binary-file-requires-review"})
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": rel, "issue": label})
        if any(value.lower() in (rel + "\n" + text).lower() for value in forbidden if value):
            findings.append({"file": rel, "issue": "forbidden-identity-marker"})
        for value in re.findall(r'''https?://[^\s'"<>\\)]+''', text):
            parsed = urlsplit(value)
            host = parsed.hostname or ""
            if parsed.username or parsed.password or (host != "generativelanguage.googleapis.com" and not host.endswith("example.invalid")):
                findings.append({"file": rel, "issue": "URL-requires-review"})
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=rel)
                compile(tree, rel, "exec")
            except SyntaxError:
                findings.append({"file": rel, "issue": "syntax"})
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = path.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        names = [node.module] if node.module else [a.name for a in node.names]
                        for name in names:
                            destination = base.joinpath(*name.split("."))
                            if not destination.with_suffix(".py").is_file() and not (destination / "__init__.py").is_file():
                                findings.append({"file": rel, "issue": "unresolved-relative-import"})
                    elif node.module:
                        imports.add(node.module.split(".")[0])
    if sum(f["bytes"] for f in files) > MAX_TOTAL_BYTES:
        findings.append({"file": ".", "issue": "release-size-limit"})
    return {"passed": not findings, "findings": findings, "files": files, "file_count": len(files),
            "total_bytes": sum(f["bytes"] for f in files), "largest_files": sorted(files, key=lambda f: -f["bytes"])[:5],
            "absolute_import_roots": sorted(imports),
            "scope": "Heuristic text/filename scan including hidden files; findings never contain matched values"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PACKAGE.parent)
    parser.add_argument("--forbid", action="append", default=[], help="Additional private identity marker; never printed")
    args = parser.parse_args()
    report = inspect(args.root, args.forbid)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
