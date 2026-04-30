#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def md_files(root: Path):
    for p in sorted(root.rglob("*.md")):
        if p.is_file() and not p.name.startswith("."):
            yield p


def sha1_prefix(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.input_dir).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as wf:
        for p in md_files(root):
            text = p.read_text(encoding="utf-8", errors="ignore")
            rel = p.relative_to(root).as_posix()
            first_heading = ""
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    first_heading = line.lstrip("#").strip()
                    break
            rec = {
                "source_file": rel,
                "abs_path": str(p),
                "size_bytes": p.stat().st_size,
                "line_count": text.count("\n") + 1 if text else 0,
                "char_count": len(text),
                "first_heading": first_heading,
                "sha1_prefix": sha1_prefix(p),
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"files": count, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
