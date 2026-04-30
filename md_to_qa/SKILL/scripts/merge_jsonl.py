#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as wf:
        for name in sorted(args.inputs):
            p = Path(name)
            if not p.exists() or not p.is_file():
                continue
            with p.open("r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        wf.write(line.rstrip("\n") + "\n")
                        written += 1
    print(f"merged_lines={written} output={out}")


if __name__ == "__main__":
    main()
