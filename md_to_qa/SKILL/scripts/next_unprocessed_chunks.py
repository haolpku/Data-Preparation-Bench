#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

VALID_STATUS = {"kept", "skipped"}


def load_processed(status_path: Path):
    processed = {}
    if not status_path.exists():
        return processed
    with status_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                raise ValueError(f"invalid json in status file at line {line_no}")
            chunk_id = obj.get("chunk_id")
            status = obj.get("status")
            if not chunk_id or status not in VALID_STATUS:
                continue
            processed[chunk_id] = status
    return processed


def iter_chunks(paths):
    for name in paths:
        p = Path(name)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    raise ValueError(
                        f"invalid json in chunk file {p} at line {line_no}"
                    )
                yield obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_files", nargs="+")
    ap.add_argument("--status", required=True)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--output")
    args = ap.parse_args()

    processed = load_processed(Path(args.status))
    selected = []
    total_unprocessed = 0
    for obj in iter_chunks(args.chunk_files):
        chunk_id = obj.get("chunk_id")
        if not chunk_id:
            continue
        if chunk_id in processed:
            continue
        total_unprocessed += 1
        if len(selected) < args.limit:
            selected.append(obj)

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as wf:
            for obj in selected:
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "selected": len(selected),
                "unprocessed_before_selection": total_unprocessed,
                "remaining_after_selection": max(0, total_unprocessed - len(selected)),
                "output": str(Path(args.output).resolve()) if args.output else "",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
