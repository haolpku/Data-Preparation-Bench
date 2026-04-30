#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def split_sections(text: str):
    lines = text.splitlines()
    sections = []
    current = {"title_path": [], "lines": [], "start_line": 1}
    title_stack = []

    for idx, line in enumerate(lines, start=1):
        m = HEADING_RE.match(line)
        if m:
            if current["lines"]:
                current["end_line"] = idx - 1
                sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            title_stack = title_stack[: level - 1] + [title]
            current = {
                "title_path": title_stack.copy(),
                "lines": [line],
                "start_line": idx,
            }
        else:
            current["lines"].append(line)
    if current["lines"]:
        current["end_line"] = len(lines)
        sections.append(current)
    return sections


def chunk_paragraphs(text: str, max_chars: int):
    paras = re.split(r"\n\s*\n", text)
    chunks, buf = [], []
    cur_len = 0
    for para in paras:
        para = para.strip("\n")
        if not para:
            continue
        extra = len(para) + (2 if buf else 0)
        if buf and cur_len + extra > max_chars:
            chunks.append("\n\n".join(buf))
            buf, cur_len = [para], len(para)
        else:
            buf.append(para)
            cur_len += extra
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def normalize_source_file(path: Path, source_root: Path | None):
    if source_root is None:
        return path.name
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except Exception:
        return path.name


def emit_record(
    wf,
    p: Path,
    source_file: str,
    book_title: str,
    sec_idx: int,
    part_idx: int,
    title_path,
    start_line,
    end_line,
    text,
    merged_title_paths=None,
):
    rec = {
        "source_file": source_file,
        "book_title": book_title,
        "chunk_id": f"{p.stem}_{sec_idx:04d}_{part_idx:02d}",
        "title_path": title_path,
        "start_line": start_line,
        "end_line": end_line,
        "char_count": len(text),
        "text": text,
    }
    if merged_title_paths:
        rec["merged_title_paths"] = merged_title_paths
    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--min-chars", type=int, default=1200)
    ap.add_argument("--source-root")
    args = ap.parse_args()

    p = Path(args.input_md).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text(encoding="utf-8", errors="ignore")
    sections = split_sections(text)
    source_root = Path(args.source_root).resolve() if args.source_root else None
    source_file = normalize_source_file(p, source_root)

    book_title = p.stem
    for s in sections:
        if s["title_path"]:
            book_title = s["title_path"][0]
            break

    written = 0
    pending = None

    with out.open("w", encoding="utf-8") as wf:
        for sec_idx, sec in enumerate(sections, start=1):
            raw = "\n".join(sec["lines"]).strip()
            if not raw:
                continue

            if pending is not None:
                combined = pending["raw"] + "\n\n" + raw
                if (
                    len(pending["raw"]) < args.min_chars
                    and len(combined) <= args.max_chars
                ):
                    pending["raw"] = combined
                    pending["end_line"] = sec["end_line"]
                    pending["merged_title_paths"].append(sec.get("title_path", []))
                    continue
                parts = chunk_paragraphs(pending["raw"], args.max_chars)
                for part_idx, part in enumerate(parts, start=1):
                    emit_record(
                        wf,
                        p,
                        source_file,
                        book_title,
                        pending["sec_idx"],
                        part_idx,
                        pending["title_path"],
                        pending["start_line"],
                        pending["end_line"],
                        part,
                        merged_title_paths=(
                            pending["merged_title_paths"]
                            if len(pending["merged_title_paths"]) > 1
                            else None
                        ),
                    )
                    written += 1
                pending = None

            pending = {
                "raw": raw,
                "sec_idx": sec_idx,
                "title_path": sec.get("title_path", []),
                "start_line": sec.get("start_line"),
                "end_line": sec.get("end_line"),
                "merged_title_paths": [sec.get("title_path", [])],
            }

        if pending is not None:
            parts = chunk_paragraphs(pending["raw"], args.max_chars)
            for part_idx, part in enumerate(parts, start=1):
                emit_record(
                    wf,
                    p,
                    source_file,
                    book_title,
                    pending["sec_idx"],
                    part_idx,
                    pending["title_path"],
                    pending["start_line"],
                    pending["end_line"],
                    part,
                    merged_title_paths=(
                        pending["merged_title_paths"]
                        if len(pending["merged_title_paths"]) > 1
                        else None
                    ),
                )
                written += 1

    print(json.dumps({"chunks": written, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
