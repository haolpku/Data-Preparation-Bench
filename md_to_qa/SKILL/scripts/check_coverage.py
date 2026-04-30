#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

VALID_STATUS = {"kept", "skipped"}
VALID_SAMPLE_TYPES = {"concept_qa", "process_qa", "case_application"}


def read_jsonl(paths):
    for name in paths:
        p = Path(name)
        if not p.exists() or not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, line_no, json.loads(line)
                except Exception:
                    raise ValueError(f"invalid json in {p} at line {line_no}")


def status_total(obj):
    if "total_sample_count" in obj:
        return int(obj.get("total_sample_count", 0) or 0)
    return (
        int(obj.get("concept_count", 0) or 0)
        + int(obj.get("process_count", 0) or 0)
        + int(obj.get("case_count", 0) or 0)
        + int(obj.get("qa_count", 0) or 0)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_files", nargs="+")
    ap.add_argument("--status", required=True)
    ap.add_argument("--qa", nargs="*", default=[])
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    all_chunks = {}
    per_source_total = Counter()
    for _, _, obj in read_jsonl(args.chunk_files):
        chunk_id = obj.get("chunk_id")
        source_file = obj.get("source_file", "")
        if not chunk_id:
            continue
        all_chunks[chunk_id] = source_file
        per_source_total[source_file] += 1

    status_by_chunk = {}
    duplicates_in_status = 0
    skip_reason_counts = Counter()
    status_counts_by_type = {}
    total_count_from_status = Counter()
    for _, _, obj in read_jsonl([args.status]):
        chunk_id = obj.get("chunk_id")
        status = obj.get("status")
        if not chunk_id or status not in VALID_STATUS:
            continue
        if chunk_id in status_by_chunk:
            duplicates_in_status += 1
        status_by_chunk[chunk_id] = obj
        if status == "skipped":
            skip_reason_counts[obj.get("skip_reason", "") or "unspecified"] += 1
        status_counts_by_type[chunk_id] = {
            "concept_qa": int(obj.get("concept_count", obj.get("qa_count", 0)) or 0),
            "process_qa": int(obj.get("process_count", 0) or 0),
            "case_application": int(obj.get("case_count", 0) or 0),
        }
        total_count_from_status[chunk_id] = status_total(obj)

    actual_counts_by_type = Counter()
    actual_total_counts = Counter()
    sample_type_totals = Counter()
    for _, _, obj in read_jsonl(args.qa):
        chunk_id = obj.get("chunk_id")
        sample_type = obj.get("sample_type", "concept_qa")
        if not chunk_id:
            continue
        if sample_type not in VALID_SAMPLE_TYPES:
            sample_type = "concept_qa"
        actual_counts_by_type[(chunk_id, sample_type)] += 1
        actual_total_counts[chunk_id] += 1
        sample_type_totals[sample_type] += 1

    kept_chunks = 0
    skipped_chunks = 0
    unprocessed_chunks = []
    status_not_in_chunks = []
    sample_without_status = []
    sample_status_mismatch = []
    per_source_kept = Counter()
    per_source_skipped = Counter()
    per_source_unprocessed = Counter()

    for chunk_id, source_file in all_chunks.items():
        status_obj = status_by_chunk.get(chunk_id)
        if not status_obj:
            unprocessed_chunks.append(chunk_id)
            per_source_unprocessed[source_file] += 1
            continue
        status = status_obj.get("status")
        if status == "kept":
            kept_chunks += 1
            per_source_kept[source_file] += 1
        elif status == "skipped":
            skipped_chunks += 1
            per_source_skipped[source_file] += 1

        actual_total = actual_total_counts.get(chunk_id, 0)
        stated_total = total_count_from_status.get(chunk_id, 0)
        stated_by_type = status_counts_by_type.get(chunk_id, {})
        actual_by_type = {
            "concept_qa": actual_counts_by_type.get((chunk_id, "concept_qa"), 0),
            "process_qa": actual_counts_by_type.get((chunk_id, "process_qa"), 0),
            "case_application": actual_counts_by_type.get(
                (chunk_id, "case_application"), 0
            ),
        }
        if actual_total != stated_total or actual_by_type != stated_by_type:
            sample_status_mismatch.append(
                {
                    "chunk_id": chunk_id,
                    "status_total_sample_count": stated_total,
                    "actual_total_sample_count": actual_total,
                    "status_counts_by_type": stated_by_type,
                    "actual_counts_by_type": actual_by_type,
                }
            )

    for chunk_id in status_by_chunk:
        if chunk_id not in all_chunks:
            status_not_in_chunks.append(chunk_id)
    for chunk_id, count in actual_total_counts.items():
        if chunk_id not in status_by_chunk:
            sample_without_status.append({"chunk_id": chunk_id, "sample_count": count})

    coverage_ratio = (
        (len(all_chunks) - len(unprocessed_chunks)) / len(all_chunks)
        if all_chunks
        else 1.0
    )

    per_source = []
    for source_file in sorted(per_source_total):
        total = per_source_total[source_file]
        kept = per_source_kept[source_file]
        skipped = per_source_skipped[source_file]
        unprocessed = per_source_unprocessed[source_file]
        per_source.append(
            {
                "source_file": source_file,
                "total_chunks": total,
                "kept_chunks": kept,
                "skipped_chunks": skipped,
                "unprocessed_chunks": unprocessed,
                "coverage_ratio": (total - unprocessed) / total if total else 1.0,
            }
        )

    result = {
        "total_chunks": len(all_chunks),
        "processed_chunks": len(all_chunks) - len(unprocessed_chunks),
        "kept_chunks": kept_chunks,
        "skipped_chunks": skipped_chunks,
        "unprocessed_chunks": len(unprocessed_chunks),
        "coverage_ratio": coverage_ratio,
        "duplicates_in_status": duplicates_in_status,
        "skip_reason_counts": dict(skip_reason_counts),
        "sample_type_totals": dict(sample_type_totals),
        "status_not_in_chunks_preview": status_not_in_chunks[:100],
        "sample_without_status_preview": sample_without_status[:100],
        "sample_status_mismatch_preview": sample_status_mismatch[:100],
        "unprocessed_chunk_preview": unprocessed_chunks[:100],
        "per_source": per_source,
    }

    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
