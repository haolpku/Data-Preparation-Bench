#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MedRBench diagnose accuracy folders into metrics."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/EvalResults",
        help="Root directory that contains `acc_results_*` folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="acc_results_diagnose_*",
        help="Glob-style pattern that folder names must match when scanning base-dir.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Optional explicit folders to summarize. "
        "When omitted the script scans base-dir with the provided pattern.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=3,
        help="How many error examples to keep per dataset.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth to search inside each matched folder for JSON case files.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the aggregated metrics as JSON.",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        default=None,
        help="If set, dataset names are made relative to this directory to keep output tidy.",
    )
    return parser.parse_args()


def gather_target_roots(args: argparse.Namespace) -> List[Path]:
    if args.folders:
        return [Path(folder).resolve() for folder in args.folders]

    base_path = Path(args.base_dir).resolve()
    if not base_path.exists():
        print(f"[warn] base directory {base_path} does not exist", file=sys.stderr)
        return []

    roots: List[Path] = []
    for child in sorted(base_path.iterdir()):
        if child.is_dir() and fnmatch(child.name, args.pattern):
            roots.append(child)
    return roots


def enumerate_case_dirs(root: Path, max_depth: int) -> List[Path]:
    """
    Return directories under `root` that directly store JSON case files.

    The traversal stops as soon as JSON files are found to avoid descending further.
    """
    results: List[Path] = []
    if max_depth < 0 or not root.exists():
        return results

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        print(f"[warn] cannot list {root}: {exc}", file=sys.stderr)
        return results

    has_json = any(entry.is_file() and entry.suffix.lower() == ".json" for entry in entries)
    if has_json:
        return [root]

    if max_depth == 0:
        return results

    for entry in entries:
        if entry.is_dir():
            results.extend(enumerate_case_dirs(entry, max_depth - 1))

    # Deduplicate while preserving order
    dedup: Dict[Path, None] = dict.fromkeys(results, None)
    return list(dedup.keys())


def normalize_categories(raw_value: Optional[Iterable[str]], fallback: str) -> List[str]:
    if not raw_value:
        return [fallback]
    categories: List[str] = []
    for item in raw_value:
        if isinstance(item, str) and item.strip():
            categories.append(item.strip())
    return categories or [fallback]


def truncate(text: str, limit: int = 200) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def load_cases(case_dir: Path) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    json_files = sorted(case_dir.glob("*.json"))
    if not json_files:
        print(f"[warn] no JSON files found in {case_dir}", file=sys.stderr)
        return cases

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to load {json_path}: {exc}", file=sys.stderr)
            continue

        case_id = data.get("id") or json_path.stem
        result_obj = data.get("result", {}) or {}
        prediction_raw = str(result_obj.get("out_answer", "") or "")
        reasoning_raw = str(result_obj.get("out_reasoning", "") or "")
        gt_answer = str(
            data.get("generate_case", {}).get("diagnosis_results", "")
        )
        case_record = {
            "id": case_id,
            "is_correct": bool(data.get("accuracy")),
            "body_categories": normalize_categories(
                data.get("body_category"), "Unspecified Body"
            ),
            "disorder_categories": normalize_categories(
                data.get("disorder_category"), "Unspecified Disorder"
            ),
            "prediction": prediction_raw.strip(),
            "reasoning": reasoning_raw.strip(),
            "ground_truth": gt_answer.strip(),
        }
        case_record["has_prediction"] = bool(case_record["prediction"])
        case_record["has_reasoning"] = bool(case_record["reasoning"])
        case_record["answer_length"] = len(case_record["prediction"])
        case_record["normalized_prediction"] = (
            normalize_answer(case_record["prediction"])
            if case_record["prediction"]
            else ""
        )
        cases.append(case_record)
    return cases


def build_category_breakdown(cases: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cases": 0, "correct": 0})
    for case in cases:
        for category in case[key]:
            stats[category]["cases"] += 1
            if case["is_correct"]:
                stats[category]["correct"] += 1

    breakdown: List[Dict[str, object]] = []
    for category, values in stats.items():
        cases_count = values["cases"]
        accuracy = values["correct"] / cases_count if cases_count else 0.0
        breakdown.append(
            {
                "category": category,
                "cases": cases_count,
                "correct": values["correct"],
                "accuracy": round(accuracy, 4),
            }
        )

    breakdown.sort(key=lambda item: item["cases"], reverse=True)
    return breakdown


def compute_metrics(
    dataset_dir: Path,
    cases: List[Dict[str, object]],
    *,
    top_errors: int,
) -> Dict[str, object]:
    total_cases = len(cases)
    correct_cases = sum(1 for case in cases if case["is_correct"])
    answer_lengths = [case["answer_length"] for case in cases if case["has_prediction"]]
    coverage = sum(1 for case in cases if case["has_prediction"])
    reasoning_coverage = sum(1 for case in cases if case["has_reasoning"])
    unique_predictions = set(
        case["normalized_prediction"]
        for case in cases
        if case["normalized_prediction"]
    )

    body_breakdown = build_category_breakdown(cases, "body_categories")
    disorder_breakdown = build_category_breakdown(cases, "disorder_categories")

    def macro_average(breakdown: List[Dict[str, object]]) -> Optional[float]:
        if not breakdown:
            return None
        return round(
            sum(item["accuracy"] for item in breakdown) / len(breakdown), 4
        )

    average_answer_length = (
        round(sum(answer_lengths) / len(answer_lengths), 2) if answer_lengths else 0.0
    )
    median_answer_length = (
        round(statistics.median(answer_lengths), 2) if answer_lengths else 0.0
    )

    error_examples: List[Dict[str, object]] = []
    for case in cases:
        if case["is_correct"]:
            continue
        error_examples.append(
            {
                "case_id": case["id"],
                "body_category": case["body_categories"],
                "disorder_category": case["disorder_categories"],
                "prediction": truncate(case["prediction"]),
                "ground_truth": truncate(case["ground_truth"]),
            }
        )
        if len(error_examples) >= top_errors:
            break

    return {
        "dataset": str(dataset_dir),
        "num_cases": total_cases,
        "num_correct": correct_cases,
        "overall_accuracy": round(correct_cases / total_cases, 4) if total_cases else None,
        "answer_coverage": round(coverage / total_cases, 4) if total_cases else None,
        "reasoning_coverage": round(reasoning_coverage / total_cases, 4) if total_cases else None,
        "average_answer_length": average_answer_length,
        "median_answer_length": median_answer_length,
        "unique_predictions": len(unique_predictions),
        "macro_body_accuracy": macro_average(body_breakdown),
        "macro_disorder_accuracy": macro_average(disorder_breakdown),
        "body_category_breakdown": body_breakdown,
        "disorder_category_breakdown": disorder_breakdown,
        "error_examples": error_examples,
    }


def format_dataset_name(path: Path, relative_to: Optional[Path]) -> str:
    absolute = path.resolve()
    if relative_to:
        try:
            return str(absolute.relative_to(relative_to))
        except ValueError:
            pass
    return str(absolute)


def main() -> None:
    args = parse_args()
    relative_to_path = Path(args.relative_to).resolve() if args.relative_to else None

    target_roots = gather_target_roots(args)
    if not target_roots:
        print("[warn] no matching folders were found", file=sys.stderr)
        return

    all_metrics = []
    for root in target_roots:
        case_dirs = enumerate_case_dirs(root, args.max_depth)
        if not case_dirs:
            print(f"[warn] no case folders found under {root}", file=sys.stderr)
            continue

        for case_dir in case_dirs:
            cases = load_cases(case_dir)
            if not cases:
                continue
            metrics = compute_metrics(case_dir, cases, top_errors=args.top_errors)
            metrics["dataset"] = format_dataset_name(case_dir, relative_to_path)
            all_metrics.append(metrics)
            print("=" * 80)
            print(f"Dataset: {metrics['dataset']}")
            print(f"Total cases: {metrics['num_cases']}")
            print(
                f"Accuracy: {metrics['overall_accuracy']:.4f} "
                f"({metrics['num_correct']}/{metrics['num_cases']})"
            )
            print(
                f"Answer coverage: {metrics['answer_coverage']:.4f}, "
                f"Reasoning coverage: {metrics['reasoning_coverage']:.4f}"
            )
            print(
                f"Answer length avg/median: "
                f"{metrics['average_answer_length']} / {metrics['median_answer_length']}"
            )
            print(
                f"Unique normalized predictions: {metrics['unique_predictions']}"
            )
            if metrics["macro_body_accuracy"] is not None:
                print(
                    f"Macro body accuracy: {metrics['macro_body_accuracy']:.4f} "
                    f"across {len(metrics['body_category_breakdown'])} categories"
                )
            if metrics["macro_disorder_accuracy"] is not None:
                print(
                    f"Macro disorder accuracy: {metrics['macro_disorder_accuracy']:.4f} "
                    f"across {len(metrics['disorder_category_breakdown'])} categories"
                )

            def preview_breakdown(breakdown_name: str) -> None:
                breakdown = metrics[breakdown_name]
                head = breakdown[:5]
                label = breakdown_name.replace("_", " ").title()
                if not head:
                    print(f"{label}: no category info")
                    return
                print(f"{label} (top {len(head)}):")
                for item in head:
                    print(
                        f"  - {item['category']}: {item['accuracy']:.4f} "
                        f"({item['correct']}/{item['cases']})"
                    )

            preview_breakdown("body_category_breakdown")
            preview_breakdown("disorder_category_breakdown")

            if metrics["error_examples"]:
                print("Sample errors:")
                for err in metrics["error_examples"]:
                    print(
                        f"  - {err['case_id']}: pred='{err['prediction']}' "
                        f"| gt='{err['ground_truth']}'"
                    )
            print()

    if args.output_json and all_metrics:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] metrics saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
