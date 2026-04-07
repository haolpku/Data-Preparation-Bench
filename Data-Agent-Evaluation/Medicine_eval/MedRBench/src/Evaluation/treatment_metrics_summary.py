#!/usr/bin/env python3
"""
Summarize treatment planning accuracy folders to highlight LLM capabilities.

This script targets directories like
`data/EvalResults/acc_results_treatment_qwen2.5-7b-medicine-full-claude-detiled-dolly`.
Each contains one or more model sub-folders with JSON files per case that were
generated after running `oracle_treatment_planning.py` and then
`treatment_final_accuracy.py`.

For every model directory the script reports aggregated indicators:
    * overall correctness, coverage, answer length stats;
    * macro/per-category accuracy for body & disorder tags;
    * variety metrics (unique normalized outputs);
    * representative failure cases to inspect qualitative behavior.
"""

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
        description="Compute capability metrics for treatment accuracy folders."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/EvalResults",
        help="Root directory that hosts `acc_results_*` folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="acc_results_treatment_*",
        help="Glob pattern to pick target folders under base-dir.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Explicit folder paths to summarize; overrides base-dir scanning.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Depth to descend when searching for JSON case directories.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=3,
        help="Number of sample mistakes to show for each dataset.",
    )
    parser.add_argument(
        "--relative-to",
        type=str,
        default=None,
        help="If set, dataset paths are displayed relative to this directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump aggregated metrics as JSON.",
    )
    return parser.parse_args()


def gather_roots(args: argparse.Namespace) -> List[Path]:
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
    if max_depth < 0 or not root.exists():
        return []

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        print(f"[warn] cannot list {root}: {exc}", file=sys.stderr)
        return []

    has_json = any(entry.is_file() and entry.suffix.lower() == ".json" for entry in entries)
    if has_json:
        return [root]

    if max_depth == 0:
        return []

    case_dirs: List[Path] = []
    for entry in entries:
        if entry.is_dir():
            case_dirs.extend(enumerate_case_dirs(entry, max_depth - 1))

    ordered_unique = list(dict.fromkeys(case_dirs))
    return ordered_unique


def normalize_categories(values: Optional[Iterable[str]], fallback: str) -> List[str]:
    if not values:
        return [fallback]
    cleaned = [item.strip() for item in values if isinstance(item, str) and item.strip()]
    return cleaned or [fallback]


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def truncate(text: str, limit: int = 200) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def load_cases(case_dir: Path) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for json_path in sorted(case_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to read {json_path}: {exc}", file=sys.stderr)
            continue

        case_id = data.get("id") or json_path.stem
        result_obj = data.get("result") or {}
        prediction = str(result_obj.get("content", "") or "").strip()
        reasoning = str(result_obj.get("reasoning", "") or "").strip()
        gt = str(
            data.get("generate_case", {}).get("treatment_plan_results", "") or ""
        ).strip()

        case_record = {
            "id": case_id,
            "is_correct": bool(data.get("accuracy")),
            "body_categories": normalize_categories(
                data.get("body_category"), "Unspecified Body"
            ),
            "disorder_categories": normalize_categories(
                data.get("disorder_category"), "Unspecified Disorder"
            ),
            "prediction": prediction,
            "reasoning": reasoning,
            "ground_truth": gt,
        }
        case_record["has_prediction"] = bool(prediction)
        case_record["has_reasoning"] = bool(reasoning)
        case_record["answer_length"] = len(prediction)
        case_record["normalized_prediction"] = (
            normalize_answer(prediction) if prediction else ""
        )
        cases.append(case_record)
    if not cases:
        print(f"[warn] no JSON cases found under {case_dir}", file=sys.stderr)
    return cases


def build_category_breakdown(cases: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cases": 0, "correct": 0})
    for case in cases:
        for category in case[key]:
            stats[category]["cases"] += 1
            if case["is_correct"]:
                stats[category]["correct"] += 1

    breakdown = []
    for category, info in stats.items():
        total = info["cases"]
        accuracy = info["correct"] / total if total else 0.0
        breakdown.append(
            {
                "category": category,
                "cases": total,
                "correct": info["correct"],
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
    total = len(cases)
    correct = sum(1 for case in cases if case["is_correct"])
    with_answers = sum(1 for case in cases if case["has_prediction"])
    with_reasoning = sum(1 for case in cases if case["has_reasoning"])
    lengths = [case["answer_length"] for case in cases if case["has_prediction"]]

    body_breakdown = build_category_breakdown(cases, "body_categories")
    disorder_breakdown = build_category_breakdown(cases, "disorder_categories")

    def macro_avg(breakdown: List[Dict[str, object]]) -> Optional[float]:
        if not breakdown:
            return None
        return round(sum(item["accuracy"] for item in breakdown) / len(breakdown), 4)

    error_examples = []
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
        "num_cases": total,
        "num_correct": correct,
        "overall_accuracy": round(correct / total, 4) if total else None,
        "answer_coverage": round(with_answers / total, 4) if total else None,
        "reasoning_coverage": round(with_reasoning / total, 4) if total else None,
        "average_answer_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "median_answer_length": round(statistics.median(lengths), 2) if lengths else 0.0,
        "unique_predictions": len(
            {
                case["normalized_prediction"]
                for case in cases
                if case["normalized_prediction"]
            }
        ),
        "macro_body_accuracy": macro_avg(body_breakdown),
        "macro_disorder_accuracy": macro_avg(disorder_breakdown),
        "body_category_breakdown": body_breakdown,
        "disorder_category_breakdown": disorder_breakdown,
        "error_examples": error_examples,
    }


def pretty_path(path: Path, relative_to: Optional[Path]) -> str:
    abs_path = path.resolve()
    if relative_to:
        try:
            return str(abs_path.relative_to(relative_to))
        except ValueError:
            pass
    return str(abs_path)


def main() -> None:
    args = parse_args()
    relative_root = Path(args.relative_to).resolve() if args.relative_to else None

    roots = gather_roots(args)
    if not roots:
        print("[warn] no folders matched the selection criteria", file=sys.stderr)
        return

    outputs = []
    for root in roots:
        case_dirs = enumerate_case_dirs(root, args.max_depth)
        if not case_dirs:
            print(f"[warn] skipping {root}: no case files found", file=sys.stderr)
            continue
        for case_dir in case_dirs:
            cases = load_cases(case_dir)
            if not cases:
                continue
            metrics = compute_metrics(case_dir, cases, top_errors=args.top_errors)
            metrics["dataset"] = pretty_path(case_dir, relative_root)
            outputs.append(metrics)

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

            def preview(name: str) -> None:
                head = metrics[name][:5]
                label = name.replace("_", " ").title()
                if not head:
                    print(f"{label}: no category info")
                    return
                print(f"{label} (top {len(head)}):")
                for item in head:
                    print(
                        f"  - {item['category']}: {item['accuracy']:.4f} "
                        f"({item['correct']}/{item['cases']})"
                    )

            preview("body_category_breakdown")
            preview("disorder_category_breakdown")

            if metrics["error_examples"]:
                print("Sample errors:")
                for err in metrics["error_examples"]:
                    print(
                        f"  - {err['case_id']}: pred='{err['prediction']}' "
                        f"| gt='{err['ground_truth']}'"
                    )
            print()

    if args.output_json and outputs:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] metrics saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
