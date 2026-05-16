#!/usr/bin/env python3
"""
Calculate benchmark scores from a single judge_results.jsonl file.

Reads a JSONL file where each record contains judge results with metadata
(domain, source/dataset_name). Aggregates results per domain/benchmark and
outputs a flat JSON with scores.

Usage:
    python extract_score.py --input judge_results.jsonl --output scores.json
"""
import argparse
import json
import string
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Suppress sklearn warnings about y_pred containing unseen classes
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def normalize_text(text: str) -> str:
    """Mirror legalbench/evaluation.py normalize(stem=False)."""
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    text = text.lower()
    return text


# ---------------------------------------------------------------------------
# FinCDM
# ---------------------------------------------------------------------------
def aggregate_fincdm(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("judge_score") == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# XFinBench
# ---------------------------------------------------------------------------
def aggregate_xfinbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("judge_score") == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# legalbench
# ---------------------------------------------------------------------------
def aggregate_legalbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    y_true: List[str] = []
    y_pred: List[str] = []
    for r in records:
        meta = r.get("judge_meta", {})
        gold = meta.get("standard_answer", "")
        cleaned = meta.get("cleaned_pred", "")
        final_pred = cleaned
        if meta.get("was_calibrated") and "calibrated_answer" in meta:
            final_pred = meta["calibrated_answer"]
        y_true.append(normalize_text(gold) if gold else "")
        y_pred.append(normalize_text(final_pred) if final_pred else "")

    total = len(records)
    try:
        score = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        score = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total else 0.0
    return {"score": score, "metric": "balanced_accuracy", "total": total}


# ---------------------------------------------------------------------------
# lex-glue helpers (mirror scripts/compute_accuracy.py)
# ---------------------------------------------------------------------------
def _lexglue_normalize_label(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _lexglue_normalize_list(xs: Any) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, list):
        return sorted({_lexglue_normalize_label(x) for x in xs if _lexglue_normalize_label(x)})
    if isinstance(xs, str):
        parts = [p.strip() for p in xs.replace(";", ",").split(",")]
        return sorted({_lexglue_normalize_label(p) for p in parts if _lexglue_normalize_label(p)})
    return [_lexglue_normalize_label(xs)]


def _is_empty_pred(pred: Any) -> bool:
    if pred is None:
        return True
    if isinstance(pred, list):
        return len(pred) == 0
    if isinstance(pred, str):
        return pred.strip() == ""
    return True


def _is_parse_failure(gold: Any, pred: Any) -> bool:
    if isinstance(gold, list):
        if pred is None:
            return True
        if isinstance(pred, list):
            return False
        return True
    if pred is None:
        return True
    if isinstance(pred, list):
        return True
    if isinstance(pred, str) and pred.strip() == "":
        return True
    return False


def aggregate_lexglue(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    task_name = records[0].get("judge_meta", {}).get("dataset_name", "")

    if task_name == "ledgar":
        valid = []
        for r in records:
            meta = r.get("judge_meta", {})
            gold = meta.get("standard_answer", "")
            pred = meta.get("parsed_answer", "")
            if _is_parse_failure(gold, pred) or _is_empty_pred(pred):
                continue
            valid.append(r)
        total = len(valid)
        correct = sum(1 for r in valid if r.get("judge_score") == 1.0)
        return {
            "score": correct / total if total else 0.0,
            "metric": "accuracy",
            "total": total,
            "correct": correct,
            "skipped": len(records) - len(valid),
        }
    else:
        # eurlex / unfair_tos
        total_gold_labels = 0
        matched_gold_labels = 0
        valid_samples = 0
        skipped = 0
        for r in records:
            meta = r.get("judge_meta", {})
            gold = meta.get("standard_answer", [])
            pred = meta.get("parsed_answer", [])
            if _is_parse_failure(gold, pred) or _is_empty_pred(pred):
                skipped += 1
                continue
            gold_set = _lexglue_normalize_list(gold)
            pred_set = _lexglue_normalize_list(pred)
            total_gold_labels += len(gold_set)
            if gold_set and pred_set:
                matched_gold_labels += len(set(gold_set) & set(pred_set))
            valid_samples += 1
        score = matched_gold_labels / total_gold_labels if total_gold_labels else 0.0
        return {
            "score": score,
            "metric": "multilabel_accuracy",
            "total_samples": valid_samples,
            "total_gold_labels": total_gold_labels,
            "matched_gold_labels": matched_gold_labels,
            "skipped": skipped,
        }


# ---------------------------------------------------------------------------
# MedCaseReasoning
# ---------------------------------------------------------------------------
def aggregate_medcasereasoning(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("judge_score") == 1.0)
    avg_recall = sum(r.get("judge_meta", {}).get("reasoning_recall", 0.0) for r in records) / total if total else 0.0
    return {"accuracy": correct / total if total else 0.0, "recall": avg_recall, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# MedmcQA
# ---------------------------------------------------------------------------
def aggregate_medmcqa(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r.get("judge_score") == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# MedRBench
# ---------------------------------------------------------------------------
def aggregate_medrbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if not r.get("should_retry", True)]
    total = len(valid)
    correct = sum(1 for r in valid if r.get("judge_score") == 1.0)
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "skipped": len(records) - len(valid),
    }


AGGREGATORS = {
    "fincdm": aggregate_fincdm,
    "xfinbench": aggregate_xfinbench,
    "legalbench": aggregate_legalbench,
    "lexglue": aggregate_lexglue,
    "medcasereasoning": aggregate_medcasereasoning,
    "medmcqa": aggregate_medmcqa,
    "medrbench": aggregate_medrbench,
}

DOMAIN_BENCHMARKS = {
    "business": ["fincdm", "xfinbench"],
    "law": ["legalbench", "lexglue"],
    "medicine": ["medcasereasoning", "medmcqa", "medrbench"],
}


def load_jsonl(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Read a judge_results.jsonl file and return list of records.
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: {file_path}:{line_num} JSON decode error: {e}", file=sys.stderr)
                continue
            # Ensure required keys exist to avoid later crashes
            if "judge_score" not in record:
                if verbose:
                    print(f"Warning: {file_path}:{line_num} missing 'judge_score', skipping", file=sys.stderr)
                continue
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate benchmark scores from a single judge_results.jsonl file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a judge_results.jsonl file."
    )
    parser.add_argument(
        "--output",
        default="calculated_scores.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed warnings and progress"
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.is_file():
        print(f"Error: {input_file} is not a file", file=sys.stderr)
        sys.exit(1)

    # Load all records from the jsonl file
    records = load_jsonl(input_file, verbose=args.verbose)
    print(f"Loaded {len(records)} records from {input_file}", file=sys.stderr)

    # Group records by source -> dataset_name
    source_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        orig_meta = record.get("original_data", {}).get("metadata", {})
        source = orig_meta.get("source", "unknown")
        dataset_name = record.get("judge_meta", {}).get("dataset_name", source)
        source_dict[source][dataset_name].append(record)

    # Aggregate per domain -> flat output
    output: Dict[str, Any] = {
        "input_file": str(input_file),
        "total_records": len(records),
    }

    for domain, benchmarks in DOMAIN_BENCHMARKS.items():
        domain_correct = 0
        domain_total = 0
        domain_scores: Dict[str, Any] = {}

        for source in benchmarks:
            dataset_dict = source_dict.get(source)
            if not dataset_dict:
                continue

            aggregator = AGGREGATORS[source]
            if len(dataset_dict) == 1 and list(dataset_dict.keys())[0] == source:
                recs = list(dataset_dict.values())[0]
                bench_result = aggregator(recs)
                domain_scores[source] = bench_result

                count_records = recs
                if source == "medrbench":
                    count_records = [r for r in recs if not r.get("should_retry", True)]
                domain_correct += sum(1 for r in count_records if r.get("judge_score") == 1.0)
                domain_total += len(count_records)
            else:
                bench_details: Dict[str, Any] = {}
                all_records: List[Dict[str, Any]] = []
                for dataset_name, recs in dataset_dict.items():
                    bench_details[dataset_name] = aggregator(recs)
                    all_records.extend(recs)

                count_records = all_records
                if source == "medrbench":
                    count_records = [r for r in all_records if not r.get("should_retry", True)]
                domain_correct += sum(1 for r in count_records if r.get("judge_score") == 1.0)
                domain_total += len(count_records)

                if source == "fincdm":
                    tot = sum(r.get("total", 0) for r in bench_details.values())
                    corr = sum(r.get("correct", 0) for r in bench_details.values())
                    bench_details["_overall"] = {
                        "accuracy": corr / tot if tot else 0.0,
                        "total": tot,
                        "correct": corr,
                    }
                elif source == "legalbench":
                    tot = sum(r.get("total", 0) for r in bench_details.values())
                    weighted = sum(r.get("score", 0) * r.get("total", 0) for r in bench_details.values())
                    bench_details["_overall"] = {
                        "score": weighted / tot if tot else 0.0,
                        "metric": "balanced_accuracy",
                        "total": tot,
                    }
                elif source == "lexglue":
                    scores = []
                    for subresult in bench_details.values():
                        score = subresult.get("score")
                        if score is not None:
                            scores.append(score)
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    bench_details["_overall"] = {
                        "score": avg_score,
                        "metric": "average_of_subsets",
                    }
                domain_scores[source] = bench_details

        if domain_total > 0:
            output[domain] = {
                "overall_accuracy": domain_correct / domain_total,
                "total": domain_total,
                "correct": domain_correct,
                "benchmarks": domain_scores,
            }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved scores to {args.output}")

    # Print summary to stdout
    for domain in DOMAIN_BENCHMARKS:
        if domain in output:
            info = output[domain]
            print(f"  {domain}: overall_accuracy={info['overall_accuracy']:.4f} ({info['correct']}/{info['total']})")
            for bench_name, bench_data in info["benchmarks"].items():
                if isinstance(bench_data, dict) and "_overall" in bench_data:
                    overall = bench_data["_overall"]
                    s = overall.get("accuracy") or overall.get("score", 0)
                    print(f"    {bench_name}: {s:.4f} (aggregated)")
                elif isinstance(bench_data, dict):
                    s = bench_data.get("accuracy") or bench_data.get("score", 0)
                    print(f"    {bench_name}: {s:.4f}")


if __name__ == "__main__":
    main()
