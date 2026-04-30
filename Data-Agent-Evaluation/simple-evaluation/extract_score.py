#!/usr/bin/env python3
"""
Calculate benchmark scores from judge_results.jsonl.

Strictly aligns with the aggregation logic executed by run_{domain}_eval.sh.
Output is grouped by domain:
  domain -> overall_accuracy (all samples in this domain)
  domain -> details -> benchmark-specific scores

Modified: skip domain entries where total == 0.
"""

import argparse
import json
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


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
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# XFinBench
# ---------------------------------------------------------------------------
def aggregate_xfinbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# legalbench
# ---------------------------------------------------------------------------
def aggregate_legalbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    y_true: List[str] = []
    y_pred: List[str] = []
    for r in records:
        meta = r["judge_meta"]
        gold = meta.get("standard_answer", "")
        cleaned = meta.get("cleaned_pred", "")
        final_pred = cleaned
        if meta.get("was_calibrated") and "calibrated_answer" in meta:
            final_pred = meta["calibrated_answer"]
        y_true.append(normalize_text(gold))
        y_pred.append(normalize_text(final_pred))

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
    task_name = records[0]["judge_meta"].get("dataset_name", "")

    if task_name == "ledgar":
        valid = []
        for r in records:
            meta = r["judge_meta"]
            gold = meta.get("standard_answer", "")
            pred = meta.get("parsed_answer", "")
            if _is_parse_failure(gold, pred) or _is_empty_pred(pred):
                continue
            valid.append(r)
        total = len(valid)
        correct = sum(1 for r in valid if r["judge_score"] == 1.0)
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
            meta = r["judge_meta"]
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
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    avg_recall = sum(r["judge_meta"].get("reasoning_recall", 0.0) for r in records) / total if total else 0.0
    return {"accuracy": correct / total if total else 0.0, "recall": avg_recall, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# MedmcQA
# ---------------------------------------------------------------------------
def aggregate_medmcqa(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# MedRBench
# ---------------------------------------------------------------------------
def aggregate_medrbench(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if not r.get("should_retry", True)]
    total = len(valid)
    correct = sum(1 for r in valid if r["judge_score"] == 1.0)
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "skipped": len(records) - len(valid),
    }



# ---------------------------------------------------------------------------
# MMLU-Redux (text domain)
# ---------------------------------------------------------------------------
def aggregate_mmlu_redux(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple accuracy (exact_match with ignore_case + ignore_punctuation)."""
    total = len(records)
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# Qwen2.5-Math benchmarks (math domain)
# ---------------------------------------------------------------------------
def _aggregate_qwen_math_simple(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple accuracy for qwen_math benchmarks."""
    total = len(records)
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}



# ---------------------------------------------------------------------------
# Science benchmarks (science domain)
# ---------------------------------------------------------------------------
def _aggregate_science_simple(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple accuracy for science benchmarks."""
    total = len(records)
    correct = sum(1 for r in records if r["judge_score"] == 1.0)
    return {"accuracy": correct / total if total else 0.0, "total": total, "correct": correct}


AGGREGATORS = {
    # business
    "fincdm": aggregate_fincdm,
    "xfinbench": aggregate_xfinbench,
    # law
    "legalbench": aggregate_legalbench,
    "lexglue": aggregate_lexglue,
    # medicine
    "medcasereasoning": aggregate_medcasereasoning,
    "medmcqa": aggregate_medmcqa,
    "medrbench": aggregate_medrbench,
    # text (MMLU-Redux)
    "mmlu_redux": aggregate_mmlu_redux,
    # math (Qwen2.5-Math) — all use simple accuracy
    "qwen_math_gsm8k": _aggregate_qwen_math_simple,
    "qwen_math_amc23": _aggregate_qwen_math_simple,
    "qwen_math_aime24": _aggregate_qwen_math_simple,
    "qwen_math_minerva_math": _aggregate_qwen_math_simple,
    "qwen_math_gaokao2024_mix": _aggregate_qwen_math_simple,
    "qwen_math_olympiadbench": _aggregate_qwen_math_simple,
    "qwen_math_math": _aggregate_qwen_math_simple,
    # science — individual benchmarks (actual source values)
    "science_mmlu": _aggregate_science_simple,
    "science_mmlu_pro": _aggregate_science_simple,
    "science_gpqa_diamond": _aggregate_science_simple,
    "science_gpqa_main": _aggregate_science_simple,
    "science_super_gpqa": _aggregate_science_simple,
    "science_ChemBench-multi-choise": _aggregate_science_simple,
    "science_ChemBench-str-match": _aggregate_science_simple,
    "science_piqa": _aggregate_science_simple,
    "science_scibench-physics": _aggregate_science_simple,
    "science_scibench-chemistry": _aggregate_science_simple,
    "science_scibench-math": _aggregate_science_simple,
}

DOMAIN_BENCHMARKS = {
    "business": ["fincdm", "xfinbench"],
    "law": ["legalbench", "lexglue"],
    "medicine": ["medcasereasoning", "medmcqa", "medrbench"],
    "text": ["mmlu_redux"],
    "math": [
        "qwen_math_gsm8k", "qwen_math_amc23", "qwen_math_aime24",
        "qwen_math_minerva_math", "qwen_math_gaokao2024_mix",
        "qwen_math_olympiadbench", "qwen_math_math",
    ],
    "science": [
        "science_mmlu", "science_mmlu_pro",
        "science_gpqa_diamond", "science_gpqa_main",
        "science_super_gpqa",
        "science_ChemBench-multi-choise", "science_ChemBench-str-match",
        "science_piqa",
        "science_scibench-physics", "science_scibench-chemistry", "science_scibench-math",
    ],
}

# Science benchmarks that should be merged in the LaTeX table.
# e.g., "GPQA" column = gpqa_diamond + gpqa_main averaged
# This mapping is used by extract.py to produce the final TEX columns.
SCIENCE_MERGE_MAP = {
    "GPQA": ["science_gpqa_diamond", "science_gpqa_main"],
    "ChemBench": ["science_ChemBench-multi-choise", "science_ChemBench-str-match"],
    "SciBench": ["science_scibench-physics", "science_scibench-chemistry", "science_scibench-math"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate benchmark scores from judge_results.jsonl")
    parser.add_argument("--input", default="judge_results.jsonl", help="Path to judge_results.jsonl")
    parser.add_argument("--output", default="calculated_scores.json", help="Output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Group by model_family -> training_set -> source -> dataset_name
    grouped = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    total_records = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1

            orig_meta = record.get("original_data", {}).get("metadata", {})
            model_family = orig_meta.get("model_family", "unknown")
            training_set = orig_meta.get("training_set", "unknown")
            source = orig_meta.get("source", "unknown")
            dataset_name = record.get("judge_meta", {}).get("dataset_name", source)

            grouped[model_family][training_set][source][dataset_name].append(record)

    results: Dict[str, Any] = {}
    for model_family, train_dict in grouped.items():
        results[model_family] = {}
        for training_set, source_dict in train_dict.items():
            results[model_family][training_set] = {}
            for domain, benchmarks in DOMAIN_BENCHMARKS.items():
                domain_correct = 0
                domain_total = 0
                details: Dict[str, Any] = {}

                for source in benchmarks:
                    dataset_dict = source_dict.get(source)
                    if not dataset_dict:
                        continue

                    aggregator = AGGREGATORS[source]
                    if len(dataset_dict) == 1 and list(dataset_dict.keys())[0] == source:
                        records = list(dataset_dict.values())[0]
                        bench_result = aggregator(records)
                        details[source] = bench_result

                        # Count for domain overall (only score == 1 is pass)
                        # Special case: medrbench excludes should_retry records
                        count_records = records
                        if source == "medrbench":
                            count_records = [r for r in records if not r.get("should_retry", True)]
                        domain_correct += sum(1 for r in count_records if r["judge_score"] == 1.0)
                        domain_total += len(count_records)
                    else:
                        bench_details: Dict[str, Any] = {}
                        all_records: List[Dict[str, Any]] = []
                        for dataset_name, records in dataset_dict.items():
                            bench_details[dataset_name] = aggregator(records)
                            all_records.extend(records)

                        # Count for domain overall
                        count_records = all_records
                        if source == "medrbench":
                            count_records = [r for r in all_records if not r.get("should_retry", True)]
                        domain_correct += sum(1 for r in count_records if r["judge_score"] == 1.0)
                        domain_total += len(count_records)

                        # Add overall convenience key for fincdm / legalbench inside details
                        if source == "fincdm":
                            tot = sum(r["total"] for r in bench_details.values())
                            corr = sum(r.get("correct", 0) for r in bench_details.values())
                            bench_details["_overall"] = {
                                "accuracy": corr / tot if tot else 0.0,
                                "total": tot,
                                "correct": corr,
                            }
                        elif source == "legalbench":
                            tot = sum(r["total"] for r in bench_details.values())
                            weighted = sum(r["score"] * r["total"] for r in bench_details.values())
                            bench_details["_overall"] = {
                                "score": weighted / tot if tot else 0.0,
                                "metric": "balanced_accuracy",
                                "total": tot,
                            }
                        details[source] = bench_details

                # Only add domain entry if there is at least one valid sample
                if domain_total > 0:
                    results[model_family][training_set][domain] = {
                        "overall_accuracy": domain_correct / domain_total,
                        "total": domain_total,
                        "correct": domain_correct,
                        "details": details,
                    }

    output = {
        "meta": {
            "input_file": str(args.input),
            "total_records": total_records,
        },
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
