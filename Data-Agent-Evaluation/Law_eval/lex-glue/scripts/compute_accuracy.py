#!/usr/bin/env python
# coding=utf-8
"""
Compute accuracy from temp_result jsonl files.

Rules:
  - singlelabel: exact string match (case-insensitive, stripped)
  - multilabel: exact set match (case-insensitive)
  - empty/invalid extracted_answer -> incorrect
"""

import argparse
import json
import os
from typing import Any, Dict, List


def _normalize_label(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _normalize_list(xs: Any) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, list):
        return sorted({_normalize_label(x) for x in xs if _normalize_label(x)})
    # If a string accidentally stored, split on common separators
    if isinstance(xs, str):
        parts = [p.strip() for p in xs.replace(";", ",").split(",")]
        return sorted({_normalize_label(p) for p in parts if _normalize_label(p)})
    return [_normalize_label(xs)]


def _is_multilabel(value: Any) -> bool:
    return isinstance(value, list)


def _is_parse_failure(gold: Any, pred: Any) -> bool:
    # Multilabel expects list
    if isinstance(gold, list):
        if pred is None:
            return True
        if isinstance(pred, list):
            return False
        # wrong type -> parse failure
        return True
    # Singlelabel expects non-empty string
    if pred is None:
        return True
    if isinstance(pred, list):
        return True
    if isinstance(pred, str) and pred.strip() == "":
        return True
    return False


def _is_empty_pred(pred):
    if pred is None:
        return True
    if isinstance(pred, list):
        return len(pred) == 0
    if isinstance(pred, str):
        return pred.strip() == ""
    return False


def _extract_and_filter(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".jsonl"):
            continue
        src_path = os.path.join(input_dir, name)
        dst_path = os.path.join(output_dir, name)
        kept = 0
        removed = 0
        with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out = {
                    "gold": row.get("gold"),
                    "extracted_answer": row.get("extracted_answer"),
                }
                if _is_empty_pred(out.get("extracted_answer")):
                    removed += 1
                    continue
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{name}: extracted kept={kept}, removed_empty={removed}")


def compute_file_accuracy(path: str) -> Dict[str, Any]:
    total = 0
    correct = 0
    skipped = 0
    total_gold_labels = 0
    matched_gold_labels = 0
    sample_hit_total = 0
    sample_hit_correct = 0
    precision_pred_total = 0
    precision_correct = 0
    sample_f1_sum = 0.0
    sample_f1_count = 0
    topk_recall = {1: 0, 3: 0, 5: 0}
    topk_total = 0
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        gold = row.get("gold")
        pred = row.get("extracted_answer")
        if _is_parse_failure(gold, pred):
            skipped += 1
            continue

        if _is_multilabel(gold):
            gold_set = _normalize_list(gold)
            pred_set = _normalize_list(pred)
            total_gold_labels += len(gold_set)
            if gold_set and pred_set:
                matched_gold_labels += len(set(gold_set) & set(pred_set))
            # sample-level hit rate
            sample_hit_total += 1
            if set(gold_set) & set(pred_set):
                sample_hit_correct += 1
            # precision (label-level)
            precision_pred_total += len(pred_set)
            if pred_set and gold_set:
                precision_correct += len(set(gold_set) & set(pred_set))
            # sample-level F1
            if gold_set or pred_set:
                inter = len(set(gold_set) & set(pred_set))
                denom = len(gold_set) + len(pred_set)
                f1 = (2 * inter / denom) if denom > 0 else 0.0
                sample_f1_sum += f1
                sample_f1_count += 1
            # top-k recall (based on current pred order after normalization)
            topk_total += 1
            for k in topk_recall.keys():
                topk = set(pred_set[:k])
                if gold_set and topk and (set(gold_set) & topk):
                    topk_recall[k] += 1
        else:
            total += 1
            gold_norm = _normalize_label(gold)
            pred_norm = _normalize_label(pred)
            if gold_norm and pred_norm and gold_norm == pred_norm:
                correct += 1
            elif gold_norm == "" and pred_norm == "":
                correct += 1

    acc = correct / total if total else 0.0
    multilabel_acc = (
        matched_gold_labels / total_gold_labels if total_gold_labels else 0.0
    )
    sample_hit_rate = sample_hit_correct / sample_hit_total if sample_hit_total else 0.0
    precision = precision_correct / precision_pred_total if precision_pred_total else 0.0
    sample_f1 = sample_f1_sum / sample_f1_count if sample_f1_count else 0.0
    topk_recall_rate = {
        f"top{k}": (topk_recall[k] / topk_total if topk_total else 0.0)
        for k in sorted(topk_recall.keys())
    }
    return {
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "accuracy": acc,
        "multilabel_total_gold": total_gold_labels,
        "multilabel_matched": matched_gold_labels,
        "multilabel_accuracy": multilabel_acc,
        "multilabel_sample_hit_rate": sample_hit_rate,
        "multilabel_precision": precision,
        "multilabel_sample_f1": sample_f1,
        "multilabel_topk_recall": topk_recall_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="temp_result", help="Directory containing *.jsonl files")
    parser.add_argument("--output_dir", default="temp_result_min", help="Directory for extracted min files")
    parser.add_argument("--output", default="score_result", help="Optional output json file path")
    args = parser.parse_args()

    results = {}
    if not os.path.isdir(args.input_dir):
        raise RuntimeError(f"Input directory not found: {args.input_dir}")

    # Extract gold/pred and drop empty preds first, then compute scores on output_dir
    _extract_and_filter(args.input_dir, args.output_dir)

    for name in sorted(os.listdir(args.output_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(args.output_dir, name)
        results[name] = compute_file_accuracy(path)
        r = results[name]

        # Unify task score naming
        if name.startswith("eurlex_") or name.startswith("unfair_tos_"):
            score = r["multilabel_accuracy"]
        elif name.startswith("ledgar_"):
            score = r["accuracy"]
        else:
            score = r["accuracy"]
        r["score"] = score

        msg = (
            f"{name}: score={score:.4f} "
            f"(skipped={r['skipped']})"
        )
        print(msg)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
