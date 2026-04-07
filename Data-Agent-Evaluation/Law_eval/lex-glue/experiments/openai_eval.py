#!/usr/bin/env python
# coding=utf-8

import json
import os
import re
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from sklearn.metrics import f1_score
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "OpenAI client not available. Install the 'openai' package before running."
    ) from exc

try:
    from datasets import load_from_disk
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "datasets package not available. Install 'datasets' before running."
    ) from exc


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ecthr_a": {
        "data_dir": "data_filtered_48k/ecthr_a_test",
        "type": "multilabel_ecthr",
        "text_field": "text",
    },
    "ecthr_b": {
        "data_dir": "data_filtered_48k/ecthr_b_test",
        "type": "multilabel_ecthr",
        "text_field": "text",
    },
    "eurlex": {
        "data_dir": "data_filtered_48k/eurlex_test",
        "type": "multilabel",
        "text_field": "text",
    },
    "unfair_tos": {
        "data_dir": "data_filtered_48k/unfair_tos_test",
        "type": "multilabel",
        "text_field": "text",
    },
    "ledgar": {
        "data_dir": "data_filtered_48k/ledgar_test",
        "type": "singlelabel",
        "text_field": "text",
    },
    "scotus": {
        "data_dir": "data_filtered_48k/scotus_test",
        "type": "singlelabel",
        "text_field": "text",
    },
    "case_hold": {
        "data_dir": "data_filtered_48k/case_hold_test",
        "type": "case_hold",
        "context_field": "context",
        "endings_field": "endings",
    },
}


LABEL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "ecthr_a": {
        "2": "Right to life",
        "3": "Prohibition of torture and inhuman or degrading treatment",
        "5": "Right to liberty and security",
        "6": "Right to a fair trial",
        "8": "Right to respect for private and family life",
        "9": "Freedom of thought, conscience and religion",
        "10": "Freedom of expression",
        "11": "Freedom of assembly and association",
        "14": "Prohibition of discrimination",
        "P1-1": "Protection of property (Protocol 1, Article 1)",
    },
    "ecthr_b": {
        "2": "Right to life",
        "3": "Prohibition of torture and inhuman or degrading treatment",
        "5": "Right to liberty and security",
        "6": "Right to a fair trial",
        "8": "Right to respect for private and family life",
        "9": "Freedom of thought, conscience and religion",
        "10": "Freedom of expression",
        "11": "Freedom of assembly and association",
        "14": "Prohibition of discrimination",
        "P1-1": "Protection of property (Protocol 1, Article 1)",
    },
    "unfair_tos": {
        "Limitation of liability": "Limits or excludes the platform's liability",
        "Unilateral termination": "Allows the platform to terminate the contract unilaterally",
        "Unilateral change": "Allows the platform to change terms unilaterally",
        "Content removal": "Allows the platform to remove user content",
        "Contract by using": "Deems agreement accepted by mere use of the service",
        "Choice of law": "Specifies governing law",
        "Jurisdiction": "Specifies the court/forum for disputes",
        "Arbitration": "Requires arbitration instead of court",
    },
}

TASK_INSTRUCTIONS: Dict[str, str] = {
    "ecthr_a": "Predict which ECHR articles were violated, based on the facts.",
    "ecthr_b": "Predict which ECHR articles were allegedly violated (considered by the court).",
    "scotus": "Predict the issue area ID for the US Supreme Court case.",
    "eurlex": "Predict EuroVoc concept IDs for the EU legal document.",
    "ledgar": "Predict the single best contract provision category.",
    "unfair_tos": "Identify all unfair terms categories present in the sentence(s).",
    "case_hold": "Select the correct holding statement for the given case context.",
}

# Request timeout (seconds)
REQUEST_TIMEOUT = 30

# Concurrency controls for OpenAI calls
MAX_WORKERS = 8


def _load_dataset(task_name: str):
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    data_dir = TASK_CONFIGS[task_name]["data_dir"]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return load_from_disk(data_dir)


def _maybe_truncate(ds, is_full_eval: bool):
    if is_full_eval:
        return ds
    n = min(3, len(ds))
    if n == 0:
        return ds
    # Ensure at least one positive-label example when possible
    label_field = "labels" if "labels" in ds.features else "label"
    if label_field == "labels":
        non_empty = [i for i in range(len(ds)) if len(ds[i]["labels"]) > 0]
        if non_empty:
            keep = [non_empty[0]]
            for i in range(len(ds)):
                if i not in keep:
                    keep.append(i)
                if len(keep) >= n:
                    break
            return ds.select(keep[:n])
    return ds.select(range(n))


def _get_label_names(ds, label_field: str) -> List[str]:
    feature = ds.features[label_field]
    if hasattr(feature, "names"):
        return list(feature.names)
    if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
        return list(feature.feature.names)
    raise ValueError(f"Unable to read label names from field: {label_field}")


def _format_label_list(label_names: List[str], label_descriptions: Dict[str, str] = None) -> str:
    rows = []
    for name in label_names:
        if label_descriptions and name in label_descriptions:
            rows.append(f"- {name}: {label_descriptions[name]}")
        else:
            rows.append(f"- {name}")
    return "\n".join(rows)


def _build_prompt(task_name: str, example: Dict[str, Any], label_names: List[str]) -> List[Dict[str, str]]:
    task_instruction = TASK_INSTRUCTIONS.get(task_name, "")
    label_descriptions = LABEL_DESCRIPTIONS.get(task_name)

    task_type = TASK_CONFIGS[task_name]["type"]
    if task_type == "case_hold":
        context = example[TASK_CONFIGS[task_name]["context_field"]]
        endings = example[TASK_CONFIGS[task_name]["endings_field"]]
        options = [f"{chr(65+i)}) {opt}" for i, opt in enumerate(endings)]
        user = (
            "Select the correct option to complete the holding.\n\n"
            f"Context:\n{context}\n\n"
            "Options:\n" + "\n".join(options) + "\n\n"
            "Return only the option letter (A, B, C, D, or E)."
        )
        system = "You are a legal NLP model that answers multiple-choice questions."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    text_field = TASK_CONFIGS[task_name].get("text_field", "text")
    text = example[text_field]
    if isinstance(text, list):
        text = "\n\n".join(text)

    if task_type == "singlelabel":
        system = "You are a legal NLP classifier. Choose exactly one label from the list."
        user = (
            f"Task: {task_instruction}\n\n"
            f"Text:\n{text}\n\n"
            f"Labels:\n{_format_label_list(label_names, label_descriptions)}\n\n"
            "Return only the label name exactly as shown (no extra text, no code fences)."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if task_type in {"multilabel", "multilabel_ecthr"}:
        system = "You are a legal NLP classifier. Select only the most applicable labels."
        user = (
            f"Task: {task_instruction}\n\n"
            f"Text:\n{text}\n\n"
            f"Labels:\n{_format_label_list(label_names, label_descriptions)}\n\n"
            "Return a JSON array of label names exactly as shown (no descriptions). "
            "If none apply, return []. "
            "Example: [\"6\", \"10\"]"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    raise ValueError(f"Unsupported task type: {task_type}")


def _format_messages(messages: List[Dict[str, str]]) -> str:
    return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])


def _format_question(task_name: str, example: Dict[str, Any]) -> str:
    task_type = TASK_CONFIGS[task_name]["type"]
    if task_type == "case_hold":
        context = example[TASK_CONFIGS[task_name]["context_field"]]
        endings = example[TASK_CONFIGS[task_name]["endings_field"]]
        options = [f"{chr(65+i)}) {opt}" for i, opt in enumerate(endings)]
        return f"Context:\n{context}\n\nOptions:\n" + "\n".join(options)
    text_field = TASK_CONFIGS[task_name].get("text_field", "text")
    text = example[text_field]
    if isinstance(text, list):
        return "\n\n".join(text)
    return text


def _extract_json_array(text: str) -> List[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _parse_single_label(text: str, label_names: List[str]) -> Tuple[int, bool]:
    raw = text.strip()
    lower = raw.lower()
    for idx, name in enumerate(label_names):
        if lower == name.lower():
            return idx, True
    for idx, name in enumerate(label_names):
        if name.lower() in lower:
            return idx, True
    if ":" in raw:
        left = raw.split(":", 1)[0].strip()
        for idx, name in enumerate(label_names):
            if left.lower() == name.lower():
                return idx, True
    digit = re.search(r"\b(\d+)\b", raw)
    if digit and digit.group(1) in label_names:
        return label_names.index(digit.group(1)), True
    return 0, False


def _map_label_item(item: str, label_names: List[str]) -> str:
    if not item:
        return ""
    raw = item.strip()
    if not raw:
        return ""
    for name in label_names:
        if raw.lower() == name.lower():
            return name
    if ":" in raw:
        left = raw.split(":", 1)[0].strip()
        for name in label_names:
            if left.lower() == name.lower():
                return name
    lower = raw.lower()
    for name in label_names:
        if name.lower() in lower:
            return name
    return ""


def _parse_multilabel(text: str, label_names: List[str], max_labels: int = None) -> Tuple[List[int], bool]:
    lower = text.lower()
    if "none" in lower or "no label" in lower:
        return [], True
    parsed = _extract_json_array(text)
    if not parsed:
        parts = re.split(r"[\n,;]", text)
        parsed = [p.strip() for p in parts if p.strip()]
    label_map = {name.lower(): idx for idx, name in enumerate(label_names)}
    indices = []
    for item in parsed:
        mapped = _map_label_item(item, label_names)
        if mapped:
            indices.append(label_map[mapped.lower()])
    if not indices:
        # fallback: scan the whole response for any label names
        for name in label_names:
            if name.lower() in lower:
                indices.append(label_map[name.lower()])
    indices = sorted(set(indices))
    if max_labels is not None and len(indices) > max_labels:
        indices = indices[:max_labels]
    return indices, bool(indices) or ("[]" in text)


def _parse_case_hold(text: str) -> Tuple[int, bool]:
    upper = text.upper()
    letter = re.search(r"\b([A-E])\b", upper)
    if letter:
        return ord(letter.group(1)) - 65, True
    digit = re.search(r"\b([1-5])\b", upper)
    if digit:
        return int(digit.group(1)) - 1, True
    return 0, False


def _call_openai(client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens: int = 64) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    return ""


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _labels_from_indices(indices: List[int], label_names: List[str]) -> List[str]:
    return [label_names[i] for i in indices if 0 <= i < len(label_names)]


def run_task(
    task_name: str,
    api_key: str,
    base_url: str,
    model_name: str,
    is_full_eval: bool,
):
    ds = _load_dataset(task_name)
    ds = _maybe_truncate(ds, is_full_eval)

    task_type = TASK_CONFIGS[task_name]["type"]
    label_field = "labels" if task_type in {"multilabel", "multilabel_ecthr"} else "label"
    label_names = _get_label_names(ds, label_field)

    y_true = []
    y_pred = []
    parse_failures = 0
    _ensure_dir("temp_result")
    _ensure_dir("result")
    temp_path = os.path.join("temp_result", f"{task_name}_{model_name}.jsonl")
    result_path = os.path.join("result", f"{task_name}_{model_name}.json")
    # Overwrite previous temp_result to keep a clean run, but write incrementally.
    with open(temp_path, "w", encoding="utf-8") as _:
        pass
    write_lock = Lock()
    parsed_rows = 0

    def _process_example(idx: int, example: Dict[str, Any]):
        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = _build_prompt(task_name, example, label_names)
        prompt_text = _format_messages(messages)
        question_text = _format_question(task_name, example)
        response_text = _call_openai(client, model_name, messages)

        if task_type == "case_hold":
            pred, ok = _parse_case_hold(response_text)
            gold = int(example["label"])
            gold_label = chr(65 + gold)
            pred_label = chr(65 + pred) if 0 <= pred <= 4 else ""
            return (
                idx,
                gold,
                pred,
                ok,
                response_text,
                "case_hold",
                question_text,
                prompt_text,
                gold_label,
                pred_label,
            )

        if task_type == "singlelabel":
            pred, ok = _parse_single_label(response_text, label_names)
            gold = int(example["label"])
            gold_label = label_names[gold] if 0 <= gold < len(label_names) else ""
            pred_label = label_names[pred] if 0 <= pred < len(label_names) else ""
            return (
                idx,
                gold,
                pred,
                ok,
                response_text,
                "singlelabel",
                question_text,
                prompt_text,
                gold_label,
                pred_label,
            )

        if task_type in {"multilabel", "multilabel_ecthr"}:
            pred_labels, ok = _parse_multilabel(response_text, label_names, max_labels=None)
            gold_labels = list(example["labels"])
            gold_label_names = _labels_from_indices(gold_labels, label_names)
            pred_label_names = _labels_from_indices(pred_labels, label_names)
            return (
                idx,
                gold_labels,
                pred_labels,
                ok,
                response_text,
                "multilabel",
                question_text,
                prompt_text,
                gold_label_names,
                pred_label_names,
            )

        return idx, None, None, False, response_text, "unknown", question_text, prompt_text, "", ""

    results_by_idx = [None] * len(ds)
    y_true = []
    y_pred = []
    parse_failures = 0

    def _progress_iter(total: int):
        if tqdm is not None:
            return tqdm(total=total, desc=f"{task_name} eval", ncols=100)

        class _SimpleProgress:
            def __init__(self, total_count: int):
                self.total = total_count
                self.count = 0

            def update(self, n: int = 1):
                self.count += n
                print(f"{task_name} eval: {self.count}/{self.total}", end="\r", flush=True)

            def close(self):
                print(f"{task_name} eval: {self.count}/{self.total}")

        return _SimpleProgress(total)

    def _write_row(row: Dict[str, Any]):
        with write_lock:
            with open(temp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    if MAX_WORKERS > 1 and len(ds) > 1:
        progress = _progress_iter(len(ds))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_process_example, idx, example) for idx, example in enumerate(ds)]
            for fut in as_completed(futures):
                (
                    idx,
                    gold,
                    pred,
                    ok,
                    response_text,
                    kind,
                    question_text,
                    prompt_text,
                    gold_label,
                    pred_label,
                ) = fut.result()
                if not ok:
                    parse_failures += 1
                if kind in {"singlelabel", "case_hold"}:
                    y_true.append(gold)
                    y_pred.append(pred)
                elif kind == "multilabel":
                    y_true.append(gold)
                    y_pred.append(pred)
                row = {
                    "id": idx,
                    "question": question_text,
                    "prompt": prompt_text,
                    "raw_response": response_text,
                    "gold": gold_label,
                    "extracted_answer": pred_label,
                }
                results_by_idx[idx] = row
                _write_row(row)
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()
    else:
        progress = _progress_iter(len(ds))
        for idx, example in enumerate(ds):
            (
                idx,
                gold,
                pred,
                ok,
                response_text,
                kind,
                question_text,
                prompt_text,
                gold_label,
                pred_label,
            ) = _process_example(idx, example)
            if not ok:
                parse_failures += 1
            if kind in {"singlelabel", "case_hold"}:
                y_true.append(gold)
                y_pred.append(pred)
            elif kind == "multilabel":
                y_true.append(gold)
                y_pred.append(pred)
            row = {
                "id": idx,
                "question": question_text,
                "prompt": prompt_text,
                "raw_response": response_text,
                "gold": gold_label,
                "extracted_answer": pred_label,
            }
            results_by_idx[idx] = row
            _write_row(row)
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()

    results = [row for row in results_by_idx if row is not None]

    metrics = {}
    if task_type in {"singlelabel", "case_hold"}:
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)
        metrics = {"macro-f1": macro_f1, "micro-f1": micro_f1}
    elif task_type == "multilabel":
        num_labels = len(label_names)
        y_true_bin = np.zeros((len(y_true), num_labels), dtype=np.int32)
        y_pred_bin = np.zeros((len(y_true), num_labels), dtype=np.int32)
        for i, labels in enumerate(y_true):
            y_true_bin[i, labels] = 1
        for i, labels in enumerate(y_pred):
            y_pred_bin[i, labels] = 1
        macro_f1 = f1_score(y_true=y_true_bin, y_pred=y_pred_bin, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true=y_true_bin, y_pred=y_pred_bin, average="micro", zero_division=0)
        metrics = {"macro-f1": macro_f1, "micro-f1": micro_f1}
    elif task_type == "multilabel_ecthr":
        num_labels = len(label_names)
        y_true_bin = np.zeros((len(y_true), num_labels + 1), dtype=np.int32)
        y_pred_bin = np.zeros((len(y_true), num_labels + 1), dtype=np.int32)
        for i, labels in enumerate(y_true):
            if labels:
                y_true_bin[i, labels] = 1
            y_true_bin[i, -1] = 1 if len(labels) == 0 else 0
        for i, labels in enumerate(y_pred):
            if labels:
                y_pred_bin[i, labels] = 1
            y_pred_bin[i, -1] = 1 if len(labels) == 0 else 0
        macro_f1 = f1_score(y_true=y_true_bin, y_pred=y_pred_bin, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true=y_true_bin, y_pred=y_pred_bin, average="micro", zero_division=0)
        metrics = {"macro-f1": macro_f1, "micro-f1": micro_f1}

    metrics["parse_failures"] = parse_failures
    metrics["num_examples"] = len(y_true)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics
