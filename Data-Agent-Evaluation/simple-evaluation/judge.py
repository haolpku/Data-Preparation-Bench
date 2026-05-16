#!/usr/bin/env python3
"""
Unified judge script for Data-Agent-Evaluation.
Reads an inference-result JSONL and dispatches to benchmark-specific judge logic.
Supports resume, debug limit + shuffle + seed, async OpenAI with retry, and mock mode.
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Async LLM helper with retry
# ---------------------------------------------------------------------------


async def async_llm_chat(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    mock: bool = False,
) -> str:
    if mock:
        return "mocked response"
    for attempt in range(retries):
        try:
            kwargs: Dict[str, Any] = {"model": model, "messages": messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1 + attempt)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# FinCDM
# ---------------------------------------------------------------------------


def parse_answer_fincdm(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r'"?answer"?\s*[:：]\s*"?([ABCD])"?', raw_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", raw_text.upper())
    return match.group(1) if match else ""


def parse_referee_response(content: str) -> Dict[str, Any]:
    candidate = content.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    lowered = candidate.lower()
    guess = "true" in lowered and "false" not in lowered
    return {"is_correct": guess, "reason": candidate}


async def judge_fincdm(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str, mock: bool = False
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    standard = record["metadata"].get("standard_answer", "")
    parsed = parse_answer_fincdm(content)

    meta = {
        "domain": "business",
        "dataset_name": record["metadata"].get("dataset_name", "fincdm"),
        "parsed_answer": parsed,
        "standard_answer": standard,
    }

    if parsed and parsed == standard:
        return 1.0, f"Rule match: parsed answer {parsed} equals gold {standard}", meta, False, False

    user_msg = (
        f"Question ID: {record.get('id')}\n"
        f"Gold answer: {standard}\n"
        f"Answer parsed by code: {parsed}\n"
        "Original model response:\n<<<\n"
        f"{content}\n>>><<\n"
        "Decide whether the original response clearly selects the same option letter as the gold answer. "
        "If no valid letter can be identified, return false.\n"
        'Return JSON like {"is_correct": true, "reason": "brief justification"}. '
        "reason must briefly explain the decision."
    )

    try:
        raw = await async_llm_chat(
            client,
            model_name,
            [
                {"role": "system", "content": "You are a meticulous exam referee. Decide strictly whether the model's answer matches the gold answer."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
            mock=mock,
        )
    except Exception as exc:
        meta["error"] = str(exc)
        return 0.5, f"LLM judge failed: {exc}", meta, True, True

    meta["llm_judge_response"] = raw
    parsed_resp = parse_referee_response(raw)
    meta["llm_judge_parsed"] = parsed_resp
    is_correct = bool(parsed_resp.get("is_correct", False))
    reason = parsed_resp.get("reason", "")
    score = 1.0 if is_correct else 0.0
    return score, f"Referee: {reason}", meta, True, False


# ---------------------------------------------------------------------------
# XFinBench
# ---------------------------------------------------------------------------


def resp2ans_bool(resp: str) -> Any:
    if not isinstance(resp, str):
        return ""
    end_sent = "Therefore, my answer is"
    if end_sent not in resp:
        return ""
    resp = resp.split(end_sent)[-1]
    resp_lower = resp.lower()
    if "true" in resp_lower or "correct" in resp_lower or "1" in resp_lower:
        return 1
    if "false" in resp_lower or "incorrect" in resp_lower or "0" in resp_lower:
        return 0
    return ""


def parse_judge_response_xfin(text: str) -> Dict[str, Any]:
    json_candidate = text.strip()
    if not json_candidate:
        return {"should_be_marked_correct": False, "explanation": "empty judge response"}

    def _attempt_parse(candidate: str) -> Dict[str, Any]:
        data = json.loads(candidate)
        return {
            "should_be_marked_correct": bool(data.get("should_be_marked_correct")),
            "explanation": str(data.get("explanation", "")),
        }

    try:
        return _attempt_parse(json_candidate)
    except Exception:
        pass

    start = json_candidate.find("{")
    end = json_candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _attempt_parse(json_candidate[start : end + 1])
        except Exception:
            pass

    lowered = json_candidate.lower()
    flag = "true" in lowered and "false" not in lowered
    return {"should_be_marked_correct": flag, "explanation": "fallback judge heuristic"}


async def judge_xfinbench(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str, mock: bool = False
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    gt = record["metadata"].get("ground_truth")
    question = record["metadata"].get("question", "")
    parsed = resp2ans_bool(content)

    meta = {
        "domain": "business",
        "dataset_name": "xfinbench",
        "parsed_answer": parsed,
        "standard_answer": gt,
        "question": question,
    }

    if isinstance(parsed, int) and parsed == gt:
        return 1.0, f"Rule match: parsed {parsed} equals gold {gt}", meta, False, False

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth label (1=True, 0=False):\n"
        f"{gt}\n\n"
        "Candidate model full response:\n"
        f"{content}\n\n"
        "Return strict JSON like "
        '{"should_be_marked_correct": true, "explanation": "why"}'
    )

    try:
        raw = await async_llm_chat(
            client,
            model_name,
            [
                {
                    "role": "system",
                    "content": "You are a financial QA judge. Determine if the candidate model's final answer matches the provided ground-truth label (1=True, 0=False). Respond in strict JSON.",
                },
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            mock=mock,
        )
    except Exception as exc:
        meta["error"] = str(exc)
        return 0.5, f"LLM judge failed: {exc}", meta, True, True

    meta["llm_judge_response"] = raw
    parsed_resp = parse_judge_response_xfin(raw)
    meta["llm_judge_parsed"] = parsed_resp
    is_correct = bool(parsed_resp.get("should_be_marked_correct", False))
    reason = parsed_resp.get("explanation", "")
    score = 1.0 if is_correct else 0.0
    return score, f"Judge: {reason}", meta, True, False


# ---------------------------------------------------------------------------
# legalbench
# ---------------------------------------------------------------------------


def clean_r1_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r"^(answer|the answer is|result|prediction)[:\s]+", "", cleaned)
    if cleaned.startswith("yes"):
        return "Yes"
    if cleaned.startswith("no"):
        return "No"
    for label in ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]:
        if label in cleaned:
            return label
    if "ucc" in cleaned:
        return "UCC"
    if "common law" in cleaned:
        return "Common Law"
    return cleaned.capitalize()


async def judge_legalbench(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str, mock: bool = False
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    gold = str(record["metadata"].get("gold", ""))
    task_name = record["metadata"].get("task_name", "")
    prompt_text = record["metadata"].get("prompt_text", "")

    cleaned = clean_r1_output(content)
    meta = {
        "domain": "law",
        "dataset_name": task_name,
        "cleaned_pred": cleaned,
        "standard_answer": gold,
        "prompt_text": prompt_text,
        "was_calibrated": False,
    }

    final_pred = cleaned
    used_llm = False
    should_retry = False

    if final_pred == "" or final_pred != gold:
        used_llm = True
        judge_prompt = f"""You are an expert legal evaluator. Your task is to judge the correctness of a model's answer based on the given legal question. 

Task: {task_name}

Question:
{prompt_text}

Model's raw answer:
{content}

Please output ONLY the correct answer (e.g., "Yes", "No", "UCC", "Common Law", "generic", etc.) based on the question. Do not include any additional explanation.
"""
        try:
            calibrated = await async_llm_chat(
                client,
                model_name,
                [
                    {"role": "system", "content": "You are a legal expert judge. Output only the answer."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0,
                mock=mock,
            )
            calibrated_cleaned = clean_r1_output(calibrated)
            meta["calibrated_answer"] = calibrated_cleaned
            meta["llm_judge_response"] = calibrated
            meta["was_calibrated"] = True
            final_pred = calibrated_cleaned
        except Exception as exc:
            meta["calibration_error"] = str(exc)
            should_retry = True

    score = 1.0 if final_pred == gold else 0.0
    reason = f"Cleaned pred: {cleaned}"
    if meta["was_calibrated"]:
        reason += f" | Calibrated: {meta.get('calibrated_answer', 'N/A')}"
    reason += f" | Gold: {gold}"
    return score, reason, meta, used_llm, should_retry


# ---------------------------------------------------------------------------
# lex-glue
# ---------------------------------------------------------------------------

LEXGLUE_EXTRA_DIR = Path("lex-glue-extra")
_lexglue_label_cache: Dict[str, Optional[List[str]]] = {}


def get_lexglue_labels(task_name: str) -> Optional[List[str]]:
    if task_name in _lexglue_label_cache:
        return _lexglue_label_cache[task_name]
    try:
        from datasets import load_from_disk

        data_dir = str(LEXGLUE_EXTRA_DIR / f"{task_name}_test")
        ds = load_from_disk(data_dir)
        label_field = "labels" if task_name in ("eurlex", "unfair_tos") else "label"
        feature = ds.features[label_field]
        if hasattr(feature, "names"):
            label_names = list(feature.names)
        elif hasattr(feature, "feature") and hasattr(feature.feature, "names"):
            label_names = list(feature.feature.names)
        else:
            label_names = None
        _lexglue_label_cache[task_name] = label_names
        return label_names
    except Exception:
        _lexglue_label_cache[task_name] = None
        return None


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


def _parse_multilabel(text: str, label_names: List[str], max_labels: Optional[int] = None) -> Tuple[List[int], bool]:
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
        for name in label_names:
            if name.lower() in lower:
                indices.append(label_map[name.lower()])
    indices = sorted(set(indices))
    if max_labels is not None and len(indices) > max_labels:
        indices = indices[:max_labels]
    return indices, bool(indices) or ("[]" in text)


async def judge_lexglue(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    _ = client, model_name  # lex-glue does not use LLM judge
    content = record["messages"][-1]["content"]
    task_name = record["metadata"].get("task_name", "")
    gold = record["metadata"].get("gold")
    question_text = record["metadata"].get("question_text", "")

    label_names = get_lexglue_labels(task_name)
    meta = {
        "domain": "law",
        "dataset_name": task_name,
        "standard_answer": gold,
        "question_text": question_text,
        "label_names_available": label_names is not None,
    }

    if task_name == "ledgar":
        if label_names:
            pred_idx, ok = _parse_single_label(content, label_names)
            pred_label = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else ""
        else:
            pred_label = content.strip()
            ok = True
            gold_str = str(gold).strip().lower() if isinstance(gold, str) else ""
            if gold_str and gold_str in content.lower():
                pred_label = str(gold)
        meta["parsed_answer"] = pred_label
        meta["parse_ok"] = ok
        gold_norm = str(gold).strip().lower() if isinstance(gold, str) else ""
        pred_norm = pred_label.strip().lower()
        score = 1.0 if (gold_norm and pred_norm and gold_norm == pred_norm) else 0.0
        reason = f"Parsed: {pred_label} | Gold: {gold}"
        if not ok:
            reason += " (parse flagged as uncertain)"
        return score, reason, meta, False, False
    else:
        # multilabel: eurlex, unfair_tos
        if label_names:
            pred_indices, ok = _parse_multilabel(content, label_names)
            pred_labels = [label_names[i] for i in pred_indices if 0 <= i < len(label_names)]
        else:
            parsed = _extract_json_array(content)
            if not parsed:
                parts = re.split(r"[\n,;]", content)
                parsed = [p.strip() for p in parts if p.strip()]
            pred_labels = parsed
            ok = True
        meta["parsed_answer"] = pred_labels
        meta["parse_ok"] = ok

        if isinstance(gold, list):
            gold_set = {str(x).strip().lower() for x in gold if str(x).strip()}
        else:
            gold_set = {str(gold).strip().lower()} if str(gold).strip() else set()
        pred_set = {str(x).strip().lower() for x in pred_labels if str(x).strip()}

        score = 1.0 if gold_set == pred_set else 0.0
        reason = f"Parsed set: {pred_set} | Gold set: {gold_set}"
        if not ok:
            reason += " (parse flagged as uncertain)"
        return score, reason, meta, False, False


# ---------------------------------------------------------------------------
# MedCaseReasoning
# ---------------------------------------------------------------------------


def extract_tag(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


async def get_llm_decision(
    client: AsyncOpenAI, model_name: str, prompt: str, mock: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,
    }
    resp_text = await async_llm_chat(
        client, model_name, payload["messages"], temperature=0, max_tokens=5, mock=mock
    )
    decision = "y" in resp_text.lower()
    trace = {"prompt": prompt, "request": payload, "response_text": resp_text}
    return decision, trace


async def judge_medcasereasoning(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str, mock: bool = False
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    pred_diag = extract_tag(content, "answer")
    model_think = extract_tag(content, "think")
    true_diag = record["metadata"].get("final_diagnosis", "")
    gold_reasoning = record["metadata"].get("diagnostic_reasoning", "")

    meta = {
        "domain": "medicine",
        "dataset_name": "medcasereasoning",
        "standard_answer": true_diag,
        "predicted_diagnosis": pred_diag,
    }

    acc_prompt = (
        f"Is our predicted diagnosis correct (y/n)?\n"
        f"Predicted diagnosis: {pred_diag}, True diagnosis: {true_diag}\n"
        f"Answer [y/n]."
    )
    try:
        is_correct, acc_trace = await get_llm_decision(client, model_name, acc_prompt, mock=mock)
    except Exception as exc:
        meta["error"] = str(exc)
        return 0.5, f"LLM accuracy judge failed: {exc}", meta, True, True

    meta["accuracy_trace"] = acc_trace
    should_retry = False

    gold_points = re.split(r"\d+\.", gold_reasoning)
    gold_points = [p.strip() for p in gold_points if p.strip()]
    hits = 0
    recall_traces = []

    for point in gold_points:
        match_prompt = f"""Analyze if the model's reasoning covers the following clinician's point.
Clinician's point: {point}
Model's reasoning: {model_think}
Does the model mention or imply this specific point? Answer [y/n]."""
        try:
            is_match, match_trace = await get_llm_decision(client, model_name, match_prompt, mock=mock)
        except Exception as exc:
            is_match = False
            match_trace = {"error": str(exc)}
            should_retry = True
        recall_traces.append({
            "gold_point": point,
            "decision": is_match,
            "trace": match_trace,
        })
        if is_match:
            hits += 1

    recall = hits / len(gold_points) if gold_points else 0.0
    meta["reasoning_recall"] = recall
    meta["reasoning_traces"] = recall_traces
    meta["num_points"] = len(gold_points)
    meta["hits"] = hits

    score = 1.0 if is_correct else 0.0
    reason = (
        f"Accuracy: {'correct' if is_correct else 'incorrect'} | "
        f"Reasoning recall: {recall:.2%} ({hits}/{len(gold_points)})"
    )
    return score, reason, meta, True, should_retry


# ---------------------------------------------------------------------------
# MedmcQA
# ---------------------------------------------------------------------------


def extract_answer_medmcqa(text: str) -> str:
    m = re.search(r"^\s*Answer\s*:\s*([A-Z])\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()
    return ""


async def judge_medmcqa(
    record: Dict[str, Any], client: AsyncOpenAI, model_name: str, mock: bool = False
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    gold = record["metadata"].get("gold", "")
    pred = extract_answer_medmcqa(content)

    meta = {
        "domain": "medicine",
        "dataset_name": "medmcqa",
        "standard_answer": gold,
        "original_pred": pred,
    }

    final_pred = pred
    refer_used = False
    refer_output = None
    used_llm = False
    should_retry = False

    if pred and gold and pred.upper() != gold.upper():
        used_llm = True
        refer_prompt = (
            "You are an evaluator. Given the model's answer (which may include reasoning and a final answer) "
            "and the correct answer (a single letter), determine what answer the model actually chose. "
            "The model might express its answer in various ways; your task is to extract the final choice. "
            "Output only the letter (A, B, C, D, or E). Do not include any other text.\n\n"
            f"Model's answer:\n{content}\n\n"
            f"Correct answer: {gold}\n\n"
            "Output the letter that the model chose:"
        )
        try:
            refer_response = await async_llm_chat(
                client,
                model_name,
                [{"role": "user", "content": refer_prompt}],
                temperature=0,
                mock=mock,
            )
            m = re.search(r"\b([A-E])\b", refer_response.strip(), flags=re.IGNORECASE)
            if m:
                final_pred = m.group(1).upper()
                refer_used = True
                refer_output = refer_response
        except Exception as exc:
            refer_output = f"ERROR: {exc}"
            meta["refer_error"] = str(exc)
            should_retry = True

    meta["final_pred"] = final_pred
    meta["refer_used"] = refer_used
    meta["refer_output"] = refer_output

    score = 1.0 if (final_pred and gold and final_pred.upper() == gold.upper()) else 0.0
    reason = f"Final pred: {final_pred} | Gold: {gold}"
    if refer_used:
        reason += f" | Refer extracted: {refer_output}"
    return score, reason, meta, used_llm, should_retry


# ---------------------------------------------------------------------------
# MedRBench
# ---------------------------------------------------------------------------

MEDRBENCH_EXTRA_JSON = Path("MedRBench-extra.json")
_medrbench_gt_cache: Optional[Dict[str, str]] = None


def load_medrbench_gt(medrbench_gt_path: Optional[str] = None) -> Dict[str, str]:
    global _medrbench_gt_cache
    if _medrbench_gt_cache is not None:
        return _medrbench_gt_cache

    p = Path(medrbench_gt_path) if medrbench_gt_path else MEDRBENCH_EXTRA_JSON
    if not p.exists():
        raise FileNotFoundError(f"MedRBench ground truth file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt = {}
    for data_id, item in data.items():
        gt[data_id] = item.get("generate_case", {}).get("diagnosis_results", "")
    _medrbench_gt_cache = gt
    return gt


def extract_answer_content(text: str) -> str:
    if "### Answer" in text:
        return text.split("### Answer")[-1].replace("\n", "").replace(":", "")
    return text


async def judge_medrbench(
    record: Dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    medrbench_gt_path: Optional[str] = None,
    mock: bool = False,
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    content = record["messages"][-1]["content"]
    data_id = record["metadata"].get("data_id", "")
    gt_map = load_medrbench_gt(medrbench_gt_path)
    gt = gt_map.get(data_id, "")

    meta = {
        "domain": "medicine",
        "dataset_name": "medrbench",
        "data_id": data_id,
    }

    if not gt:
        meta["error"] = f"Ground truth not found for data_id={data_id}"
        return 0.5, f"Missing ground truth for {data_id}", meta, False, True

    meta["standard_answer"] = gt
    model_prediction = extract_answer_content(content)
    meta["parsed_answer"] = model_prediction

    template_path = Path("MedRBench_instructions/acc_diagnose.txt")
    if not template_path.exists():
        meta["error"] = f"Prompt template not found: {template_path}"
        return 0.5, f"Missing prompt template: {template_path}", meta, False, True

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    evaluation_prompt = template.format(pred_diagnose=model_prediction, gt_diagnose=gt)
    system_prompt = "You are a professional medical diagnosis evaluation system."

    try:
        evaluation_result = await async_llm_chat(
            client,
            model_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt},
            ],
            mock=mock,
        )
    except Exception as exc:
        meta["error"] = str(exc)
        return 0.5, f"LLM judge failed: {exc}", meta, True, True

    meta["llm_judge_response"] = evaluation_result
    is_correct = "correct" in evaluation_result.lower()
    score = 1.0 if is_correct else 0.0
    reason = f"LLM evaluation: {evaluation_result.strip()}"
    return score, reason, meta, True, False


# ---------------------------------------------------------------------------
# Dispatcher & main
# ---------------------------------------------------------------------------


async def judge_one(
    record: Dict[str, Any],
    client: AsyncOpenAI,
    judge_model: str,
    medrbench_gt_path: Optional[str],
    mock: bool = False,
) -> Tuple[float, str, Dict[str, Any], bool, bool]:
    source = record.get("metadata", {}).get("source", "")
    if source == "fincdm":
        return await judge_fincdm(record, client, judge_model, mock=mock)
    if source == "xfinbench":
        return await judge_xfinbench(record, client, judge_model, mock=mock)
    if source == "legalbench":
        return await judge_legalbench(record, client, judge_model, mock=mock)
    if source == "lexglue":
        return await judge_lexglue(record, client, judge_model)
    if source == "medcasereasoning":
        return await judge_medcasereasoning(record, client, judge_model, mock=mock)
    if source == "medmcqa":
        return await judge_medmcqa(record, client, judge_model, mock=mock)
    if source == "medrbench":
        return await judge_medrbench(record, client, judge_model, medrbench_gt_path, mock=mock)
    meta = {"source": source, "error": "Unknown source"}
    return 0.5, f"Unknown source: {source}", meta, False, True


async def write_consumer(
    queue: asyncio.Queue,
    writers: Dict[str, Any],
    mock: bool,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        result = item["result"]
        used_llm = item["used_llm"]
        await writers["judge"].write(json.dumps(result, ensure_ascii=False) + "\n")
        if result["judge_score"] == 0.0:
            await writers["failed"].write(json.dumps(result, ensure_ascii=False) + "\n")
        elif result["judge_score"] == 0.5:
            await writers["error"].write(json.dumps(result, ensure_ascii=False) + "\n")
        if mock:
            if used_llm:
                await writers["pending"].write(json.dumps(result["original_data"], ensure_ascii=False) + "\n")
            else:
                await writers["no_llm"].write(json.dumps(result, ensure_ascii=False) + "\n")
        queue.task_done()


def compute_stable_id(record: Dict[str, Any]) -> str:
    """基于 record 内容生成与遍历顺序无关的稳定 ID。"""
    meta = record.get("metadata", {})
    key_data = {
        "domain": meta.get("domain", ""),
        "source": meta.get("source", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "task_name": meta.get("task_name", ""),
        "data_id": meta.get("data_id", ""),
        "original_id": meta.get("original_id", ""),
        "qa_id": meta.get("qa_id", ""),
        "qid": meta.get("qid", ""),
        "pmcid": meta.get("pmcid", ""),
        "messages": record.get("messages", []),
    }
    content = json.dumps(key_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def compute_judge_id(record: Dict[str, Any]) -> str:
    meta = record.get("metadata", {})
    stable_id = meta.get("stable_id") or compute_stable_id(record)
    return "-".join([
        str(meta.get("model_family", "unknown")),
        str(meta.get("domain", "unknown")),
        str(meta.get("training_set", "unknown")),
        str(meta.get("source", "unknown")),
        stable_id,
    ])


async def process_one(
    args: argparse.Namespace,
    record: Dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    medrbench_gt_path: Optional[str],
    queue: asyncio.Queue,
    mock: bool = False,
) -> None:
    async with semaphore:
        score, reason, meta, used_llm, should_retry = await judge_one(
            record, client, args.judge_model, medrbench_gt_path, mock=mock
        )

        source = record.get("metadata", {}).get("source", "")
        messages = record.get("messages", [])
        if source == "fincdm" and len(messages) >= 2:
            original_question = messages[1].get("content", "")
        elif source == "xfinbench":
            original_question = record["metadata"].get("question", "")
        elif source == "legalbench":
            original_question = record["metadata"].get("prompt_text", "")
        elif source == "lexglue":
            original_question = record["metadata"].get("question_text", "")
        elif source == "medcasereasoning" and messages:
            original_question = messages[0].get("content", "")
        elif source == "medmcqa":
            original_question = record["metadata"].get("question", "")
        elif source == "medrbench":
            original_question = record["metadata"].get("patient_case", "")
        else:
            original_question = ""

        standard_answer = meta.get("standard_answer", "")

        meta["used_llm"] = used_llm
        result = {
            "_judge_id": compute_judge_id(record),
            "id": record.get("id", ""),
            "original_data": record,
            "judge_score": score,
            "judge_reason": reason,
            "standard_answer": standard_answer,
            "original_question": original_question,
            "should_retry": should_retry,
            "judge_meta": meta,
        }

    await queue.put({"result": result, "used_llm": used_llm})


async def main() -> None:
    parser = argparse.ArgumentParser(description="Unified judge script for Data-Agent-Evaluation")
    parser.add_argument("--input-jsonl", required=True, help="Input inference results JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--judge-url", required=True, help="Judge model base URL")
    parser.add_argument("--judge-api-key", required=True, help="Judge model API key")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model name")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--debug-limit", type=int, default=None, help="Only judge first N lines")
    parser.add_argument("--debug-seed", type=int, default=None, help="Seed for shuffling debug samples")
    parser.add_argument("--mock-openai", action="store_true", help="Mock all OpenAI judge calls")
    parser.add_argument(
        "--medrbench-gt-json",
        default=None,
        help="Path to MedRBench ground truth JSON (diagnosis_957_cases_with_rare_disease_491.json)",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.judge_url, api_key=args.judge_api_key)

    # Resume support: read existing results
    done_ids: set = set()
    done_records: Dict[str, Dict[str, Any]] = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_results_path = out_dir / "judge_results.jsonl"
    failed_results_path = out_dir / "failed_results.jsonl"
    error_path = out_dir / "error.jsonl"
    extra_no_llm_path = out_dir / "no_llm_needed.jsonl"
    extra_pending_path = out_dir / "pending_llm_judge.jsonl"

    if judge_results_path.exists():
        with open(judge_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rec_id = obj.get("_judge_id")
                if not rec_id:
                    rec_id = compute_judge_id(obj.get("original_data", {})) or obj.get("id", "")
                if not obj.get("should_retry", True):
                    done_ids.add(rec_id)
                    # Backward compatibility: 旧结果可能没有 stable_id，这里用
                    # original_data 重新计算新的 _judge_id 并加入 done_ids，
                    # 这样 re-merge 后换了一批 enumerate id 也能正常 resume。
                    original_data = obj.get("original_data", {})
                    if original_data:
                        done_ids.add(compute_judge_id(original_data))
                    # Backfill used_llm for records written by older script versions
                    meta = obj.setdefault("judge_meta", {})
                    if "used_llm" not in meta:
                        source = obj.get("original_data", {}).get("metadata", {}).get("source", "")
                        meta["used_llm"] = source != "lexglue"
                    done_records[rec_id] = obj
        print(f"Resume: skipping {len(done_ids)} already judged records", file=sys.stderr)

    # Write resumed successful records back synchronously (always overwrite to discard stale retries)
    with open(judge_results_path, "w", encoding="utf-8") as f_judge, \
         open(failed_results_path, "w", encoding="utf-8") as f_failed, \
         open(error_path, "w", encoding="utf-8") as f_error:
        extra_files_sync: List[Any] = []
        if args.mock_openai:
            extra_files_sync = [
                open(extra_no_llm_path, "w", encoding="utf-8"),
                open(extra_pending_path, "w", encoding="utf-8"),
            ]
        for obj in done_records.values():
            f_judge.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if obj["judge_score"] == 0.0:
                f_failed.write(json.dumps(obj, ensure_ascii=False) + "\n")
            elif obj["judge_score"] == 0.5:
                f_error.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if args.mock_openai:
                used_llm = obj.get("judge_meta", {}).get("used_llm", False)
                if used_llm:
                    extra_files_sync[1].write(json.dumps(obj.get("original_data", {}), ensure_ascii=False) + "\n")
                else:
                    extra_files_sync[0].write(json.dumps(obj, ensure_ascii=False) + "\n")
        for f in extra_files_sync:
            f.close()
    del done_records

    records: List[Dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.debug_seed is not None:
        rng = random.Random(args.debug_seed)
        rng.shuffle(records)
    if args.debug_limit is not None:
        records = records[: args.debug_limit]

    # Inject composite judge id and resume filtering
    for rec in records:
        rec["_judge_id"] = compute_judge_id(rec)

    records = [r for r in records if r.get("_judge_id") not in done_ids]

    # Early dependency checks
    missing_deps: List[str] = []
    lex_tasks = ["eurlex", "ledgar", "unfair_tos"]
    for task in lex_tasks:
        if not (LEXGLUE_EXTRA_DIR / f"{task}_test").exists():
            missing_deps.append(f"{LEXGLUE_EXTRA_DIR}/{task}_test")
    if not MEDRBENCH_EXTRA_JSON.exists():
        missing_deps.append(str(MEDRBENCH_EXTRA_JSON))
    if missing_deps:
        print("Missing required dependencies:", file=sys.stderr)
        for dep in missing_deps:
            print(f"  - {dep}", file=sys.stderr)
        sys.exit(1)

    print(f"Judging {len(records)} records with concurrency={args.concurrency} ...", file=sys.stderr)

    writers: Dict[str, Any] = {
        "judge": await aiofiles.open(judge_results_path, "a", encoding="utf-8"),
        "failed": await aiofiles.open(failed_results_path, "a", encoding="utf-8"),
        "error": await aiofiles.open(error_path, "a", encoding="utf-8"),
    }
    if args.mock_openai:
        writers["no_llm"] = await aiofiles.open(extra_no_llm_path, "a", encoding="utf-8")
        writers["pending"] = await aiofiles.open(extra_pending_path, "a", encoding="utf-8")

    queue: asyncio.Queue = asyncio.Queue()
    consumer_task = asyncio.create_task(write_consumer(queue, writers, args.mock_openai))

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_one(args, rec, client, semaphore, args.medrbench_gt_json, queue, args.mock_openai)
        for rec in records
    ]

    try:
        try:
            from tqdm.asyncio import tqdm_asyncio
            await tqdm_asyncio.gather(*tasks, desc="Judging")
        except ImportError:
            await asyncio.gather(*tasks)
    finally:
        await queue.put(None)
        await consumer_task
        for w in writers.values():
            await w.close()

    print(f"Done. Judged {len(tasks)} new records. Output dir: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
