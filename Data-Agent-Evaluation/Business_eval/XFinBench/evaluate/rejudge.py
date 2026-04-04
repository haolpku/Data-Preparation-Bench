import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI


# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.environ.get("PROJECT_PATH", os.path.dirname(SCRIPT_DIR))
TEMP_RESULT_DIR = os.path.join(PROJECT_PATH, "temp_result")
RESULT_DIR = os.path.join(PROJECT_PATH, "result")
MAX_TOKENS = 512


JUDGE_SYSTEM_PROMPT = (
    "You are a financial QA judge. Determine if the candidate model's final answer "
    "matches the provided ground-truth label (1=True, 0=False). Respond in strict JSON."
)


def ensure_directories() -> None:
    os.makedirs(TEMP_RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def iter_temp_result_files() -> Iterable[str]:
    if not os.path.isdir(TEMP_RESULT_DIR):
        return []
    entries = sorted(
        [
            os.path.join(TEMP_RESULT_DIR, name)
            for name in os.listdir(TEMP_RESULT_DIR)
            if name.endswith(".json")
        ]
    )
    return entries


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json_file(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def display_progress(prefix: str, current: int, total: int) -> None:
    bar_len = 30
    filled = int(bar_len * current / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = (current / total * 100) if total else 0
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent:.1f}%)")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def build_reference_client() -> OpenAI:
    return OpenAI(
        api_key=REFER_API_KEY,
        base_url=REFER_API_BASE_URL,
    )


def parse_judge_response(text: str) -> Tuple[bool, str, str]:
    json_candidate = text.strip()
    if not json_candidate:
        return False, "empty judge response", text

    def _attempt_parse(candidate: str) -> Tuple[bool, str]:
        data = json.loads(candidate)
        flag = bool(data.get("should_be_marked_correct"))
        reason = str(data.get("explanation", ""))
        return flag, reason

    try:
        flag, reason = _attempt_parse(json_candidate)
        return flag, reason, text
    except Exception:
        pass

    start = json_candidate.find("{")
    end = json_candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            flag, reason = _attempt_parse(json_candidate[start : end + 1])
            return flag, reason, text
        except Exception:
            pass

    lowered = json_candidate.lower()
    flag = "true" in lowered and "false" not in lowered
    return flag, "fallback judge heuristic", text


def judge_with_reference_model(
    client: OpenAI, question: str, ground_truth: Any, llm_response: str
) -> Tuple[bool, str, str]:
    if client is None:
        raise ValueError("Reference model client is not initialized.")

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth label (1=True, 0=False):\n"
        f"{ground_truth}\n\n"
        "Candidate model full response:\n"
        f"{llm_response}\n\n"
        "Return strict JSON like "
        '{"should_be_marked_correct": true, "explanation": "why"}'
    )

    completion = client.chat.completions.create(
        model=REFER_MODEL_NAME,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
    )
    reply = completion.choices[0].message.content or ""
    return parse_judge_response(reply)


def process_result_file(path: str, client: OpenAI) -> Dict[str, Any]:
    data = load_json_file(path)
    items: List[Dict[str, Any]] = data.get("items", [])
    updated_items: List[Dict[str, Any]] = []
    final_correct = 0
    total_items = len(items)
    processed = 0

    for item in items:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        llm_response = item.get("llm_response", "")
        initial_correct = bool(item.get("is_correct", False))
        judge_used = False
        judge_reason = ""
        judge_raw = ""
        corrected_flag = initial_correct

        if not initial_correct:
            judge_used = True
            corrected_flag, judge_reason, judge_raw = judge_with_reference_model(
                client, question, ground_truth, llm_response
            )

        if corrected_flag:
            final_correct += 1

        updated_item = dict(item)
        updated_item["is_correct_initial"] = initial_correct
        updated_item["is_correct_final"] = corrected_flag
        updated_item["judge_invoked"] = judge_used
        updated_item["judge_explanation"] = judge_reason
        updated_item["judge_raw_response"] = judge_raw
        updated_items.append(updated_item)
        processed += 1
        display_progress(f"{os.path.basename(path)} progressing", processed, total_items)

    total = len(updated_items)
    final_accuracy = final_correct / total if total else 0.0
    payload = {
        "model_name": data.get("model_name", NEW_MODEL_NAME),
        "source_file": os.path.relpath(path, PROJECT_PATH),
        "evaluated_at": data.get("evaluated_at"),
        "initial_accuracy": data.get("accuracy"),
        "final_accuracy": final_accuracy,
        "total_questions": total,
        "rejudged_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "items": updated_items,
    }
    return payload


def run_rejudge() -> None:
    ensure_directories()
    client = build_reference_client()
    temp_files = list(iter_temp_result_files())
    if not temp_files:
        print("temp_result can't find any files。")
        return

    for path in temp_files:
        payload = process_result_file(path, client)
        filename = os.path.basename(path)
        save_path = os.path.join(RESULT_DIR, filename)
        dump_json_file(save_path, payload)
        print(
            f"File {filename}: Initial accuracy={payload.get('initial_accuracy')}, "
            f"Rejudged accuracy={payload.get('final_accuracy'):.4f}"
        )


if __name__ == "__main__":
    run_rejudge()
