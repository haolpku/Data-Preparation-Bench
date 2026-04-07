import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd

from eval_metrics import resp2ans
from load_models import build_model


# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                           
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy" 
# ========================================================


MAX_OUTPUT_TOKENS = 1024
MAX_WORKERS = 8
SAMPLE_SIZE = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.environ.get("PROJECT_PATH", os.path.dirname(SCRIPT_DIR))
DATASET_FILE = os.path.join(PROJECT_PATH, "dataset", "validation_set.csv")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt_template", "bool_DA.txt")
TEMP_RESULT_DIR = os.path.join(PROJECT_PATH, "temp_result")
RESULT_DIR = os.path.join(PROJECT_PATH, "result")


def ensure_directories() -> None:
    os.makedirs(TEMP_RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_prompt_template() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as fp:
        return fp.read()


def normalize_ground_truth(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "true", "t", "yes"}:
            return 1
        if stripped in {"0", "false", "f", "no"}:
            return 0
        try:
            num = float(stripped)
            if num == 1.0:
                return 1
            if num == 0.0:
                return 0
        except ValueError:
            return None
    return None


def load_bool_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILE)
    df = df[df["task"] == "bool"].copy()
    df["ground_truth_norm"] = df["ground_truth"].apply(normalize_ground_truth)
    df = df.dropna(subset=["ground_truth_norm"])  
    df = df[df["figure"].isna()]  
    if not IS_FULL_EVAL:
        df = df.head(min(SAMPLE_SIZE, len(df)))
    return df


def build_prompt(template: str, question: str) -> str:
    return template.format(knowledge="", question=question)


def display_progress(prefix: str, current: int, total: int) -> None:
    bar_len = 30
    filled = int(bar_len * current / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = (current / total * 100) if total else 0
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent:.1f}%)")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def evaluate_row(model: build_model, template: str, row: pd.Series) -> Dict[str, Any]:
    question = str(row["question"])
    ground_truth = int(row["ground_truth_norm"])
    qa_id = row.get("id", "")
    prompt = build_prompt(template, question)
    record: Dict[str, Any] = {
        "id": qa_id,
        "question": question,
        "ground_truth": ground_truth,
        "llm_response": "",
        "parsed_answer": None,
        "is_correct": False,
        "error": "",
        "token_usage": "",
        "model_name": NEW_MODEL_NAME,
    }

    try:
        response, token_usage = model.get_model_response(
            sys_msg="",
            msg=prompt,
            model_name=NEW_MODEL_NAME,
            image_pt="",
            sys_msg_bool=0,
            max_token_=MAX_OUTPUT_TOKENS,
        )
        record["llm_response"] = response
        record["token_usage"] = token_usage
    except Exception as exc:  
        record["error"] = str(exc)
        return record

    parsed_answer = resp2ans("bool", record["llm_response"])
    record["parsed_answer"] = parsed_answer
    if isinstance(parsed_answer, (int, float)):
        parsed_int = int(parsed_answer)
        record["is_correct"] = parsed_int == ground_truth
    else:
        record["is_correct"] = False
    return record


def run_evaluation() -> Dict[str, Any]:
    ensure_directories()
    dataset = load_bool_dataset()
    if dataset.empty:
        raise RuntimeError("Not Find bool data。")

    prompt_template = load_prompt_template()
    model = build_model(NEW_MODEL_NAME)

    rows: List[pd.Series] = [dataset.iloc[idx] for idx in range(len(dataset))]
    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    progress_lock = Lock()
    completed = 0

    print(f"Start：Model ={NEW_MODEL_NAME}, num ={len(rows)}, Workers ={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(evaluate_row, model, prompt_template, row): idx
            for idx, row in enumerate(rows)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  
                results[idx] = {
                    "id": rows[idx].get("id", ""),
                    "question": rows[idx].get("question", ""),
                    "ground_truth": int(rows[idx]["ground_truth_norm"]),
                    "llm_response": "",
                    "parsed_answer": None,
                    "is_correct": False,
                    "error": str(exc),
                    "token_usage": "",
                    "model_name": NEW_MODEL_NAME,
                }
            finally:
                with progress_lock:
                    completed += 1
                    display_progress("Processing", completed, len(rows))


    filled_results: List[Dict[str, Any]] = [r for r in results if r is not None]  
    correct_count = sum(1 for item in filled_results if item.get("is_correct"))
    total = len(filled_results)
    accuracy = correct_count / total if total else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{NEW_MODEL_NAME.replace('/', '_')}_bool_{timestamp}.json"
    save_path = os.path.join(TEMP_RESULT_DIR, filename)
    payload = {
        "model_name": NEW_MODEL_NAME,
        "evaluated_at": timestamp,
        "dataset_file": os.path.relpath(DATASET_FILE, PROJECT_PATH),
        "is_full_eval": IS_FULL_EVAL,
        "accuracy": accuracy,
        "total_questions": total,
        "items": filled_results,
    }
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"Finished, accuracy={accuracy:.4f}，results saved to: {save_path}")
    return payload


if __name__ == "__main__":
    run_evaluation()
