# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://172.96.160.199:3000/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://172.96.160.199:3000/v1"                 
REFER_API_KEY = "sk-dummy"  
# ========================================================

import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATA_PATH = Path("eval/data/medmcqa/input/medmcqa_input.jsonl")
TEMP_DIR = Path("temp_result")
RESULT_DIR = Path("result")

SYSTEM_PROMPT = (
    "You are a medical QA assistant. Answer with the single best option letter. "
    "Your final line must be exactly: Answer: <LETTER> (e.g., Answer: A)."
)


REFER_PROMPT_TEMPLATE = (
    "You are an evaluator. Given the model's answer (which may include reasoning and a final answer) "
    "and the correct answer (a single letter), determine what answer the model actually chose. "
    "The model might express its answer in various ways; your task is to extract the final choice. "
    "Output only the letter (A, B, C, D, or E). Do not include any other text.\n\n"
    "Model's answer:\n{full_answer}\n\n"
    "Correct answer: {gold}\n\n"
    "Output the letter that the model chose:"
)

def load_data(path: Path, full: bool):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if not full and len(items) >= 3:
                break
    return items

def call_llm(question: str):
    url = NEW_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {NEW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NEW_MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_refer_model(prompt: str):
    url = REFER_API_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {REFER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REFER_MODEL_NAME,
        "temperature": 0,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def extract_answer(text: str):
    # Prefer explicit "Answer: X" on any line
    m = re.search(r"^\s*Answer\s*:\s*([A-Z])\b", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone option letter A-D/E
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()
    return ""

def main():
    items = load_data(DATA_PATH, IS_FULL_EVAL)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = TEMP_DIR / f"{NEW_MODEL_NAME}_results.jsonl"

    correct = 0
    total = 0

    with temp_path.open("w", encoding="utf-8") as out:
        for item in tqdm(items, desc="Evaluating", unit="q"):
            qid = item.get("id")
            question = item.get("question")
            gold = ""
            label = item.get("label")
            if isinstance(label, list) and label:
                gold = label[0]
            elif isinstance(label, str):
                gold = label

            try:
                full_answer = call_llm(question)
            except Exception as e:
                full_answer = f"ERROR: {e}"

            pred = extract_answer(full_answer)


            final_pred = pred
            refer_used = False
            refer_output = None

            if pred and gold and pred.upper() != gold.upper():
                try:
                    prompt = REFER_PROMPT_TEMPLATE.format(full_answer=full_answer, gold=gold)
                    refer_response = call_refer_model(prompt)
     
                    m = re.search(r"\b([A-E])\b", refer_response.strip(), flags=re.IGNORECASE)
                    if m:
                        corrected_pred = m.group(1).upper()
                        final_pred = corrected_pred
                        refer_used = True
                        refer_output = refer_response
                except Exception as e:

                    refer_output = f"ERROR: {e}"

            total += 1
            if final_pred and gold and final_pred.upper() == gold.upper():
                correct += 1

            record = {
                "id": qid,
                "gold": gold,
                "llm_answer": full_answer,
                "pred": final_pred,               
                "original_pred": pred,            
                "refer_used": refer_used,
                "refer_output": refer_output,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


            time.sleep(0.1)

    acc = correct / total if total else 0.0
    result_path = RESULT_DIR / f"accuracy_{NEW_MODEL_NAME}.txt"
    with result_path.open("w", encoding="utf-8") as f:
        f.write(f"model: {NEW_MODEL_NAME}\n")
        f.write(f"total: {total}\n")
        f.write(f"correct: {correct}\n")
        f.write(f"accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    main()