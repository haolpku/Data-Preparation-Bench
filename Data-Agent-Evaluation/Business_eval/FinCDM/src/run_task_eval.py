# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                           
REFER_API_BASE_URL = "http://XXX/v1"                 
REFER_API_KEY = "sk-dummy"  
# ========================================================
"""Run multi-choice financial QA evaluation for the English FinCDM datasets."""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

DATA_DIR = Path("data_en")
TEMP_DIR = Path("temp_result")
PARTIAL_SAMPLE_SIZE = 20
DEFAULT_CONCURRENCY = 5
SYSTEM_PROMPT = (
    "You are a meticulous financial exam coach. "
    "Read the prompt carefully and respond only in English."
)
OUTPUT_TEMPLATE = (
    "Return JSON strictly in the format "
    '{"answer": "A", "confidence": "high", "analysis": "One-sentence reasoning."}. '
    "answer must be uppercase A/B/C/D, analysis must be at least one sentence."
)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_examples(items: List[Dict[str, Any]], full_eval: bool, limit: int) -> List[Dict[str, Any]]:
    if full_eval or len(items) <= limit:
        return items
    return random.sample(items, k=min(limit, len(items)))


def parse_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r'"?answer"?\s*[:：]\s*"?([ABCD])"?', raw_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", raw_text.upper())
    return match.group(1) if match else ""


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{question}\n\n{OUTPUT_TEMPLATE}"
        },
    ]


def _render_progress(prefix: str, completed: int, total: int) -> None:
    total = max(total, 1)
    bar_len = 30
    filled = int(bar_len * completed / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {completed}/{total}", end="", flush=True)


async def query_model(client: AsyncOpenAI, question: str, model_name: str) -> str:
    messages = build_messages(question)
    completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
        top_p=1.0,
    )
    return completion.choices[0].message.content.strip()


async def evaluate_item(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        try:
            raw_answer = await query_model(client, item["query"], model_name)
        except Exception as exc:
            raw_answer = f"[ERROR] {exc}"
        parsed_answer = parse_answer(raw_answer)
        standard = (item.get("answer") or "").strip().upper()
        return {
            "id": item.get("id"),
            "standard_answer": standard,
            "llm_response": raw_answer,
            "parsed_answer": parsed_answer,
            "is_correct": bool(parsed_answer) and parsed_answer == standard,
        }


async def evaluate_dataset(
    dataset_path: Path,
    client: AsyncOpenAI,
    model_name: str,
    max_concurrency: int,
    full_eval: bool,
) -> Dict[str, Any]:
    data = load_dataset(dataset_path)
    selected = select_examples(data, full_eval, PARTIAL_SAMPLE_SIZE)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        evaluate_item(client, item, model_name, semaphore)
        for item in selected
    ]
    progress_desc = f"Evaluating {dataset_path.stem}"
    results: List[Dict[str, Any]] = []
    total_tasks = len(tasks)
    completed = 0
    for future in asyncio.as_completed(tasks):
        results.append(await future)
        completed += 1
        _render_progress(progress_desc, completed, total_tasks)
    print()
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct / total if total else 0.0
    return {
        "task_name": dataset_path.stem,
        "model_name": model_name,
        "total_questions": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "results": results,
    }


def save_results(dataset_summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset_summary['task_name']}__{dataset_summary['model_name']}.json"
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
    return output_path


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(api_key=NEW_API_KEY, base_url=NEW_BASE_URL)
    dataset_paths = (
        [Path(p) for p in args.datasets]
        if args.datasets
        else sorted(DATA_DIR.glob("*.json"))
    )
    if not dataset_paths:
        raise SystemExit("No dataset files found.")
    for dataset_path in dataset_paths:
        summary = await evaluate_dataset(
            dataset_path,
            client,
            NEW_MODEL_NAME,
            args.max_concurrency,
            IS_FULL_EVAL,
        )
        output_path = save_results(summary, TEMP_DIR)
        print(
            f"Saved {summary['task_name']} results "
            f"({summary['correct_count']}/{summary['total_questions']} correct, "
            f"acc={summary['accuracy']:.2%}) -> {output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinCDM evaluation")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset file paths (default: all JSONs under data_en/)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent OpenAI requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
