# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                            
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================
"""Re-score incorrect answers with the referee model and update accuracy."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

TEMP_DIR = Path("temp_result")
RESULT_DIR = Path("result")
DEFAULT_CONCURRENCY = 3
REFEREE_SYSTEM_PROMPT = (
    "You are a meticulous exam referee. "
    "Decide strictly whether the model's answer matches the gold answer."
)
REFEREE_INSTRUCTION = (
    "Return JSON like {\"is_correct\": true, \"reason\": \"brief justification\"}. "
    "reason must briefly explain the decision."
)


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


async def judge_entry(
    client: AsyncOpenAI,
    entry: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    user_message = (
        f"Question ID: {entry.get('id')}\n"
        f"Gold answer: {entry.get('standard_answer', '')}\n"
        f"Answer parsed by code: {entry.get('parsed_answer', '')}\n"
        "Original model response:\n<<<\n"
        f"{entry.get('llm_response', '')}\n>>><<\n"
        "Decide whether the original response clearly selects the same option letter as the gold answer. "
        "If no valid letter can be identified, return false.\n"
        f"{REFEREE_INSTRUCTION}"
    )
    async with semaphore:
        try:
            completion = await client.chat.completions.create(
                model=REFER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": REFEREE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = completion.choices[0].message.content.strip()
        except Exception as exc:
            raw = f"[ERROR] {exc}"
    parsed = parse_referee_response(raw)
    return {
        "referee_response": raw,
        "referee_correct": bool(parsed.get("is_correct")),
        "referee_reason": parsed.get("reason", ""),
    }


async def process_file(
    path: Path,
    client: AsyncOpenAI,
    max_concurrency: int,
) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks: List[asyncio.Task] = []
    indices: List[int] = []
    for idx, entry in enumerate(data.get("results", [])):
        if entry.get("is_correct"):
            entry["final_correct"] = True
            continue
        indices.append(idx)
        tasks.append(asyncio.create_task(judge_entry(client, entry, semaphore)))
    updates = await asyncio.gather(*tasks) if tasks else []
    for idx, update in zip(indices, updates):
        entry = data["results"][idx]
        entry.update(update)
        entry["final_correct"] = entry.get("referee_correct", False)
    for idx, entry in enumerate(data.get("results", [])):
        if entry.get("is_correct"):
            entry.setdefault("referee_correct", False)
            entry.setdefault("final_correct", True)
    total = len(data.get("results", []))
    initial_correct = data.get("correct_count", 0)
    initial_accuracy = data.get("accuracy", 0.0)
    final_correct = sum(1 for entry in data.get("results", []) if entry.get("final_correct"))
    final_accuracy = final_correct / total if total else 0.0
    data["initial_correct_count"] = initial_correct
    data["initial_accuracy"] = initial_accuracy
    data["correct_count"] = final_correct
    data["accuracy"] = final_accuracy
    data["referee_model_name"] = REFER_MODEL_NAME
    data["referee_checked_questions"] = len(indices)
    return data


def save_result(data: Dict[str, Any], source_path: Path) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / source_path.name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(api_key=REFER_API_KEY, base_url=REFER_API_BASE_URL)
    target_files = (
        [Path(p) for p in args.files]
        if args.files
        else sorted(TEMP_DIR.glob("*.json"))
    )
    if not target_files:
        raise SystemExit("No temp_result JSON files found.")
    for path in target_files:
        data = await process_file(path, client, args.max_concurrency)
        output_path = save_result(data, path)
        print(
            f"Updated {path.name}: final acc={data['accuracy']:.2%} "
            f"({data['correct_count']}/{data.get('total_questions', len(data.get('results', [])))}) -> {output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate incorrect answers with referee model")
    parser.add_argument("--files", nargs="*", help="Specific temp_result JSON files to process")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent referee calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
