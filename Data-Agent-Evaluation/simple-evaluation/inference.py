#!/usr/bin/env python3
"""
AsyncOpenAI inference script for unified evaluation JSONL.
"""

import argparse
import asyncio
import json
import os
from typing import Any, Dict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def process_one(
    client: AsyncOpenAI,
    record: Dict[str, Any],
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    messages = record["messages"]
    completion_args = record.get("completion_args", {})

    async with semaphore:
        for attempt in range(3):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                }
                if completion_args.get("temperature") is not None:
                    kwargs["temperature"] = completion_args["temperature"]
                if completion_args.get("max_tokens") is not None:
                    kwargs["max_tokens"] = completion_args["max_tokens"]
                if completion_args.get("top_p") is not None:
                    kwargs["top_p"] = completion_args["top_p"]

                resp = await client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content

                out_record = dict(record)
                out_record["messages"] = list(messages) + [
                    {"role": "assistant", "content": content}
                ]
                out_record["raw_response"] = resp.model_dump()
                return out_record
            except Exception as e:
                if attempt == 2:
                    out_record = dict(record)
                    out_record["messages"] = list(messages) + [
                        {"role": "assistant", "content": "Fail to generate response"}
                    ]
                    out_record["raw_response"] = {"error": str(e)}
                    return out_record
                await asyncio.sleep(1 + attempt)

    # Unreachable, but keeps type checker happy
    return dict(record)


async def main():
    parser = argparse.ArgumentParser(
        description="Run AsyncOpenAI inference on a unified evaluation JSONL file."
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to the input JSONL file (e.g., extracted_data.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the output JSONL file.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent API requests (default: 8).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to use for inference.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of the OpenAI-compatible API endpoint.",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key for authentication.",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key=args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "inference_results.jsonl")

    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records. Starting inference with concurrency={args.concurrency} ...")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_one(client, rec, args.model_name, semaphore)
        for rec in records
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for result in results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    fail_count = sum(
        1 for r in results
        if r["messages"][-1]["content"] == "Fail to generate response"
    )
    print(f"Done. Results saved to {output_path}")
    print(f"Total: {len(results)}, Failed: {fail_count}")


if __name__ == "__main__":
    asyncio.run(main())
