from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

from distflow.data.data_formatter import (
    AlpacaFormatter,
    ShareGptFormatter,
)
from distflow.data.data_loader import load_dataset
from distflow.embed.openai_embed import OpenAIEmbed
from distflow.mmd import MMDDistance
from distflow.utils import logger
from distflow.utils.timing import (
    get_timing_report,
    get_timings,
    reset_timing,
)

# ==================== Configuration ====================

# Embedding model (served via vLLM OpenAI-compatible API)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
OPENAI_BASE_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = "EMPTY"
MAX_CONCURRENT_REQUESTS = 1024
TRUNCATE_PROMPT_TOKENS = 40960

# RBF kernel configuration
SIGMA_CONSTANT_VALUE = 1.0
BIAS = True

# Dataset 1 configuration
DS1_CONFIG = {
    "name": "oda-math",
    "data_path": "OpenDataArena/ODA-Math-460k",
    "data_size": 5000,
    "split": "train",
    "shuffle_seed": 42,
}
formatter1 = AlpacaFormatter(
    user_key="question",
    assistant_key="response",
)

# Dataset 2 configuration
DS2_CONFIG = {
    "name": "infinity-instruct",
    "data_path": "BAAI/Infinity-Instruct",
    "data_size": 5000,
    "split": "train",
    "shuffle_seed": 42,
}
formatter2 = ShareGptFormatter(
    conversations_key="conversations",
)


# ==================== Utilities ====================


def save_json(data: dict[str, Any], path: str) -> None:
    """Save data as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== Main ====================


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Compute MMD distance between two datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory path",
    )
    args = parser.parse_args()

    output_dir: str | None = args.output

    logger.info("=" * 60)
    logger.info("MMD distance computation started")
    logger.info("=" * 60)

    # Reset timers
    reset_timing()
    total_start = time.perf_counter()

    # Load datasets
    ds1_name, ds1_items = load_dataset(
        dataset_name=DS1_CONFIG["name"],
        data_path=DS1_CONFIG["data_path"],
        load_type="datasets",
        formatter=formatter1,
        data_size=DS1_CONFIG["data_size"],
        split=DS1_CONFIG["split"],
        shuffle_seed=DS1_CONFIG["shuffle_seed"],
    )

    ds2_name, ds2_items = load_dataset(
        dataset_name=DS2_CONFIG["name"],
        data_path=DS2_CONFIG["data_path"],
        load_type="datasets",
        formatter=formatter2,
        data_size=DS2_CONFIG["data_size"],
        split=DS2_CONFIG["split"],
        shuffle_seed=DS2_CONFIG["shuffle_seed"],
    )

    logger.info(f"Dataset 1 loaded: {ds1_name}, samples: {len(ds1_items)}")
    logger.info(f"Dataset 2 loaded: {ds2_name}, samples: {len(ds2_items)}")

    # Initialize embedding model (vLLM OpenAI-compatible API)
    logger.info(f"Initializing embedder: {EMBEDDING_MODEL}")
    embedder = OpenAIEmbed(
        model_name=EMBEDDING_MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        truncate_prompt_tokens=TRUNCATE_PROMPT_TOKENS,
        truncation_side="right",
    )

    # Initialize MMD distance calculator
    logger.info(f"Initializing MMD calculator, biased estimator: {BIAS}")
    distance = MMDDistance(
        embedder=embedder,
        kernel_type="RBF",
        bias=BIAS,
        rbf_sigma=SIGMA_CONSTANT_VALUE,
    )

    # Compute MMD distance
    logger.info(f"Computing MMD distance: {ds1_name} vs {ds2_name}")
    print(f"Computing MMD distance: {ds1_name} vs {ds2_name}...")

    results = distance.compute(ds1_items, ds2_items)
    mmd_value = results[0].value

    logger.info(f"MMD distance computed: {mmd_value:.6f}")
    print(f"MMD Value: {mmd_value}")

    # Total time
    total_time = time.perf_counter() - total_start

    # Timing report
    print(get_timing_report())
    print(f"  {'Total time':<20} : {total_time:>8.3f}s")
    print("=" * 60)
    logger.info(f"Total time: {total_time:.3f}s")

    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_ds1 = ds1_name.replace("/", "_").replace(" ", "_")
        safe_ds2 = ds2_name.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(
            output_dir, f"mmd_{safe_ds1}_vs_{safe_ds2}_{timestamp}.json"
        )

        result_data = {
            "ds1": {
                "name": ds1_name,
                "data_path": DS1_CONFIG["data_path"],
                "size": len(ds1_items),
                "shuffle_seed": DS1_CONFIG["shuffle_seed"],
            },
            "ds2": {
                "name": ds2_name,
                "data_path": DS2_CONFIG["data_path"],
                "size": len(ds2_items),
                "shuffle_seed": DS2_CONFIG["shuffle_seed"],
            },
            "embedding_model": EMBEDDING_MODEL,
            "results": [
                {"name": r.name, "value": r.value, "meta": r.meta} for r in results
            ],
            "timing": {
                "details": get_timings(),
                "total_time": total_time,
            },
        }

        save_json(result_data, output_path)
        logger.info(f"Results saved to: {output_path}")
        print(f"Results saved to: {output_path}")

    logger.info("MMD distance computation finished")


if __name__ == "__main__":
    main()
