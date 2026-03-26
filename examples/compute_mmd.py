from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any

from data_preparation_bench.data.data_formatter import (
    AlpacaFormatter,
    ShareGptFormatter,
)
from data_preparation_bench.data.data_loader import load_dataset
from data_preparation_bench.embed.vllm import VllmEmbed
from data_preparation_bench.mmd import MMDDistance
from data_preparation_bench.utils import logger
from data_preparation_bench.utils.timing import (
    get_timing_report,
    get_timings,
    reset_timing,
)

# ==================== 配置区域 ====================

# 嵌入模型配置
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.95
MAX_NUM_SEQS = 4096
TRUNCATE_MAX_LENGTH = 40960

# RBF 核函数配置
SIGMA_CONSTANT_VALUE = 1.0
BIAS = True

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


# ==================== 工具函数 ====================


def save_json(data: dict[str, Any], path: str) -> None:
    """保存数据为 JSON 文件."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== 主函数 ====================


async def main_async(output_dir: str | None) -> None:
    """异步主函数."""
    logger.info("=" * 60)
    logger.info("MMD 距离计算任务启动")
    logger.info("=" * 60)

    # 重置计时器
    reset_timing()
    total_start = time.perf_counter()

    # 加载数据集
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

    logger.info(f"数据集1加载完成: {ds1_name}, 样本数: {len(ds1_items)}")
    logger.info(f"数据集2加载完成: {ds2_name}, 样本数: {len(ds2_items)}")

    # 初始化嵌入模型
    logger.info(f"初始化嵌入模型: {EMBEDDING_MODEL}")
    embedder = VllmEmbed(
        model_name=EMBEDDING_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        truncate_max_length=TRUNCATE_MAX_LENGTH,
    )

    # 初始化 MMD 距离计算器
    logger.info(f"初始化 MMD 距离计算器, 有偏估计: {BIAS}")
    distance = MMDDistance(
        embedder=embedder,
        kernel_type="RBF",
        bias=BIAS,
        rbf_sigma=SIGMA_CONSTANT_VALUE,
    )

    # 计算 MMD 距离
    logger.info(f"开始计算 MMD 距离: {ds1_name} vs {ds2_name}")
    print(f"Computing MMD distance: {ds1_name} vs {ds2_name}...")

    results = await distance.async_compute(ds1_items, ds2_items)
    mmd_value = results[0].value

    logger.info(f"MMD 距离计算完成: {mmd_value:.6f}")
    print(f"MMD Value: {mmd_value}")

    # 计算总时间
    total_time = time.perf_counter() - total_start

    # 打印时间报告
    print(get_timing_report())
    print(f"  {'总耗时':<20} : {total_time:>8.3f}s")
    print("=" * 60)
    logger.info(f"任务总耗时: {total_time:.3f} 秒")

    # 保存结果
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
        logger.info(f"结果已保存到: {output_path}")
        print(f"Results saved to: {output_path}")

    logger.info("MMD 距离计算任务完成")


def main() -> None:
    """程序入口点."""
    parser = argparse.ArgumentParser(description="计算两个数据集之间的 MMD 距离")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录路径",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.output))


if __name__ == "__main__":
    main()
