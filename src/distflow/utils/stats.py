"""矩阵 / 向量统计摘要工具."""

from typing import Any

import numpy as np


def ndarray_summary(arr: np.ndarray, name: str = "") -> dict[str, Any]:
    """生成 numpy 数组的统计摘要.

    包含形状、元素数量、均值、标准差、最小值、最大值、中位数、
    各分位数等信息。对于二维矩阵还包含对角线统计。

    Args:
        arr: 待统计的 numpy 数组
        name: 可选的名称标签

    Returns:
        统计摘要字典
    """
    summary: dict[str, Any] = {}
    if name:
        summary["name"] = name

    summary["shape"] = list(arr.shape)
    summary["dtype"] = str(arr.dtype)
    summary["size"] = int(arr.size)

    if arr.size == 0:
        return summary

    flat = arr.astype(np.float64).ravel()
    summary["mean"] = float(np.mean(flat))
    summary["std"] = float(np.std(flat))
    summary["min"] = float(np.min(flat))
    summary["max"] = float(np.max(flat))
    summary["median"] = float(np.median(flat))
    summary["q25"] = float(np.percentile(flat, 25))
    summary["q75"] = float(np.percentile(flat, 75))
    summary["sum"] = float(np.sum(flat))
    summary["nonzero_count"] = int(np.count_nonzero(flat))
    summary["nan_count"] = int(np.isnan(flat).sum())
    summary["inf_count"] = int(np.isinf(flat).sum())

    # 对于二维方阵，提供对角线统计
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        diag = np.diag(arr).astype(np.float64)
        summary["diagonal"] = {
            "mean": float(np.mean(diag)),
            "std": float(np.std(diag)),
            "min": float(np.min(diag)),
            "max": float(np.max(diag)),
        }

    return summary


def embedding_list_summary(
    embeddings: list[list[float]], name: str = ""
) -> dict[str, Any]:
    """生成嵌入向量列表的统计摘要.

    包含样本数、向量维度、L2 范数分布、各维度统计等。

    Args:
        embeddings: 嵌入向量列表
        name: 可选的名称标签

    Returns:
        统计摘要字典
    """
    arr = np.array(embeddings, dtype=np.float64)
    summary: dict[str, Any] = {}
    if name:
        summary["name"] = name

    summary["num_samples"] = arr.shape[0]
    summary["embedding_dim"] = arr.shape[1] if arr.ndim > 1 else 0

    if arr.size == 0:
        return summary

    # L2 范数统计
    norms = np.linalg.norm(arr, axis=1)
    summary["l2_norm"] = {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "median": float(np.median(norms)),
    }

    # 各维度的聚合统计
    dim_means = np.mean(arr, axis=0)
    dim_stds = np.std(arr, axis=0)
    summary["per_dim"] = {
        "mean_of_means": float(np.mean(dim_means)),
        "std_of_means": float(np.std(dim_means)),
        "mean_of_stds": float(np.mean(dim_stds)),
        "std_of_stds": float(np.std(dim_stds)),
    }

    # 全局值域
    summary["global"] = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    return summary
