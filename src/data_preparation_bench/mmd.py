from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import cdist

from data_preparation_bench.data.types import DatasetProcessOutputItem
from data_preparation_bench.embed.base import BaseEmbed
from data_preparation_bench.utils import logger
from data_preparation_bench.utils.stats import embedding_list_summary, ndarray_summary
from data_preparation_bench.utils.timing import timing_context


class MetricsResult(BaseModel):  # type: ignore[misc]
    name: str
    value: float
    meta: dict[str, Any]


class MMDDistance:
    """MMD (Maximum Mean Discrepancy) 距离计算类.

    MMD 是一种用于衡量两个概率分布之间差异的统计量。
    通过将样本映射到再生核希尔伯特空间 (RKHS) 并比较均值来度量分布差异。
    """

    def __init__(
        self,
        embedder: BaseEmbed,
        kernel_type: Literal["RBF"] = "RBF",
        bias: bool = True,
        rbf_sigma: float = 1.0,
    ) -> None:
        """初始化 MMD 距离计算器.

        Args:
            kernel_type: 核函数类型，目前仅支持 "RBF"
            embedder: 嵌入器，用于将数据转换为向量表示
            bias: 是否使用有偏估计，默认为 True
        """
        self.embedder = embedder
        self.kernel_type = kernel_type
        assert kernel_type == "RBF", "目前仅支持 RBF 核函数"
        self.bias = bias
        self.rbf_sigma = rbf_sigma

    def _compute_kernel(self, x: list[list[float]], y: list[list[float]]) -> np.ndarray:
        assert all(
            len(xi) == len(x[0]) for xi in x
        ), "All vectors in x must have the same dimension"
        assert all(
            len(yi) == len(y[0]) for yi in y
        ), "All vectors in y must have the same dimension"
        assert len(x[0]) == len(y[0]), "Vectors in x and y must have the same dimension"

        logger.debug(
            f"计算 RBF 核矩阵, 输入维度: ({len(x)}, {len(y)}), 向量维度: {len(x[0])}"
        )
        x_np = np.array(x, dtype=np.float64)
        y_np = np.array(y, dtype=np.float64)

        dist_sq = cdist(x_np, y_np, metric="sqeuclidean")
        logger.debug(f"距离矩阵计算完成, 最大距离: {np.sqrt(dist_sq.max()):.6f}")

        k = np.exp(-dist_sq / (2 * self.rbf_sigma**2))  # (m, n)
        logger.debug(f"RBF 核矩阵计算完成, 形状: {k.shape}")

        return k.astype(np.float64)

    def compute(
        self, src: list[DatasetProcessOutputItem], tgt: list[DatasetProcessOutputItem]
    ) -> list[MetricsResult]:
        """计算两个数据集之间的距离（异步）.

        Args:
            src: 源数据集
            tgt: 目标数据集

        Returns:
            距离计算结果列表
        """
        logger.info(f"开始嵌入计算, 源数据集: {len(src)} 条, 目标数据集: {len(tgt)} 条")
        with timing_context("嵌入计算"):
            logger.debug("开始计算源数据集嵌入...")
            embedded_src_results = self.embedder.embed(src)
            logger.debug(f"源数据集嵌入完成, 共 {len(embedded_src_results)} 条")
            logger.debug("开始计算目标数据集嵌入...")
            embedded_tgt_results = self.embedder.embed(tgt)
            logger.debug(f"目标数据集嵌入完成, 共 {len(embedded_tgt_results)} 条")

        # 提取嵌入向量
        embedded_src = [item.embedding for item in embedded_src_results]
        embedded_tgt = [item.embedding for item in embedded_tgt_results]
        logger.info(
            f"嵌入向量提取完成, 维度: {len(embedded_src[0]) if embedded_src else 0}"
        )

        return self._compute_distance(embedded_src, embedded_tgt)

    def _compute_distance(
        self,
        embedded_src: list[list[float]],
        embedded_tgt: list[list[float]],
    ) -> list[MetricsResult]:
        """计算两个嵌入向量集之间的 MMD 距离.

        Args:
            embedded_src: 源数据集的嵌入向量列表
            embedded_tgt: 目标数据集的嵌入向量列表

        Returns:
            包含 MMD 距离值的 MetricsResult 列表
        """
        n_src = len(embedded_src)
        n_tgt = len(embedded_tgt)
        embedding_dim = len(embedded_src[0]) if n_src > 0 else 0

        with timing_context("MMD核计算"):
            logger.info(f"开始计算核矩阵, 源样本数: {n_src}, 目标样本数: {n_tgt}")
            src_src = self._compute_kernel(embedded_src, embedded_src)
            tgt_tgt = self._compute_kernel(embedded_tgt, embedded_tgt)
            src_tgt = self._compute_kernel(embedded_src, embedded_tgt)
            logger.debug("核矩阵计算完成")

            # 确保是 ndarray（满足类型检查）
            assert isinstance(src_src, np.ndarray)
            assert isinstance(tgt_tgt, np.ndarray)
            assert isinstance(src_tgt, np.ndarray)

            k_xx_mean = float(src_src.mean())
            k_yy_mean = float(tgt_tgt.mean())
            k_xy_mean = float(src_tgt.mean())

            if self.bias:
                # 有偏 MMD 估计
                mmd_value = k_xx_mean + k_yy_mean - 2 * k_xy_mean
                logger.debug(f"有偏 MMD 估计: {mmd_value:.6f}")
            else:
                # 无偏 MMD 估计：排除对角线元素
                mmd_value = (
                    (src_src.sum() - n_src) / (n_src * (n_src - 1))
                    + (tgt_tgt.sum() - n_tgt) / (n_tgt * (n_tgt - 1))
                    - 2 * k_xy_mean
                )
                logger.debug(f"无偏 MMD 估计: {mmd_value:.6f}")

        # 获取核函数参数（如 RBF sigma）
        kernel_params: dict[str, object] = {
            "rbf_sigma": self.rbf_sigma,
        }

        meta = {
            "estimator": "biased" if self.bias else "unbiased",
            "n_src": n_src,
            "n_tgt": n_tgt,
            "embedding_dim": embedding_dim,
            "kernel_terms": {
                "K_xx_mean": k_xx_mean,
                "K_yy_mean": k_yy_mean,
                "K_xy_mean": k_xy_mean,
            },
            "kernel_params": kernel_params,
            "kernel_matrices": {
                "K_xx": ndarray_summary(src_src, name="K(src,src)"),
                "K_yy": ndarray_summary(tgt_tgt, name="K(tgt,tgt)"),
                "K_xy": ndarray_summary(src_tgt, name="K(src,tgt)"),
            },
            "embeddings": {
                "src": embedding_list_summary(embedded_src, name="src_embeddings"),
                "tgt": embedding_list_summary(embedded_tgt, name="tgt_embeddings"),
            },
        }

        logger.info(f"MMD 计算完成: {mmd_value:.6f}")
        return [MetricsResult(name="MMD", value=mmd_value, meta=meta)]
