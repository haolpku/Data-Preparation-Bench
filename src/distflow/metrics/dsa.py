from typing import Literal

from distflow.data.dataset import DistflowDataset
from distflow.embed.base import BaseEmbed
from distflow.metrics.types import MetricsResult
from distflow.mmd import MMDDistance


class DASMetric:
    def __init__(
        self,
        target_set: DistflowDataset,
        embedder: BaseEmbed,
        kernel_type: Literal["RBF"] = "RBF",
        bias: bool = True,
        rbf_sigma: float = 1.0,
        max_fail_ratio: float = 0.02,
    ):
        self.distance_calculator = MMDDistance(
            embedder=embedder,
            kernel_type=kernel_type,
            bias=bias,
            rbf_sigma=rbf_sigma,
            max_fail_ratio=max_fail_ratio,
        )
        self.target_set = target_set

    def compute(self, candidate_set: DistflowDataset) -> list[MetricsResult]:
        target_data = self.target_set.load()
        candidate_data = candidate_set.load()

        mmd_value, meta = self.distance_calculator.compute(target_data, candidate_data)

        return [MetricsResult(name="DAS", value=-mmd_value, meta=meta)]
