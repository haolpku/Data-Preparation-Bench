from typing import cast

import numpy as np
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr

from distflow.data.dataset import DistflowDataset
from distflow.metrics.protocol import MetricsProtocol


class BenchmarkResult(BaseModel):
    slope: float
    intercept: float
    pearson_corr: float
    spearman_corr: float
    pearson_p_value: float
    spearman_p_value: float


class DataQualityEvaluatorBenchmark:
    def __init__(self, datasets: list[DistflowDataset], accuracys: dict[str, float]):
        self.datasets = datasets
        self.accuracys = accuracys
        assert set(accuracys.keys()) == {
            dataset.name for dataset in datasets
        }, "Accuracy keys must match dataset names"

    def run_benchmark(self, metrics: MetricsProtocol) -> dict[str, BenchmarkResult]:
        metrics_results_mapping: dict[str, dict[str, float]] = {}
        for dataset in self.datasets:
            metric_results = metrics.score(dataset)
            for metric_result in metric_results:
                if metric_result.name not in metrics_results_mapping:
                    metrics_results_mapping[metric_result.name] = {}
                metrics_results_mapping[metric_result.name][
                    dataset.name
                ] = metric_result.value
        accuracy_metrics_pairs: dict[str, list[tuple[float, float]]] = {}
        for metric_name, dataset_metric_mapping in metrics_results_mapping.items():
            for dataset_name, metric_value in dataset_metric_mapping.items():
                accuracy_value = self.accuracys[dataset_name]
                if metric_name not in accuracy_metrics_pairs:
                    accuracy_metrics_pairs[metric_name] = []
                accuracy_metrics_pairs[metric_name].append(
                    (accuracy_value, metric_value)
                )

        benchmark_results: dict[str, BenchmarkResult] = {}
        for metric_name, pairs in accuracy_metrics_pairs.items():
            benchmark_results[metric_name] = self._compute_benchmark_result(pairs)

        return benchmark_results

    def _compute_benchmark_result(
        self, accuracy_metric_pairs: list[tuple[float, float]]
    ) -> BenchmarkResult:
        accuracies = np.array([pair[0] for pair in accuracy_metric_pairs])
        metrics = np.array([pair[1] for pair in accuracy_metric_pairs])
        slope, intercept = np.polyfit(metrics, accuracies, 1)
        pearson_result = pearsonr(metrics, accuracies)
        pearson_corr = cast(float, pearson_result[0])
        pearson_p_value = cast(float, pearson_result[1])

        spearman_result = spearmanr(metrics, accuracies)
        spearman_corr = cast(float, spearman_result[0])
        spearman_p_value = cast(float, spearman_result[1])

        return BenchmarkResult(
            slope=slope,
            intercept=intercept,
            pearson_corr=pearson_corr,
            spearman_corr=spearman_corr,
            pearson_p_value=pearson_p_value,
            spearman_p_value=spearman_p_value,
        )
