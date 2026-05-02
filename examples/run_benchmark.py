"""
Example: Running the Data Quality Evaluator Benchmark

This script demonstrates how to use DataQualityEvaluatorBenchmark to evaluate
how well a custom data-quality metric correlates with downstream task accuracy.

Steps:
1. Prepare one or more DistflowDataset objects.
2. Manually fill in the downstream accuracy for each dataset (the `accuracy/`
   directory is not committed, so you must provide these values yourself).
   Keys must match the `dataset_name` of each DistflowDataset.
3. Implement a metrics class with a `score(dataset) -> list[MetricsResult]` method.
4. Run the benchmark — it will compute Pearson / Spearman correlation and a
   linear fit between your metric and the provided accuracies.

Run:
    uv run examples/run_benchmark.py
"""

from distflow.benchmark import DataQualityEvaluatorBenchmark
from distflow.data.data_formatter import AlpacaFormatter, ShareGptFormatter
from distflow.data.dataset import DistflowDataset
from distflow.metrics.types import MetricsResult


class DummyMetrics:
    def score(self, dataset: DistflowDataset) -> list[MetricsResult]:
        return [MetricsResult(name="dummy_metric", value=0.5, meta={})]


def main():
    # 1. Define the datasets to benchmark
    datasets = [
        DistflowDataset(
            dataset_name="dataflow",
            data_path="OpenDCAI/dataflow-instruct-10k",
            load_type="datasets",
            formatter=ShareGptFormatter(conversations_key="conversations"),
            data_size=100,
            name="default",
            split="train",
            shuffle_seed=42,
        ),
        DistflowDataset(
            dataset_name="infinity-instruct",
            data_path="BAAI/Infinity-Instruct",
            load_type="datasets",
            formatter=ShareGptFormatter(conversations_key="conversations"),
            data_size=100,
            name="0625",
            split="train",
            shuffle_seed=42,
        ),
        DistflowDataset(
            dataset_name="openr1",
            data_path="open-r1/OpenR1-Math-220k",
            load_type="datasets",
            formatter=AlpacaFormatter(
                user_key="problem",
                assistant_key="solution",
            ),
            data_size=100,
            name="default",
            split="train",
            shuffle_seed=42,
        ),
    ]

    # 2. Provide downstream accuracy for each dataset manually.
    #    Keys must match the `dataset_name` of the DistflowDataset objects.
    accuracys = {
        "dataflow": 0.2,
        "infinity-instruct": 0.3,
        "openr1": 0.4,
    }

    # 3. Run the benchmark with your metric
    benchmark = DataQualityEvaluatorBenchmark(
        datasets=datasets,
        accuracys=accuracys,
    )
    result = benchmark.run_benchmark(metrics=DummyMetrics())
    print(result)


if __name__ == "__main__":
    main()
