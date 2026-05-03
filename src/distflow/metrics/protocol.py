from typing import Protocol

from distflow.data.dataset import DistflowDataset
from distflow.metrics.types import MetricsResult


class MetricsProtocol(Protocol):
    def score(self, dataset: DistflowDataset) -> list[MetricsResult]: ...
