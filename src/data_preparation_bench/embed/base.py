from abc import ABC, abstractmethod

from data_preparation_bench.embed.types import EmbeddingInputItem, EmbeddingResult


class BaseEmbed(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult]:
        """异步嵌入计算.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
