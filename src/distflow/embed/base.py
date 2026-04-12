from abc import ABC, abstractmethod

from distflow.embed.types import EmbeddingInputItem, EmbeddingResult


class BaseEmbed(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult | None]:
        """异步嵌入计算.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表，失败项为 None
        """
