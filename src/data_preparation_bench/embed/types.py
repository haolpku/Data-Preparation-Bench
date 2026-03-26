from typing import Any, TypeAlias

from pydantic import BaseModel

from data_preparation_bench.data.types import DatasetProcessOutputItem

EmbeddingInputItem: TypeAlias = DatasetProcessOutputItem


class EmbeddingResult(BaseModel):  # type: ignore[misc]
    embedding: list[float]
    data_item: DatasetProcessOutputItem
    meta: dict[str, Any]
