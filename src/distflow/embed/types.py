from typing import Any, TypeAlias

from pydantic import BaseModel

from distflow.data.types import DatasetProcessOutputItem

EmbeddingInputItem: TypeAlias = DatasetProcessOutputItem


class EmbeddingResult(BaseModel):  # type: ignore[misc]
    embedding: list[float]
    data_item: DatasetProcessOutputItem
    meta: dict[str, Any]
