from typing import Any

from pydantic import BaseModel


class MessageData(BaseModel):  # type: ignore[misc]
    role: str
    content: str | dict[str, Any]


class DatasetProcessOutputItem(BaseModel):  # type: ignore[misc]
    messages: list[MessageData]
    meta: dict[str, Any]
