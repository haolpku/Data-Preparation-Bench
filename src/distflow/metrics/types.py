from typing import Any

from pydantic import BaseModel


class MetricsResult(BaseModel):  # type: ignore[misc]
    name: str
    value: float
    meta: dict[str, Any]
