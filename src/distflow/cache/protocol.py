from typing import Any, Protocol


class CacheProtocol(Protocol):
    async def load_cache(self, cache_key: str) -> dict[str, Any] | None: ...

    async def save_cache(self, cache_key: str, cache_value: dict[str, Any]) -> bool: ...
