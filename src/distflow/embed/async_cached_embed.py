"""异步缓存 OpenAI Embedding 包装器.

每条请求粒度：查缓存 → embed → 写缓存，整体受 semaphore 控制。
不继承 BaseEmbed（BaseEmbed.embed 是同步接口），提供纯异步接口。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from distflow.cache.protocol import CacheProtocol
from distflow.embed.openai_embed import OpenAIEmbed
from distflow.embed.types import EmbeddingInputItem, EmbeddingResult
from distflow.utils import logger


def dict_to_hash(d: dict[Any, Any]) -> str:
    """生成字典的SHA256哈希摘要（与 cache_wrapper.py 保持一致）."""
    s = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()


class AsyncCachedOpenAIEmbed:
    """异步缓存 OpenAI Embedding 包装器.

    每条请求粒度：查缓存 → embed → 写缓存，整体受 semaphore 控制。
    不继承 BaseEmbed（BaseEmbed.embed 是同步接口），提供纯异步接口。
    """

    def __init__(
        self,
        embedder: OpenAIEmbed,
        cache: CacheProtocol,
        semaphore: asyncio.Semaphore,
        cache_model_id: str | None = None,
        legacy_key: bool = False,
    ) -> None:
        """初始化异步缓存 OpenAI Embedding 包装器.

        Args:
            embedder: 底层 OpenAI embedder
            cache: 符合 CacheProtocol 的缓存实现（如 RedisCache）
            semaphore: 外部传入的全局并发信号量
            cache_model_id: 用于缓存键的模型标识符，默认为 embedder 的 model_name
            legacy_key: 是否使用旧版缓存键格式（包含完整 data_item）
        """
        self._embedder = embedder
        self._cache = cache
        self._semaphore = semaphore
        self._legacy_key = legacy_key

        # 模型标识符
        self._model_path = (
            getattr(embedder, "model_name", None)
            or getattr(embedder, "model_path", None)
            or getattr(embedder, "_model_name", None)
            or "unknown"
        )
        self._cache_model_id = cache_model_id if cache_model_id else self._model_path

        # dummy semaphore，用于传给 OpenAIEmbed._embed_single
        # 外层已用全局 semaphore 控制并发，内层不再限制
        self._dummy_semaphore = asyncio.Semaphore(999999)

        # 确保 embedder 客户端已初始化
        self._embedder._ensure_initialized()

    def _build_cache_key(self, item: EmbeddingInputItem) -> str:
        """构建缓存键（与 cache_wrapper.py 保持一致）.

        Args:
            item: 输入数据项

        Returns:
            SHA256 哈希键
        """
        if self._legacy_key:
            key_payload = {
                "model_path": self._model_path,
                "data_item": item.model_dump(),
            }
        else:
            key_payload = {
                "model_id": self._cache_model_id,
                "messages": [msg.model_dump() for msg in item.messages],
            }
        return dict_to_hash(key_payload)

    async def embed_single(self, item: EmbeddingInputItem) -> EmbeddingResult | None:
        """单条异步 embed，含缓存逻辑.

        整个函数受 semaphore 控制：
          async with self._semaphore:
            1. 构建 cache key
            2. await cache.load_cache(key)
            3. 若命中 → 直接返回 EmbeddingResult
            4. 若未命中 → 调用 OpenAIEmbed._embed_single(item, dummy_semaphore)
            5. await cache.save_cache(key, result)
            6. 返回 EmbeddingResult

        Args:
            item: 输入数据项

        Returns:
            嵌入结果，失败时返回 None
        """
        async with self._semaphore:
            # 1. 构建缓存键
            cache_key = self._build_cache_key(item)

            # 2. 查缓存
            try:
                cached = await self._cache.load_cache(cache_key)
            except Exception as e:
                logger.warning(f"缓存查询异常: {type(e).__name__}: {e}")
                cached = None

            if cached is not None:
                # 缓存命中，直接返回
                return EmbeddingResult(
                    embedding=cached["embedding"],
                    data_item=item,
                    meta=cached.get("meta", item.meta),
                )

            # 3. 未命中 → 调用 OpenAI API
            result = await self._embedder._embed_single(item, self._dummy_semaphore)

            if result is None:
                return None

            # 4. 写缓存
            cache_value = {
                "embedding": result.embedding,
                "meta": result.meta,
            }
            try:
                await self._cache.save_cache(cache_key, cache_value)
            except Exception as e:
                logger.warning(f"缓存写入异常: {type(e).__name__}: {e}")

            return result

    async def embed_all(
        self,
        dataset: list[EmbeddingInputItem],
        desc: str = "Embedding",
    ) -> list[EmbeddingResult | None]:
        """并发计算所有 item 的 embedding.

        为每个 item 创建 embed_single task，然后 gather。
        并发由 semaphore 控制，这里无需额外限制。

        Args:
            dataset: 待嵌入的数据项列表
            desc: tqdm 进度条描述

        Returns:
            嵌入结果列表
        """
        from tqdm.asyncio import tqdm

        tasks = [self.embed_single(item) for item in dataset]
        return await tqdm.gather(*tasks, desc=desc)
