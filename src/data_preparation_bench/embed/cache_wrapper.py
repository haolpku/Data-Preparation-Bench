import asyncio
import hashlib
import json
from typing import Any, cast

from redis.asyncio import Redis

from data_preparation_bench.embed.base import BaseEmbed
from data_preparation_bench.embed.types import EmbeddingInputItem, EmbeddingResult
from data_preparation_bench.utils import logger


def dict_to_hash(d: dict[Any, Any]) -> str:
    """生成字典的SHA256哈希摘要"""
    s = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()


class CachedEmbed(BaseEmbed):
    """使用 Redis 作为缓存后端的嵌入包装器.

    通过 Redis 客户端直接与 Redis 服务通信，实现分布式缓存。
    使用 semaphore 限制并发请求数量。
    """

    def __init__(
        self,
        embedder: BaseEmbed,
        redis_url: str = "redis://127.0.0.1:6379",
        max_concurrent_requests: int = 50,
        cache_model_id: str | None = None,
        legacy_key: bool = False,
        redis_db: int = 0,
    ) -> None:
        """初始化缓存嵌入器.

        Args:
            embedder: 底层嵌入器，用于计算未缓存的数据
            redis_url: Redis 连接 URL，例如 "redis://127.0.0.1:6379"
            max_concurrent_requests: 最大并发请求数
            cache_model_id: 用于缓存键的模型标识符，默认为模型路径。
                            可用于在移动模型后仍使用旧缓存。
            legacy_key: 是否使用旧版缓存键格式（包含完整 data_item），
                       默认为 False（使用新版：仅 model_id + messages）
            redis_db: Redis 数据库编号，默认为 0
        """
        self.embedder = embedder
        self.model_path = (
            getattr(embedder, "model_name", None)
            or getattr(embedder, "model_path", None)
            or "unknown"
        )
        # 用于缓存键的模型标识符
        self.cache_model_id = cache_model_id if cache_model_id else self.model_path
        self.legacy_key = legacy_key
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # 初始化 Redis 客户端
        self._redis: Redis | None = None
        self._redis_url = redis_url
        self._redis_db = redis_db

        super().__init__(self.model_path)

    def _get_redis(self) -> Redis:
        """获取或创建 Redis 客户端."""
        if self._redis is None:
            self._redis = Redis.from_url(
                self._redis_url,
                db=self._redis_db,
                decode_responses=True,
            )
        return self._redis

    def _build_cache_key(self, item: EmbeddingInputItem) -> str:
        """构建缓存键.

        Args:
            item: 输入数据项

        Returns:
            SHA256 哈希键
        """
        if self.legacy_key:
            # 旧版格式：包含完整 data_item（包含 messages 和 meta）
            key_payload = {
                "model_path": self.model_path,
                "data_item": item.model_dump(),
            }
        else:
            # 新版格式：仅使用 cache_model_id 和 messages（不含 meta）
            key_payload = {
                "model_id": self.cache_model_id,
                "messages": [msg.model_dump() for msg in item.messages],
            }
        return dict_to_hash(key_payload)

    async def _get_cached(self, key: str) -> dict[str, Any] | None:
        """从 Redis 获取单个缓存值（受 semaphore 限制并发）.

        Args:
            key: 缓存键

        Returns:
            缓存值字典，如果不存在则返回 None
        """
        async with self._semaphore:
            try:
                redis = self._get_redis()
                cached_data = await redis.get(key)
                if cached_data:
                    return json.loads(cached_data)
                return None
            except Exception as e:
                logger.warning(f"Redis 缓存查询失败: {e}")
                return None

    async def _set_cached(self, key: str, value: dict[str, Any]) -> bool:
        """设置单个缓存值到 Redis（受 semaphore 限制并发）.

        Args:
            key: 缓存键
            value: 缓存值

        Returns:
            是否成功
        """
        async with self._semaphore:
            try:
                redis = self._get_redis()
                serialized = json.dumps(value)
                await redis.set(key, serialized)
                return True
            except Exception as e:
                logger.warning(f"Redis 缓存写入失败: {e}")
                return False

    async def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult]:
        """异步执行嵌入计算，使用 Redis 缓存.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
        logger.info(f"开始缓存嵌入计算，数据量: {len(dataset)}")

        # 并发查询所有缓存（受 semaphore 限制并发量）
        cache_keys = [self._build_cache_key(item) for item in dataset]
        cache_tasks = [self._get_cached(key) for key in cache_keys]
        cached_values = await asyncio.gather(*cache_tasks, return_exceptions=True)

        # 分离缓存命中和未命中的项
        results: list[EmbeddingResult | None] = [None] * len(dataset)
        missing_items: list[EmbeddingInputItem] = []
        missing_indices: list[int] = []
        missing_keys: list[str] = []

        for idx, (item, key, cached_result) in enumerate(
            zip(dataset, cache_keys, cached_values)
        ):
            # 处理异常结果
            cached_item: dict[str, Any] | None
            if isinstance(cached_result, Exception):
                logger.debug(f"缓存查询异常，将重新计算: {cached_result}")
                cached_item = None
            elif cached_result is None:
                cached_item = None
            else:
                # 使用 cast 帮助类型检查器理解此处不是 BaseException
                cached_item = cast(dict[str, Any], cached_result)

            if cached_item is not None:
                results[idx] = EmbeddingResult(
                    embedding=cached_item["embedding"],
                    data_item=item,
                    meta=cached_item.get("meta", item.meta),
                )
                logger.debug(f"缓存命中: {key[:16]}...")
            else:
                missing_items.append(item)
                missing_indices.append(idx)
                missing_keys.append(key)

        logger.info(f"缓存命中: {len(dataset) - len(missing_items)}/{len(dataset)}")

        # 计算未缓存的嵌入
        if missing_items:
            new_results = await self.embedder.embed(missing_items)

            # 并发写入缓存（受 semaphore 限制并发量）
            write_tasks = []
            for key, idx, result in zip(missing_keys, missing_indices, new_results):
                cache_value = {
                    "embedding": result.embedding,
                    "meta": result.meta,
                }
                write_tasks.append(self._set_cached(key, cache_value))
                results[idx] = EmbeddingResult(
                    embedding=result.embedding,
                    data_item=dataset[idx],
                    meta=result.meta,
                )

            # 等待所有写入完成
            write_results = await asyncio.gather(*write_tasks, return_exceptions=True)
            success_count = sum(1 for r in write_results if r is True)
            logger.info(f"缓存写入完成: {success_count}/{len(write_tasks)} 成功")

        logger.info(f"嵌入计算完成，共 {len(results)} 条结果")
        return [result for result in results if result is not None]

    async def close(self) -> None:
        """关闭 Redis 连接."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis 连接已关闭")

    async def __aenter__(self) -> "CachedEmbed":
        """异步上下文管理器入口."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器退出."""
        await self.close()
