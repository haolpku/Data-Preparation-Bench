import asyncio
import json
from typing import Any

from redis.asyncio import Redis

from data_preparation_bench.utils import logger


class RedisCache:
    """使用 Redis 作为缓存后端的实现.

    通过 Redis 客户端直接与 Redis 服务通信，实现分布式缓存。
    使用 semaphore 限制并发请求数量。
    """

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379",
        max_concurrent_requests: int = 50,
        redis_db: int = 0,
    ) -> None:
        """初始化Redis缓存.

        Args:
            redis_url: Redis 连接 URL，例如 "redis://127.0.0.1:6379"
            max_concurrent_requests: 最大并发请求数
            redis_db: Redis 数据库编号，默认为 0
        """
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # 初始化 Redis 客户端
        self._redis: Redis | None = None
        self._redis_url = redis_url
        self._redis_db = redis_db

    def _get_redis(self) -> Redis:
        """获取或创建 Redis 客户端."""
        if self._redis is None:
            self._redis = Redis.from_url(
                self._redis_url,
                db=self._redis_db,
                decode_responses=True,
            )
        return self._redis

    async def load_cache(self, cache_key: str) -> dict[str, Any] | None:
        """从 Redis 获取单个缓存值（受 semaphore 限制并发）.

        Args:
            cache_key: 缓存键

        Returns:
            缓存值字典，如果不存在则返回 None
        """
        async with self._semaphore:
            try:
                redis = self._get_redis()
                cached_data = await redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
                return None
            except Exception as e:
                logger.warning(f"Redis 缓存查询失败: {e}")
                return None

    async def save_cache(self, cache_key: str, cache_value: dict[str, Any]) -> bool:
        """设置单个缓存值到 Redis（受 semaphore 限制并发）.

        Args:
            cache_key: 缓存键
            cache_value: 缓存值

        Returns:
            是否成功
        """
        async with self._semaphore:
            try:
                redis = self._get_redis()
                serialized = json.dumps(cache_value)
                await redis.set(cache_key, serialized)
                return True
            except Exception as e:
                logger.warning(f"Redis 缓存写入失败: {e}")
                return False

    async def close(self) -> None:
        """关闭 Redis 连接."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis 连接已关闭")

    async def __aenter__(self) -> "RedisCache":
        """异步上下文管理器入口."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器退出."""
        await self.close()
