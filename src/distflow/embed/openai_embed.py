from __future__ import annotations

import asyncio
from typing import Any, override

from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from distflow.data.types import MessageData
from distflow.embed.base import BaseEmbed
from distflow.embed.types import EmbeddingInputItem, EmbeddingResult
from distflow.utils import logger
from distflow.utils.timing import timing_context


class OpenAIEmbed(BaseEmbed):
    """基于 vLLM OpenAI-Compatible Embeddings API 的嵌入器实现.

    使用 openai SDK 通过 HTTP API 调用远程 vLLM embedding 服务，
    直接传递 messages 列表让 server 端处理 chat template 和 truncate。
    内部使用异步实现，通过 semaphore 限制并发，使用 tqdm 展示进度。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent_requests: int = 50,
        # vLLM chat embedding 参数
        encoding_format: str | None = None,
        truncate_prompt_tokens: int | None = None,
        truncation_side: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        embed_dtype: str = "float32",
        endianness: str = "native",
        use_activation: bool | None = None,
        max_retries: int = 3,
    ) -> None:
        """初始化 OpenAIEmbed.

        Args:
            model_name: 模型名称
            api_key: API Key，默认从环境变量读取
            base_url: 自定义 API base URL
            max_concurrent_requests: 最大并发请求数
            encoding_format: 编码格式，如 "float" 或 "base64"
            truncate_prompt_tokens: 超过此 token 数则 truncate，-1 表示不限制
            truncation_side: "left" 或 "right"
            add_generation_prompt: 是否在末尾加 generation prompt
            continue_final_message: 是否让最后一条 message 保持开放
            add_special_tokens: 是否额外加 special token
            chat_template: 自定义 Jinja chat template
            chat_template_kwargs: 传给 chat template renderer 的额外参数
            embed_dtype: 输出 dtype，如 "float32"
            endianness: 字节序，如 "native"
            use_activation: 是否对 pooler 输出使用 activation
            max_retries: 单个请求最大重试次数
        """
        logger.info(f"创建 OpenAIEmbed，模型: {model_name}")
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._max_concurrent_requests = max_concurrent_requests

        # vLLM chat embedding 参数
        self._encoding_format = encoding_format
        self._truncate_prompt_tokens = truncate_prompt_tokens
        self._truncation_side = truncation_side
        self._add_generation_prompt = add_generation_prompt
        self._continue_final_message = continue_final_message
        self._add_special_tokens = add_special_tokens
        self._chat_template = chat_template
        self._chat_template_kwargs = chat_template_kwargs
        self._embed_dtype = embed_dtype
        self._endianness = endianness
        self._use_activation = use_activation
        self._max_retries = max_retries

        self._client: AsyncOpenAI | None = None

        super().__init__(model_name)

    def _ensure_initialized(self) -> None:
        """延迟初始化 OpenAI 异步客户端."""
        if self._client is not None:
            return

        logger.info("初始化 OpenAI 异步客户端")
        from openai import AsyncOpenAI

        with timing_context("OpenAI 客户端初始化"):
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        logger.info("OpenAI 异步客户端初始化完成")

    def _format_messages(self, messages: list[MessageData]) -> list[dict[str, Any]]:
        """将消息列表格式化为 API 可用的字典列表.

        Args:
            messages: 消息列表

        Returns:
            格式化后的消息字典列表
        """
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, dict):
                content = content.get("text", str(content))
            formatted.append({"role": msg.role, "content": content})
        return formatted

    def _build_request_body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """构造 vLLM chat embeddings 请求 body.

        Args:
            messages: 格式化后的消息列表

        Returns:
            请求 body 字典
        """
        body: dict[str, Any] = {
            "messages": messages,
            "model": self._model_name,
        }
        if self._encoding_format is not None:
            body["encoding_format"] = self._encoding_format
        if self._truncate_prompt_tokens is not None:
            body["truncate_prompt_tokens"] = self._truncate_prompt_tokens
        if self._truncation_side is not None:
            body["truncation_side"] = self._truncation_side
        if self._add_generation_prompt:
            body["add_generation_prompt"] = self._add_generation_prompt
        if self._continue_final_message:
            body["continue_final_message"] = self._continue_final_message
        if self._add_special_tokens:
            body["add_special_tokens"] = self._add_special_tokens
        if self._chat_template is not None:
            body["chat_template"] = self._chat_template
        if self._chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self._chat_template_kwargs
        if self._embed_dtype != "float32":
            body["embed_dtype"] = self._embed_dtype
        if self._endianness != "native":
            body["endianness"] = self._endianness
        if self._use_activation is not None:
            body["use_activation"] = self._use_activation
        return body

    @override
    def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult | None]:
        """执行嵌入计算.

        接口为同步方法，内部使用 asyncio.run 驱动异步逻辑。

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
        return asyncio.run(self._embed_async(dataset))

    async def _embed_async(
        self, dataset: list[EmbeddingInputItem]
    ) -> list[EmbeddingResult | None]:
        """异步执行嵌入计算.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
        logger.info(f"开始嵌入计算，数据量: {len(dataset)}")

        # 确保客户端已初始化
        self._ensure_initialized()

        # 在当前 event loop 中创建 semaphore
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)

        # 为每个 item 创建 embedding task
        logger.info("开始 API 调用...")
        tasks = [self._embed_single(item, semaphore) for item in dataset]

        # 使用 tqdm 异步 gather 收集结果
        from tqdm.asyncio import tqdm

        with timing_context("API 调用"):
            results = await tqdm.gather(*tasks, desc="Embedding")

        logger.info(f"嵌入计算完成，输出 {len(results)} 条结果")
        return results

    async def _embed_single(
        self, item: EmbeddingInputItem, semaphore: asyncio.Semaphore
    ) -> EmbeddingResult | None:
        """为单个数据项调用 vLLM chat embedding API.

        受 semaphore 限制并发，失败时自动重试。

        Args:
            item: 原始数据项

        Returns:
            嵌入结果
        """
        assert self._client is not None, "OpenAI 客户端未初始化"

        messages = self._format_messages(item.messages)
        body = self._build_request_body(messages)

        for attempt in range(self._max_retries):
            async with semaphore:
                try:
                    response: CreateEmbeddingResponse = await self._client.post(
                        "/embeddings",
                        cast_to=CreateEmbeddingResponse,
                        body=body,
                    )
                    embedding = response.data[0].embedding
                    return EmbeddingResult(
                        embedding=embedding,
                        data_item=item,
                        meta=item.meta,
                    )
                except Exception as e:
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"Embedding 请求失败 {attempt + 1} / {self._max_retries}: "
                            f"{type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(0.1 * (attempt + 1))
                    else:
                        logger.error(f"Embedding 请求最终失败: {type(e).__name__}: {e}")
                        return None
