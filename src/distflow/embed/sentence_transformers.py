from __future__ import annotations

from typing import TYPE_CHECKING, override

from distflow.data.types import MessageData
from distflow.embed.base import BaseEmbed
from distflow.embed.types import EmbeddingInputItem, EmbeddingResult
from distflow.utils import logger
from distflow.utils.timing import timing_context

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbed(BaseEmbed):
    """基于 sentence-transformers 的嵌入器实现.

    使用 sentence-transformers 库进行文本嵌入计算，支持批量处理和归一化。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
        prompt: str | None = None,
    ) -> None:
        """初始化 SentenceTransformersEmbed.

        Args:
            model_name: 模型名称或路径
            device: 运行设备，默认为 "cuda"
            batch_size: 批处理大小，默认为 32
            normalize_embeddings: 是否对嵌入向量进行归一化，默认为 True
            trust_remote_code: 是否信任远程代码，默认为 False
            prompt: 可选的前缀提示文本，会添加到每个输入文本前面
        """
        logger.info(f"创建 SentenceTransformersEmbed，模型: {model_name}")
        # 存储配置用于延迟初始化
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._trust_remote_code = trust_remote_code
        self._prompt = prompt
        self._model: SentenceTransformer | None = None
        super().__init__(model_name)

    def _ensure_initialized(self) -> None:
        """延迟初始化模型 - 仅在需要嵌入计算时才调用."""
        if self._model is not None:
            return

        logger.info(f"开始加载 Sentence Transformers 模型: {self._model_name}")
        logger.debug(
            f"配置参数: device={self._device}, batch_size={self._batch_size}, "
            f"normalize_embeddings={self._normalize_embeddings}"
        )

        with timing_context("模型加载"):
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
        logger.info(f"Sentence Transformers 模型加载完成: {self._model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """获取模型实例（确保已初始化）."""
        self._ensure_initialized()
        assert self._model, "模型初始化后仍为 None"
        return self._model

    def _format_messages(self, messages: list[MessageData]) -> str:
        """将消息列表格式化为单个文本字符串.

        Args:
            messages: 消息列表，每个消息包含 role 和 content

        Returns:
            格式化后的文本字符串
        """
        parts: list[str] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, dict):
                # 如果 content 是字典，尝试提取 text 字段，否则转为字符串
                content = content.get("text", str(content))
            parts.append(f"{msg.role}: {content}")
        return "\n".join(parts)

    def _prepare_texts(self, dataset: list[EmbeddingInputItem]) -> list[str]:
        """准备输入文本列表.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            格式化后的文本列表
        """
        texts: list[str] = []
        for item in dataset:
            text = self._format_messages(item.messages)
            if self._prompt:
                text = self._prompt + text
            texts.append(text)
        return texts

    @override
    def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult | None]:
        """执行嵌入计算.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
        logger.info(f"开始嵌入计算，数据量: {len(dataset)}")

        # 确保模型已初始化
        self._ensure_initialized()

        # 准备输入文本
        logger.debug("准备输入文本...")
        texts = self._prepare_texts(dataset)

        # 执行嵌入计算
        logger.info("开始模型推理...")
        with timing_context("模型推理"):
            embeddings = self.model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=True,
            )
        logger.info(f"嵌入计算完成，输出 {len(embeddings)} 条结果")

        # 构建结果列表
        return [
            EmbeddingResult(
                embedding=embedding.tolist(),
                data_item=item,
                meta=item.meta,
            )
            for embedding, item in zip(embeddings, dataset)
        ]
