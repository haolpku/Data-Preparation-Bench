from __future__ import annotations

from typing import TYPE_CHECKING, override

from vllm.inputs.data import TokensPrompt

from data_preparation_bench.embed.base import BaseEmbed
from data_preparation_bench.embed.types import EmbeddingInputItem, EmbeddingResult
from data_preparation_bench.utils import logger
from data_preparation_bench.utils.timing import timing_context

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.transformers_utils.tokenizer import AnyTokenizer


class VllmEmbed(BaseEmbed):
    def __init__(
        self,
        model_name: str,
        max_num_seqs: int = 128,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        truncate_max_length: int = 40960,
    ) -> None:
        logger.info(f"创建 VllmEmbed，模型: {model_name}")
        # Store config for lazy initialization
        self._model_name = model_name
        self._max_num_seqs = max_num_seqs
        self._gpu_memory_utilization = gpu_memory_utilization
        self._tensor_parallel_size = tensor_parallel_size
        self._pipeline_parallel_size = pipeline_parallel_size
        self._truncate_max_length = truncate_max_length
        self._model: LLM | None = None
        self._tokenizer: AnyTokenizer | None = None
        super().__init__(model_name)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of vLLM model - only called when embed is actually needed."""
        if self._model is not None:
            return

        logger.info(f"开始初始化 vLLM 模型: {self._model_name}")
        logger.debug(
            f"配置参数: max_num_seqs={self._max_num_seqs}, gpu_memory_utilization={self._gpu_memory_utilization}, tensor_parallel_size={self._tensor_parallel_size}, pipeline_parallel_size={self._pipeline_parallel_size}"
        )

        with timing_context("模型初始化"):
            from vllm import LLM

            self._model = LLM(
                model=self._model_name,
                task="embed",
                enforce_eager=True,
                gpu_memory_utilization=self._gpu_memory_utilization,
                tensor_parallel_size=self._tensor_parallel_size,
                pipeline_parallel_size=self._pipeline_parallel_size,
                max_num_seqs=self._max_num_seqs,
            )
            self._tokenizer = self._model.get_tokenizer()
        logger.info(f"vLLM 模型初始化完成: {self._model_name}")

    @property
    def model(self) -> LLM:
        self._ensure_initialized()
        assert self._model, "model初始化后仍为None"
        return self._model

    @property
    def tokenizer(self) -> AnyTokenizer:
        self._ensure_initialized()
        assert self._tokenizer, "tokenizer初始化后仍为None"
        return self._tokenizer

    @override
    def embed(self, dataset: list[EmbeddingInputItem]) -> list[EmbeddingResult]:
        """异步执行嵌入计算.

        Args:
            dataset: 待嵌入的数据项列表

        Returns:
            嵌入结果列表
        """
        logger.info(f"开始嵌入计算，数据量: {len(dataset)}")

        # Ensure model is initialized before use
        self._ensure_initialized()

        messages_list = [
            [{"role": msg.role, "content": msg.content} for msg in item.messages]
            for item in dataset
        ]
        logger.debug(f"应用 chat template...")
        token_ids_list = self.tokenizer.apply_chat_template(
            messages_list,  # type: ignore
            tokenize=True,
            padding=False,
            truncation=True,
            max_length=self._truncate_max_length,
        )
        if not isinstance(token_ids_list, list) or any(
            not isinstance(token_ids, list) for token_ids in token_ids_list
        ):
            logger.error("Tokenizer 返回格式错误")
            raise ValueError("Tokenizer did not return list[list[int]].")

        converted_input = [
            TokensPrompt(prompt_token_ids=token_ids)
            for token_ids in token_ids_list
            if isinstance(token_ids, list)
        ]

        logger.info("开始模型推理...")
        with timing_context("模型推理"):
            # vLLM 的 embed 是同步方法，使用 to_thread 在后台线程执行
            outputs = self.model.embed(converted_input, use_tqdm=True)
        logger.info(f"嵌入计算完成，输出 {len(outputs)} 条结果")
        return [
            EmbeddingResult(
                embedding=output.outputs.embedding,
                data_item=item,
                meta=item.meta,
            )
            for output, item in zip(outputs, dataset)
        ]
