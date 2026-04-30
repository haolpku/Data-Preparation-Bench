from dataflow.operators.core_text import Text2MultiHopQAGenerator
from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    KBCTextCleaner,
)
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


class KBCleaningMDPipeline:
    def __init__(
        self, input_jsonl: str = "./md_files.jsonl", cache_dir: str = "./.cache/md_qa"
    ):
        """
        :param input_jsonl: JSONL 文件，每行格式 {"md_path": "/path/to/file.md"}
        :param cache_dir: 缓存目录，存储中间结果
        """
        # 存储管理器，从输入 JSONL 读取数据
        self.storage = FileStorage(
            first_entry_file_name=input_jsonl,
            cache_path=cache_dir,
            file_name_prefix="knowledge_cleaning_step",
            cache_type="json",
        )

        # LLM 服务（可根据需要调整 API 地址、模型、并发）
        self.llm_serving = APILLMServing_request(
            api_url="http://localhost:8000/v1/chat/completions",
            model_name="deepseek-v3.2",
            max_workers=50,
        )

        # 步骤1：文本分块（直接处理 MD 文件）
        self.chunker = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

        # 步骤2：LLM 文本清洗
        self.cleaner = KBCTextCleaner(llm_serving=self.llm_serving, lang="en")

        # 步骤3：多跳 QA 生成
        self.qa_generator = Text2MultiHopQAGenerator(
            llm_serving=self.llm_serving, lang="en", num_q=5
        )

    def forward(self):
        # 注意：第一步是分块，输入字段必须是输入 JSONL 中存在的字段名（默认为 md_path）
        self.chunker.run(
            storage=self.storage.step(),
            input_key="md_path",  # 与输入 JSONL 的键名一致
            output_key="raw_chunk",
        )

        self.cleaner.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="cleaned_chunk",
        )

        self.qa_generator.run(
            storage=self.storage.step(),
            input_key="cleaned_chunk",
            output_key="QA_pairs",
            output_meta_key="QA_metadata",
        )

        # 可选：打印最终输出文件位置
        final_cache = self.storage._get_cache_file_path(self.storage.operator_step)
        print(f"Pipeline finished. Final QA data saved to: {final_cache}")


if __name__ == "__main__":
    # 请确保输入 JSONL 文件存在且格式正确
    pipeline = KBCleaningMDPipeline(input_jsonl="./md_files.jsonl")
    pipeline.forward()
