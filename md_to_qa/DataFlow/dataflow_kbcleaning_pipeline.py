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

  
        self.storage = FileStorage(
            first_entry_file_name=input_jsonl,
            cache_path=cache_dir,
            file_name_prefix="knowledge_cleaning_step",
            cache_type="json",
        )


        self.llm_serving = APILLMServing_request(
            api_url="http://localhost:8000/v1/chat/completions",
            model_name="deepseek-v3.2",
            max_workers=50,
        )


        self.chunker = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

        self.cleaner = KBCTextCleaner(llm_serving=self.llm_serving, lang="en")

        self.qa_generator = Text2MultiHopQAGenerator(
            llm_serving=self.llm_serving, lang="en", num_q=5
        )

    def forward(self):

        self.chunker.run(
            storage=self.storage.step(),
            input_key="md_path", 
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


        final_cache = self.storage._get_cache_file_path(self.storage.operator_step)
        print(f"Pipeline finished. Final QA data saved to: {final_cache}")


if __name__ == "__main__":

    pipeline = KBCleaningMDPipeline(input_jsonl="./md_files.jsonl")
    pipeline.forward()
