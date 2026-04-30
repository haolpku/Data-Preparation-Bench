import os

import pandas as pd
from dataflow.operators.core_text import (
    FormatStrPromptedGenerator,
    GeneralFilter,
    Text2QAGenerator,
)
from dataflow.operators.knowledge_cleaning import KBCCompositeCleaningFlashOperator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


class MdToQAPipeline:
    def __init__(self):
        # Input JSONL format (one JSON object per line):
        # {"md_path": "/path/to/document.md"}
        self.storage = FileStorage(
            first_entry_file_name="test/md2qa.jsonl",
            cache_path="./cache_md_qa",
            file_name_prefix="md_qa",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://localhost:8000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=5,
        )

        # Step 1: Read .md files, chunk, and clean
        # (.md files skip MinerU — passed directly to chunking)
        self.kbc_cleaner = KBCCompositeCleaningFlashOperator(
            llm_serving=self.llm_serving,
            intermediate_dir="./kbc_intermediate/",
            mineru_model_path=None,
            chunk_size=512,
            chunk_overlap=50,
            lang="en",
        )

        # Step 2: Generate QA pairs from cleaned chunks
        self.qa_generator = Text2QAGenerator(
            llm_serving=self.llm_serving,
        )

        # Step 3: Score QA pairs (needs both instruction + response → multi-field)
        self.qa_scorer = FormatStrPromptedGenerator(
            self.llm_serving,
            system_prompt=(
                "Evaluate this QA pair on a scale of 1-5. "
                "Consider: question clarity, answer accuracy, answer completeness, "
                "and relevance to the source material. "
                "Output **only** a single integer from 1 to 5, "
                "without any other text, explanation, or punctuation. "
                "Example output: 4"
            ),
            prompt_template=FormatStrPrompt(
                f_str_template="Question: {instruction}\nAnswer: {response}"
            ),
        )

        # Step 4: Keep only high-quality pairs (score >= 4)
        self.qa_filter = GeneralFilter(
            [
                lambda df: pd.to_numeric(df["qa_score"], errors="coerce").fillna(0)
                >= 4,
            ]
        )

    def forward(self):
        self.kbc_cleaner.run(
            storage=self.storage,
            input_key="md_path",
            output_key="cleaned_chunk",
        )

        self.qa_generator.run(
            storage=self.storage.step(),
            input_key="cleaned_chunk",
            input_question_num=3,
            output_prompt_key="qa_prompt",
            output_quesion_key="instruction",
            output_answer_key="response",
        )

        self.qa_scorer.run(
            storage=self.storage.step(),
            output_key="qa_score",
            instruction="instruction",
            response="response",
        )

        self.qa_filter.run(
            storage=self.storage.step(),
        )

        # Export selected columns to final output
        final_cache = self.storage._get_cache_file_path(self.storage.operator_step)
        df = pd.read_json(final_cache, lines=True)
        output_path = "./output/qa_output.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df[["md_path", "instruction", "response"]].to_json(
            output_path, orient="records", lines=True, force_ascii=False
        )
        print(f"Output saved to {output_path} ({len(df)} QA pairs)")


if __name__ == "__main__":
    pipeline = MdToQAPipeline()
    pipeline.forward()
