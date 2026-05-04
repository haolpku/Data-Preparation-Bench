# Data Construction Skill

Data Construction Skill is an agent skill designed to help generate supervision datasets from markdown books or long markdown documents. It extracts reusable domain knowledge from unstructured text and structures it into three main forms of QA supervision:

1. **Concept QA**: Focuses on teaching atomic knowledge such as definitions, categories, and rules.
2. **Process QA**: Teaches grounded reasoning patterns like rule application, causal explanation, and exception handling.
3. **Case Application**: Transfers knowledge into real-world scenarios where the model applies book knowledge to solve practical situations.

## Key Features:

* **Comprehensive Dataset Generation**: Creates supervision datasets that capture all knowledge propositions across the document, ensuring that no critical information is missed.
* **Batch Processing**: Supports batch processing with full coverage, enabling scalable data generation from large markdown sources.
* **Reusability and Flexibility**: Can work with raw markdown files or precomputed chunk files, providing flexibility in input data handling.
* **Robust Coverage Auditing**: Features automatic validation and coverage checking to ensure the generated dataset is complete and of high quality.

## Outputs:

* `supervision_batches`: Contains the generated supervision records in batch files.
* `chunk_status.jsonl`: Tracks the processing status of each chunk, whether it was "kept" or "skipped."
* `supervision_merged.jsonl`: Merged supervision data for further processing or model training.
* `validation.json` and `coverage.json`: Contain validation and coverage reports for auditing the generated dataset's quality.

## Usage:

Data Construction Skill is ideal for agents that need to convert unstructured markdown content into structured training data. It focuses on full coverage and grounded reasoning, ensuring each chunk of text is processed with detailed status and sample synchronization. This skill is perfect for creating book-to-SFT pipelines where traditional answer-only QA is insufficient and the dataset needs to teach rule application and reasoning patterns across various scenarios.
