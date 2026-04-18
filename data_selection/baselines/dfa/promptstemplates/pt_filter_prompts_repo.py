class FilterPipelineWriter:
    system_prompt_for_write_filter_pipeline = "You are a senior data architect specializing in Data-Centric AI pipelines."

    task_prompt_for_write_filter_pipeline = """
[CONTEXT]
You need to design a 'DataFilteringPipeline' class that orchestrates multiple pre-existing data operators.

[INPUTS]
1. **Available Operators**: {example} (Read their `__init__` and `run` signatures carefully).
2. **Data Sample**: {example_data} (Understand the schema/keys available).
3. **Filtering Goal**: {target}.

[CONSTRAINTS]
- **Prohibited**: DO NOT use 'DebertaV3Filter'.
- **Independence**: Operators must run independently. The output of one operator is NEVER the input of another.
- **Verification**: Ensure the parameters passed to `.run()` exist in the operator's source code provided.
- **Key Selection**: Use keys found in 'example_data'.

[OUTPUT JSON FORMAT]
{{
  "code": "class DataFilteringPipeline:\\n    def __init__(self, ...):\\n        # Initialize operators\\n    def run(self, storage, ...):\\n        # Sequential calls to operators",
  "desc": "Logic explanation"
}}

[TASK]
Generate a clean, executable Python class `DataFilteringPipeline`.
- In `__init__`: Instantiate the selected operators.
- In `run`: Call each operator's `.run(storage.step(), ...)` method.
"""

class FilterPipelineInstantiate:
    system_prompt_for_filter_llm_instantiate = """
[ROLE]
You are a Senior Python Integration Engineer. Your mission is to take an existing logical pipeline class and wrap it into a robust, runnable Python script. You must maintain the class-based structure while adding the necessary initialization and entry point code.
"""

    task_prompt_for_filter_llm_instantiate = """
[INPUT DATA]
- target: {target}
- reference_operators: {reference_operators} (Source code of available operators)
- pipeline_code: {pipeline_code} (The existing DataFilteringPipeline class definition)
- example_data: {example_data} (Schema of the dataset)
- available_keys: {available_keys} (List of columns in the dataset)
- preselected_input_key: {preselected_input_key}
- test_data_path: {test_data_path} (Input JSONL file path)
- output_root: {output_root}
- api_url: {api_url}
- api_key: {api_key}

[TASK: GENERATE COMPLETE RUNNABLE SCRIPT WITH CLASS RETAINED]

You must generate a complete Python script that preserves the `DataFilteringPipeline` class from `pipeline_code` and adds an executable entry block. Follow these requirements:

### 1. MANDATORY HEADER & IMPORTS
- The script MUST start with the registry initialization:
```python
import os
from dataflow.utils.registry import OPERATOR_REGISTRY
if hasattr(OPERATOR_REGISTRY, "_get_all"):
    OPERATOR_REGISTRY._get_all()

from dataflow.utils.storage import FileStorage
# Explicitly import all Operator classes used in pipeline_code
````

### 2\. CLASS PRESERVATION & LLM INJECTION

  - **RETAIN THE CLASS**: Keep the `DataFilteringPipeline` class structure.
  - **LLM Serving**: If any operator in the class requires `llm_serving`, initialize the `APILLMServing_request` in the `__main__` block and pass it during the class instantiation or operator setup.

<!-- end list -->

```python
from dataflow.serving import APILLMServing_request
os.environ["API_KEY"] = "{api_key}"
llm_serving = APILLMServing_request(api_url="{api_url}", key_name_of_api_key="API_KEY")
```

### 3\. STORAGE & LINEAGE PROTOCOL (INSIDE CLASS METHODS)

  - **Class Integration**: The class methods (e.g., `run`) must be designed to accept a `storage` object.
  - **The .step() Requirement**: Inside the class methods, every time an operator's `.run()` is called, the first argument MUST be `storage.step()`. This ensures traceable data lineage for each filtering step.
  - **Independent Execution**: Within the class logic, ensure operators run independently on the `{available_keys}`. NEVER chain the output key of one operator as the input of another.

### 4\. ENTRY POINT (THE IF-MAIN BLOCK)

  - Add a standard `if __name__ == "__main__":` block at the end of the script.
  - **Inside this block**:
    1.  Initialize `FileStorage` with: `first_entry_file_name="{test_data_path}"`, `cache_path="{output_root}/"`, `file_name_prefix="dataflow_cache_step"`, `cache_type="jsonl"`.
    2.  Instantiate the `DataFilteringPipeline` class.
    3.  Execute the pipeline by calling its primary method, passing the initialized `storage`.
    4.  Print the final execution results.

[STRICT CONSTRAINTS]

  - DO NOT delete or flatten the `DataFilteringPipeline` class into a top-level script.
  - DO NOT use `storage.read()` or `storage.write()`.
  - Ensure all input keys match `{available_keys}`.

# [OUTPUT]
# Return ONLY a JSON object with a single key:
# {{"code": "\<complete source code including class and if-main block\>"}}

"""

class FilterDebugPipeline:
    system_prompt_for_filter_code_debugging = """
You are a senior DataFlow pipeline debugging assistant.
Your job is to read pipeline code and its runtime logs or traceback,
locate the root-cause, and propose an actionable fix.
Always think step-by-step before you answer.
""" 
    task_prompt_for_filter_code_debugging = """
[INPUT]
① Pipeline code (read-only):
{pipeline_code}
② Execution error message:
{exec_error}
③ Stack trace:
{stack_trace}

[OUTPUT RULES]
Reply only with a valid JSON object, no markdown, no comments.
1 The JSON must and can only contain one top-level key:
"reason": In natural language, explain in detail the root cause of the error and provide specific, actionable suggestions for a fix. Your answer must include error analysis, a detailed reasoning process, and concrete solutions, clearly indicating which code needs to be modified or added.

2 All JSON keys and string values must be double-quoted, with no trailing commas.
3 If you are unsure about any value, use an empty string.
4 Double-check that your response is a valid JSON. Do not output anything else.

"""

class FilterCodeRewriter:
    system_prompt_for_filter_code_rewriting = """
You are a Python code expert.
"""
    task_prompt_for_filter_code_pipe_rewriting = """
    [INPUT]

The input consists of:
1. Pipeline code (read-only):
{pipeline_code}
2. Error trace / shell output:
[1] {stack_trace}

3. Debug analysis and suggestions from the previous step:
{debug_reason}

4. Sample data [For the first operator in 'run', the key (for example, is one of the keys in the sampled data), you need to determine it yourself]:
{data_sample}

5. Target description:
{target}

[OUTPUT RULES]
1.Reply only with a valid JSON object, no markdown, no comments.
2.For the pipeline, the output_key of the previous operator and the input_key of the next operator must be filled in correctly and must match the data flow. Modify them logically as needed；
3.The JSON must and can only contain one top-level key:
{"code": Return the modified and corrected version of the code based on the analysis, as a string.}
4.请根据Error trace, Debug analysis and suggestions修改代码；
All JSON keys and string values must be double-quoted, with no trailing commas.
If you are unsure about any value, use an empty string.
Double-check that your response is a valid JSON. Do not output anything else.
    
    """