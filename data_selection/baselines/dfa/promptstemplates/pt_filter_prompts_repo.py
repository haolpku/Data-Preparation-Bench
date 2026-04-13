# class FilterPipelineWriter:
#     system_prompt_for_write_filter_pipeline = "You are an expert AI specialized in generating data filtering pipeline code."

#     task_prompt_for_write_filter_pipeline = """
# [ROLE] You are an expert AI specialized in generating data filtering pipeline code.
# [TASK] Please refer to the example operators source code and their module position, then write a new pipeline based on the description of target.
# You must select the operators that can be written to the file.

# [INPUT FORMAT] The input includes:、
# Don't use DebertaV3Filter operator.
# - example operators: {example} 
# - example_data which will be processed: {example_data}
# - target description: {target}.

# [OUTPUT FORMAT] The JSON structure is as follows:
# {{
#   "code": "Complete source code of the pipeline",
#   "desc": "Description of the pipeline's function and its input/output"
# }}

# [RULES]
# 1. Carefully read and understand the structure and style of the example operators and the module position.
# 2. Write pipeline code that meets the minimum requirements for complete executable according to the functionality described in {target}, without any extra code or comments.
# 3. Output in JSON format containing two fields: 'code' (the complete source code string of the pipeline) and 'desc' (a concise explanation of what the pipeline does and its input/output).
# """

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
You are a Senior Python Integration Engineer. Your mission is to wrap a logical pipeline class into a robust, runnable entry script for a Data-Centric AI platform.
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

[TASK: GENERATE COMPLETE RUNNABLE ENTRY SCRIPT]

You must strictly follow these engineering requirements to generate the code:

### 1. MANDATORY HEADER INJECTION
The script MUST start with this registry initialization to ensure all operators are loaded:
```python
import os
from dataflow.utils.registry import OPERATOR_REGISTRY
if hasattr(OPERATOR_REGISTRY, "_get_all"):
    OPERATOR_REGISTRY._get_all()

from dataflow.utils.storage import FileStorage
# Explicitly import all Operator classes used in pipeline_code from their modules
```

### 2. LLM SERVING INITIALIZATION (CONDITIONAL)
For EACH operator in `pipeline_code` that accepts `llm_serving` in its `__init__`:
- You MUST initialize the serving object:
```python
from dataflow.serving import APILLMServing_request
os.environ["API_KEY"] = "{api_key}"
llm_serving = APILLMServing_request(
    api_url="{api_url}",
    key_name_of_api_key="API_KEY"
)
```
- Pass this `llm_serving` instance when instantiating that operator.

### 3. STORAGE & DATAFLOW PROTOCOL
- **Storage Setup**: Use ONLY `FileStorage` with these exact parameters:
  `storage = FileStorage(first_entry_file_name="{test_data_path}", cache_path="{output_root}/", file_name_prefix="dataflow_cache_step", cache_type="jsonl")`
- **The .step() Requirement**: When calling `operator.run()`, the first argument MUST be `storage.step()`. 
  *Reason: This prevents overwriting the same cache file and ensures a traceable data lineage.*

### 4. INDEPENDENT EXECUTION LOGIC
- **NO DATA CHAINING**: An operator's return value is a storage key, NOT the data itself. NEVER pass `output1` as an `input_key` to `op2`.
- **KEY ALIGNMENT**: Select `input_key` ONLY from `{available_keys}`. Check the operator's `run()` signature in `{reference_operators}` to match the correct parameter names.
- **Example Pattern**:
  ```python
  # Correct way to run independent operators
  res1 = op1.run(storage.step(), input_key='text_column')
  res2 = op2.run(storage.step(), input_key='text_column') 
  ```
[STRICT CONSTRAINTS]
- DO NOT rewrite the operator class.
- DO NOT use `storage.read()` or `storage.write()`.

# [OUTPUT]
# Return ONLY a JSON object with a single key:
# {"code": "<complete runnable source code>"}
"""

# class FilterPipelineInstantiate:
#     system_prompt_for_filter_llm_instantiate = """
# [ROLE]
# You are a data pipeline code integration assistant that generates runnable pipeline code using EXISTING operator classes.
# """

#     task_prompt_for_filter_llm_instantiate = """
# [INPUTS]
# - target: {target}
# - reference_operators: {reference_operators}
# - pipeline_code: {pipeline_code}  # The DataFilteringPipeline class definition - USE THIS EXACT CLASS
# - example_data: {example_data}
# - available_keys: {available_keys}
# - preselected_input_key: {preselected_input_key}
# - test_data_path: {test_data_path}

# [TASK: GENERATE COMPLETE RUNNABLE CODE]

# You must generate code that follows these rules EXACTLY:

# 1. **USE THE PROVIDED PIPELINE CLASS**
#    - Paste the EXACT DataFilteringPipeline class from pipeline_code verbatim
#    - DO NOT modify or create a different pipeline class

# 2. **CORRECT IMPORTS**
#    ```python
#    from dataflow.utils.registry import OPERATOR_REGISTRY
#    if hasattr(OPERATOR_REGISTRY, "_get_all"):
#        OPERATOR_REGISTRY._get_all()
   
#    from dataflow.utils.storage import FileStorage
#    # Import ALL operators used in the pipeline_code
#    # Check pipeline_code to see which operators need to be imported
#    ```

# 3. **STORAGE SETUP**
#    ```python
#    storage = FileStorage(
#        first_entry_file_name={test_data_path},
#        cache_path="{output_root}/",
#        file_name_prefix="dataflow_cache_step",
#        cache_type="jsonl"
#    )
#    ```
#    Don't use DataFlowStorage or create your own storage class - use the real FileStorage as shown above, with the correct parameters.

# 4. **OPERATOR PARAMETER HANDLING - CRITICAL**
#    For EACH operator.run() call in the pipeline:
#    - **Check the operator's run() method signature** from its imported class
#    - ONLY pass parameters that exist in that signature
#    - **OPERATORS ARE INDEPENDENT** - NEVER use one operator's return value as input to another operator
#    - For key parameters (ending with "_key"):
#      * Use values ONLY from available_keys (the original dataset column names)
#      * NEVER use another operator's output key as input
#      * If a key parameter's name doesn't match any available_keys, set it to None
#    - If you use operators, you must use a statement like "storage = storage.step()" every time you call the "run()" function, so that the data set output by each operator is a different file.
#    - Each operator.run() returns a storage key, NOT the actual data - DO NOT chain them

# 5. **Additional LLM SERVING INITIALIZATION**
#     -For EACH operator that has llm_serving parameter in its init:
#     -Pass the initialized llm_serving object to the operator
#     -If imports are missing, add: from dataflow.serving import APILLMServing_request.
#     os.environ["API_KEY"] = "{api_key}"
#     -llm_serving = APILLMServing_request(
#         api_url="{api_url}",
#         key_name_of_api_key="API_KEY"
#     )
#     -Example: self.operator = OperatorClass(llm_serving=llm_serving, ...)
    
# 6. **COMMON MISTAKES TO AVOID**
#    ❌ WRONG - DO NOT DO THIS:
#       output1 = op1.run(storage.step(), input_key='instruction')
#       output2 = op2.run(storage.step(), input_key=output1)  # NEVER chain operators
   
#    ✅ CORRECT:
#       output1 = op1.run(storage.step(), input_key='instruction')
#       output2 = op2.run(storage.step(), input_key='instruction')  # Use original key
#       # Each operator runs independently on the same original data

# 6. **KEY SELECTION & OUTPUT**
#    - Select input_key from available_keys (prefer preselected_input_key if it exists)
#    - Print exactly: [selected_input_key] <the_key>
#    - Print output location after execution

# [STRICT CONSTRAINTS]
# - **DO NOT** create your own FileStorage or operator classes - import the real ones
# - **DO NOT** modify the pipeline_code class
# - **DO NOT** pass parameters that don't exist in an operator's run() signature
# - **DO NOT** call storage.read() or storage.write()
# - **DO NOT** use one operator's return value as input to another operator
# - **DO NOT** chain operators in any form
# - **ALWAYS** pass storage.step() (with parentheses) to run() methods
# - **ALWAYS** use available_keys for key parameters, or None if not available
# - **ALWAYS** treat each operator as independent - they all read from the original dataset

# [OUTPUT]
# Return ONLY a JSON object with a single key:
# {"code": "<complete runnable source code>"}
# """

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