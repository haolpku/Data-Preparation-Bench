class FilterPipelineWriter:
    system_prompt_for_write_filter_pipeline = "You are an expert AI specialized in generating data filtering pipeline code."

    task_prompt_for_write_filter_pipeline = """
[ROLE] You are an expert AI specialized in generating data filtering pipeline code.
[TASK] Please refer to the example operators source code and their module position, then write a new pipeline based on the description of target.

[INPUT FORMAT] The input includes:
- example operators: {example}
- target description: {target}.

[OUTPUT FORMAT] The JSON structure is as follows:
{{
  "code": "Complete source code of the pipeline",
  "desc": "Description of the pipeline's function and its input/output"
}}

[RULES]
1. Carefully read and understand the structure and style of the example operators and the module position.
2. Write pipeline code that meets the minimum requirements for complete executable according to the functionality described in {target}, without any extra code or comments.
3. Output in JSON format containing two fields: 'code' (the complete source code string of the pipeline) and 'desc' (a concise explanation of what the pipeline does and its input/output).
"""

class FilterPipelineInstantiate:
    system_prompt_for_filter_llm_instantiate = """
[ROLE]
You are a data pipeline code integration assistant that generates runnable pipeline code using EXISTING operator classes.
"""

    task_prompt_for_filter_llm_instantiate = """
[INPUTS]
- target: {target}
- reference_operators: {reference_operators}
- pipeline_code: {pipeline_code}  # The DataFilteringPipeline class definition - USE THIS EXACT CLASS
- example_data: {example_data}
- available_keys: {available_keys}
- preselected_input_key: {preselected_input_key}
- test_data_path: {test_data_path}

[TASK: GENERATE COMPLETE RUNNABLE CODE]

You must generate code that follows these rules EXACTLY:

1. **USE THE PROVIDED PIPELINE CLASS**
   - Paste the EXACT DataFilteringPipeline class from pipeline_code verbatim
   - DO NOT modify or create a different pipeline class

2. **CORRECT IMPORTS**
   ```python
   from dataflow.utils.registry import OPERATOR_REGISTRY
   if hasattr(OPERATOR_REGISTRY, "_get_all"):
       OPERATOR_REGISTRY._get_all()
   
   from dataflow.utils.storage import FileStorage
   # Import ALL operators used in the pipeline_code
   # Check pipeline_code to see which operators need to be imported
   ```

3. **STORAGE SETUP**
   ```python
   storage = FileStorage(
       first_entry_file_name={test_data_path},
       cache_path="{output_root}/",
       file_name_prefix="dataflow_cache_step",
       cache_type="jsonl"
   )
   ```
   Don't use DataFlowStorage or create your own storage class - use the real FileStorage as shown above, with the correct parameters.

4. **OPERATOR PARAMETER HANDLING - CRITICAL**
   For EACH operator.run() call in the pipeline:
   - **Check the operator's run() method signature** from its imported class
   - ONLY pass parameters that exist in that signature
   - **OPERATORS ARE INDEPENDENT** - NEVER use one operator's return value as input to another operator
   - For key parameters (ending with "_key"):
     * Use values ONLY from available_keys (the original dataset column names)
     * NEVER use another operator's output key as input
     * If a key parameter's name doesn't match any available_keys, set it to None
   - ALWAYS use storage=storage.step() to EVERY run() call
   - Each operator.run() returns a storage key, NOT the actual data - DO NOT chain them

5. **Additional LLM SERVING INITIALIZATION**
    -For EACH operator that has llm_serving parameter in its init:
    -Pass the initialized llm_serving object to the operator
    -If imports are missing, add: from dataflow.serving import APILLMServing_request.
    -Example: self.operator = OperatorClass(llm_serving=llm_serving, ...)
    
6. **COMMON MISTAKES TO AVOID**
   ❌ WRONG - DO NOT DO THIS:
      output1 = op1.run(storage.step(), input_key='instruction')
      output2 = op2.run(storage.step(), input_key=output1)  # NEVER chain operators
   
   ✅ CORRECT:
      output1 = op1.run(storage.step(), input_key='instruction')
      output2 = op2.run(storage.step(), input_key='instruction')  # Use original key
      # Each operator runs independently on the same original data

6. **KEY SELECTION & OUTPUT**
   - Select input_key from available_keys (prefer preselected_input_key if it exists)
   - Print exactly: [selected_input_key] <the_key>
   - Print output location after execution

[STRICT CONSTRAINTS]
- **DO NOT** create your own FileStorage or operator classes - import the real ones
- **DO NOT** modify the pipeline_code class
- **DO NOT** pass parameters that don't exist in an operator's run() signature
- **DO NOT** call storage.read() or storage.write()
- **DO NOT** use one operator's return value as input to another operator
- **DO NOT** chain operators in any form
- **ALWAYS** pass storage.step() (with parentheses) to run() methods
- **ALWAYS** use available_keys for key parameters, or None if not available
- **ALWAYS** treat each operator as independent - they all read from the original dataset

[OUTPUT]
Return ONLY a JSON object with a single key:
{"code": "<complete runnable source code>"}
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
② Error trace / shell output:
{error_trace}
③ Execution error message:
{exec_error}
④ Stack trace:
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