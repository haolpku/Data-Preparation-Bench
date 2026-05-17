import asyncio
import argparse
import os
import sys
import json
import torch
import runpy
import traceback
from pathlib import Path
from dataflow_agent.state import MainState, MainRequest

from dataflow_agent.workflow import get_workflow
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from utils import (
    split_dataset, 
    create_simple_parallel_script, merge_jsonl_results,
)
from workflow import register_all_workflows

PROJDIR = get_project_root()
log = get_logger(__name__)

register_all_workflows()

async def run_workflow(name: str, state):
    factory = get_workflow(name)
    graph_builder = factory(state)

    graph = graph_builder.build()       

    return await graph.ainvoke(state)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Data Filtering Workflow CLI Tool")
    parser.add_argument("--train_file", type=str, 
                        help="Path to the full dataset to be filtered (.json or .jsonl)")
    parser.add_argument("--dataset_name", type=str, default="my_filter_task",
                        help="Task name, determines the output directory name")
    parser.add_argument("--filtered_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://123.129.219.111:3000/v1",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--api_key", type=str, default="sk-your-key-here")
    parser.add_argument("--target", type=str, 
                        default="Obtain SFT data filtering operators and generate pipeline code")
    parser.add_argument("--writer_target", type=str,
                        default="Generate executable code and save filtered training data")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Maximum rounds for Agent self-correction/debug")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable the Agent's self-repair debug phase")
    parser.add_argument("--pipeline_file_path", type=str,
                        default="pipeline_code.py",)
    return parser.parse_args()

async def run_filter_pipeline(args) -> str:
    """Stage 1: Run Agent workflow to generate and validate filtering code."""
    if not os.path.exists(args.train_file):
        log.error(f"❌ Input file not found: {args.train_file}")
        sys.exit(1)
        
    # Construct the Request object
    req = MainRequest(
        chat_api_url=args.api_url,
        api_key=args.api_key,
        target=args.target,
    )
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"❌ Required sample and complete files not found: {args.train_file}. "
                                f"Please ensure the sampling step completed successfully.")
    # Inject task configurations
    req.dataset_name = args.dataset_name
    req.real_json_file = args.train_file
    req.json_file = process_sample(args.train_file)
    
    req.writer_target = args.writer_target
    req.need_debug = not args.no_debug
    req.max_debug_rounds = args.max_rounds
    req.output_root = args.output_root
    req.pipeline_file_path = args.pipeline_file_path

    state = MainState(messages=[], request=req)
    return state
    log.info("🤖 Starting Agent workflow for code generation and validation...")
    final_state: MainState = await run_workflow("filter", state)
    return final_state
    
def parallel_exec_node_with_smart_retry(pipeline_file_path, state):
    """主函数：智能重试版本"""
    dataset_path = state.get("request", {}).real_json_file
    
    try:
        with open(pipeline_file_path, 'r', encoding='utf-8') as f:
            filter_code = f.read()
    except Exception as e:
        log.error(f"Failed to read code file: {e}")
        return False
    
    num_gpus = torch.cuda.device_count() or 1
    print(f"⚙️ 检测到 {num_gpus} 个GPU")
    
    for gpu_id in range(num_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
        print(f"GPU {gpu_id}: 空闲 {free_bytes/1024/1024:.0f}MB / 总计 {total_bytes/1024/1024:.0f}MB")
    
    chunk_files = split_dataset(dataset_path, num_gpus)
    
    results = []
    
    for i, chunk_file in enumerate(chunk_files):
        original_gpu_id = i % num_gpus
        
        print(f"\n📦 处理分片 {i+1}/{len(chunk_files)}")
        
        success = run_single_chunk_with_retry(
            chunk_file=chunk_file,
            filter_code=filter_code,
            chunk_idx=i,
            original_gpu_id=original_gpu_id,
            max_retries=3  
        )
        
        results.append(success)
        
        if not success:
            print(f"⚠️ 分片 {i} 失败，继续处理下一个...")
    
    success_count = sum(results)
    print(f"\n{'='*50}")
    print(f"完成: {success_count}/{len(chunk_files)} 个分片成功")
    
    return success_count == len(chunk_files)

def parallel_exec_node(pipeline_file_path, state):
    """Stage 2: Perform multi-GPU parallel filtering using the generated code."""
    dataset_path = state.get("request", {}).real_json_file

    try:
        with open(pipeline_file_path, 'r', encoding='utf-8') as f:
            filter_code = f.read()
    except Exception as e:
        log.error(f"Failed to read code file: {e}")
        return False

    num_gpus = torch.cuda.device_count() or 1
    num_shards = 100
    log.info(f"⚙️ Detected {num_gpus} GPU(s). Preparing data shards...")
    
    # Shard the dataset
    chunk_files = split_dataset(dataset_path, num_shards)
    
    # Construct parallel tasks
    tasks = []
    for i, chunk_file in enumerate(chunk_files):
        gpu_id = int(i) % num_gpus
        tasks.append((chunk_file, gpu_id, filter_code, i, dataset_path))
    
    # Create the parallel driver script
    script_path = create_simple_parallel_script(tasks, num_gpus)
    
    log.info(f"🚀 Launching parallel execution script: {script_path}")
    try:
        runpy.run_path(script_path, run_name="__main__")
        log.info(f"✅ Parallel filtering complete. Processed {len(chunk_files)} shards.")
        return True
    except Exception as e:
        log.error(f"💥 Crash occurred during parallel execution: {e}")
        traceback.print_exc()
        return False

def merge_file_node(args):
    """Stage 3: Merge all shard results into final files."""
    dataset_name = args.dataset_name
    # Ensure merging path logic aligns with parallel_exec output
    cache_local_dir = os.path.join(args.output_root)
    output_pattern = os.path.join(cache_local_dir, "merge_step{step}.jsonl")

    log.info(f"Merging shard results from directory: {cache_local_dir}")
    try:
        merged_files = merge_jsonl_results(
            cache_dir=cache_local_dir, 
            output_file_pattern=output_pattern,
            dataset_name=dataset_name
        )
        if merged_files:
            log.info(f"🎊 Task completed successfully! Final merged files: {merged_files}")
        return merged_files
    except Exception as e:
        log.error(f"File merging failed: {e}")
        return None

def process_sample(train_file):
    json_file_path = Path(train_file)
    lines = json_file_path.read_text(encoding="utf-8").splitlines()

    n_sample = 100
    out = lines[:n_sample]

    sample_file = json_file_path.parent / f"{json_file_path.stem}_sample.jsonl"
    sample_file.write_text("\n".join(out), encoding="utf-8")
    return str(sample_file)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    # Agent writes and validates code on small sample
    # We only proceed if we get a "validated" pipeline script path

    final_state = asyncio.run(run_filter_pipeline(args))
    # Verify execution results
    # exec_res = final_state.get("execution_result", {})
    # if exec_res.get("success") and "file_path" in exec_res:
    #     pipeline_file_path = exec_res["file_path"]
    #     log.info(f"✅ Agent validation successful! Code generated at: {pipeline_file_path}")
    # else:
    #     log.error("❌ Agent failed to generate executable code or validation failed.")
    #     exit(0)

    pipeline_file_path = "pipeline_code.py"  # For testing, we directly use the expected output path
    if pipeline_file_path:
        #  Execute large-scale parallel task
        success = parallel_exec_node(pipeline_file_path, final_state)
        
        # Merge results
        if success:
            merged_files = merge_file_node(args)
            filtered_file = os.path.join(args.output_root, args.filtered_file)
            with open(filtered_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(merged_files, ensure_ascii=False) + '\n')
        else:
            log.error("Parallel execution failed. Skipping merge phase.")
    else:
        log.error("Could not proceed to parallel execution. Please check Agent debug logs.")