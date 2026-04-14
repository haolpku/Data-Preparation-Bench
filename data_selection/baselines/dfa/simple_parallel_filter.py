import os
import sys
import subprocess
import multiprocessing as mp
from multiprocessing import Process, set_start_method

def run_filter_on_chunk(args):
    chunk_file, gpu_id, filter_code, chunk_idx, _ = args

    # 设置显卡环境变量
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import re
    import runpy

    filter_code = re.sub(
        r'first_entry_file_name\s*=\s*[^,\n)]+',
        f'first_entry_file_name="{chunk_file}"',
        filter_code
    )

    filter_code = re.sub(
        r'file_name_prefix\s*=\s*[^,\n)]+',
        f'file_name_prefix="dataflow_cache_step_{chunk_idx}"',
        filter_code
    )
    filter_code += "\nimport os\n"
    filter_code += "os._exit(0)"

    # 动态生成当前分片的执行脚本
    tmp_file = f"tmp_exec_chunk_{chunk_idx}.py"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(filter_code)

    print(f"--- [GPU {gpu_id}] 启动分片 {chunk_idx} 处理: {chunk_file} ---")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        subprocess.run([sys.executable, tmp_file], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! 子进程执行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def main():
    # 使用 spawn 模式是多卡处理的强制要求，防止 CUDA 上下文污染
    set_start_method('spawn', force=True)

    # 直接从序列化后的字符串加载任务
    tasks = [('/home/hxy/dcai/Data-Preparation-Bench/data_selection/dataset/teknium/OpenHermes-2.5/processed/openhermes2_5_alpaca.jsonl', 0, 'import os\nfrom dataflow.utils.registry import OPERATOR_REGISTRY\nif hasattr(OPERATOR_REGISTRY, "_get_all"):\n    OPERATOR_REGISTRY._get_all()\n\nfrom dataflow.utils.storage import FileStorage\nfrom dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter\nfrom dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter\n\n# Initialize LLM serving (if required by any operator)\nos.environ["API_KEY"] = "sk-URhU1qxF5iVuW2ZZND5fI0pQcuESJyBt33dv4WGf3JY9dZbI"\n\n# Initialize storage\nstorage = FileStorage(\n    first_entry_file_name="/home/hxy/dcai/Data-Preparation-Bench/data_selection/dataset/teknium/OpenHermes-2.5/processed/openhermes2_5_alpaca_sample.jsonl",\n    cache_path="/home/hxy/dcai/Data-Preparation-Bench/data_selection/output/data/processed_dfa_20260414_021910/",\n    file_name_prefix="dataflow_cache_step",\n    cache_type="jsonl"\n)\n\n# Initialize operators\ndeita_quality_filter = DeitaQualityFilter(\n    min_score=2.5,\n    max_score=10000.0,\n    device=\'cuda\',\n    model_cache_dir=\'./dataflow_cache\',\n    max_length=512\n)\n\nsuperfiltering_filter = SuperfilteringFilter(\n    min_score=0.0,\n    max_score=1.0,\n    device=\'cuda\',\n    model_cache_dir=\'./dataflow_cache\',\n    max_length=512\n)\n\n# Run pipeline\n# Step 1: Run DeitaQualityFilter\ndeita_output_key = deita_quality_filter.run(\n    storage.step(),\n    input_instruction_key=\'instruction\',\n    input_output_key=\'output\',\n    output_key=\'DeitaQualityScore\'\n)\n\n# Step 2: Run SuperfilteringFilter\nsuperfiltering_output_key = superfiltering_filter.run(\n    storage.step(),\n    input_instruction_key=\'instruction\',\n    input_input_key=\'input\',\n    input_output_key=\'output\',\n    output_key=\'SuperfilteringScore\'\n)\n\n# Final output\noutput = {\n    "DeitaQualityFilterOutput": deita_output_key,\n    "SuperfilteringFilterOutput": superfiltering_output_key\n}\n\nprint(output)', 0, '/home/hxy/dcai/Data-Preparation-Bench/data_selection/dataset/teknium/OpenHermes-2.5/processed/openhermes2_5_alpaca.jsonl')]

    processes = []
    # 按 GPU 数量分批并行
    for i in range(0, len(tasks), 4):
        batch = tasks[i : i + 4]
        current_batch_procs = []

        for task_args in batch:
            p = Process(target=run_filter_on_chunk, args=(task_args,))
            p.start()
            current_batch_procs.append(p)

        # 等待当前批次完成，保证显存不会溢出
        for p in current_batch_procs:
            p.join()

if __name__ == "__main__":
    main()