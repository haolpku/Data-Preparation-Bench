import os
import sys
import runpy
import multiprocessing as mp
from multiprocessing import Process, set_start_method

# 预加载 Registry，防止子进程重复加载导致的竞争
try:
    from dataflow.utils.registry import OPERATOR_REGISTRY
    if hasattr(OPERATOR_REGISTRY, "_get_all"):
        OPERATOR_REGISTRY._get_all()
except ImportError:
    pass

def run_filter_on_chunk(args):
    chunk_file, gpu_id, filter_code, chunk_idx, _ = args

    # 设置显卡环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 动态生成当前分片的执行脚本
    tmp_file = f"tmp_exec_chunk_{chunk_idx}.py"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write(filter_code)

    print(f"--- [GPU {gpu_id}] 启动分片 {chunk_idx} 处理: {chunk_file} ---")

    try:
        # 使用 runpy 运行，stdout 会直接流向父进程的终端
        runpy.run_path(tmp_file, run_name="__main__")
    except Exception as e:
        print(f"!!! [分片 {chunk_idx}] 执行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def main():
    # 使用 spawn 模式是多卡处理的强制要求，防止 CUDA 上下文污染
    set_start_method('spawn', force=True)

    # 直接从序列化后的字符串加载任务
    tasks = [('/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_chunk_0.jsonl', 0, '\nfrom dataflow.utils.registry import OPERATOR_REGISTRY\nif hasattr(OPERATOR_REGISTRY, "_get_all"):\n    OPERATOR_REGISTRY._get_all()\n\nfrom dataflow.utils.storage import FileStorage\nfrom dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter\nfrom dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter\n\nclass DataFilteringPipeline:\n    def __init__(self):\n        self.storage = FileStorage(\n            first_entry_file_name="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_extracted_sample.jsonl",\n            cache_path="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/output/my_filter_task/",\n            file_name_prefix="dataflow_cache_step",\n            cache_type="jsonl"\n        )\n\n    def forward(self):\n        return ["test1", "test2"]\n\npipeline = DataFilteringPipeline()\nfiltered_keys = pipeline.forward()\nprint(f"Filtered data saved with keys: {filtered_keys}")\n', 0, '/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted.jsonl'), ('/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_chunk_1.jsonl', 1, '\nfrom dataflow.utils.registry import OPERATOR_REGISTRY\nif hasattr(OPERATOR_REGISTRY, "_get_all"):\n    OPERATOR_REGISTRY._get_all()\n\nfrom dataflow.utils.storage import FileStorage\nfrom dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter\nfrom dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter\n\nclass DataFilteringPipeline:\n    def __init__(self):\n        self.storage = FileStorage(\n            first_entry_file_name="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_extracted_sample.jsonl",\n            cache_path="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/output/my_filter_task/",\n            file_name_prefix="dataflow_cache_step",\n            cache_type="jsonl"\n        )\n\n    def forward(self):\n        return ["test1", "test2"]\n\npipeline = DataFilteringPipeline()\nfiltered_keys = pipeline.forward()\nprint(f"Filtered data saved with keys: {filtered_keys}")\n', 1, '/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted.jsonl'), ('/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_chunk_2.jsonl', 2, '\nfrom dataflow.utils.registry import OPERATOR_REGISTRY\nif hasattr(OPERATOR_REGISTRY, "_get_all"):\n    OPERATOR_REGISTRY._get_all()\n\nfrom dataflow.utils.storage import FileStorage\nfrom dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter\nfrom dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter\n\nclass DataFilteringPipeline:\n    def __init__(self):\n        self.storage = FileStorage(\n            first_entry_file_name="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_extracted_sample.jsonl",\n            cache_path="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/output/my_filter_task/",\n            file_name_prefix="dataflow_cache_step",\n            cache_type="jsonl"\n        )\n\n    def forward(self):\n        return ["test1", "test2"]\n\npipeline = DataFilteringPipeline()\nfiltered_keys = pipeline.forward()\nprint(f"Filtered data saved with keys: {filtered_keys}")\n', 2, '/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted.jsonl'), ('/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_chunk_3.jsonl', 3, '\nfrom dataflow.utils.registry import OPERATOR_REGISTRY\nif hasattr(OPERATOR_REGISTRY, "_get_all"):\n    OPERATOR_REGISTRY._get_all()\n\nfrom dataflow.utils.storage import FileStorage\nfrom dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter\nfrom dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter\n\nclass DataFilteringPipeline:\n    def __init__(self):\n        self.storage = FileStorage(\n            first_entry_file_name="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_extracted_sample.jsonl",\n            cache_path="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/output/my_filter_task/",\n            file_name_prefix="dataflow_cache_step",\n            cache_type="jsonl"\n        )\n\n    def forward(self):\n        return ["test1", "test2"]\n\npipeline = DataFilteringPipeline()\nfiltered_keys = pipeline.forward()\nprint(f"Filtered data saved with keys: {filtered_keys}")\n', 3, '/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted.jsonl')]

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