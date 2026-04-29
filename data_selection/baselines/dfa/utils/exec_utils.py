import os
import sys

import re
import json
import math
import torch
import inspect
import subprocess
import threading
import textwrap
import glob

import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dataflow.utils.registry import OPERATOR_REGISTRY

from utils.data_utils import save_as_jsonl

def run_python_file(file_path, timeout=300):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    cmd = [sys.executable, "-u", str(file_path)]
    
    all_output = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            env=env,
            bufsize=1, 
            encoding='utf-8'
        )

        for line in iter(proc.stdout.readline, ''):
            if line:
                print(line, end="", flush=True) 
                all_output.append(line)

        proc.wait() 
        return proc.returncode, "".join(all_output), ""

    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, "".join(all_output), "[Error] Timeout expired"
    except Exception as e:
        return -2, "", f"[Internal Error] {str(e)}"

def create_simple_parallel_script(tasks, num_gpus: int) -> str:
    tasks_data_str = repr(tasks)

    script_content = textwrap.dedent(f'''
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
                r'first_entry_file_name\s*=\s*[^,\\n)]+',
                f'first_entry_file_name="{{chunk_file}}"',
                filter_code
            )

            filter_code = re.sub(
                r'file_name_prefix\s*=\s*[^,\\n)]+',
                f'file_name_prefix="dataflow_cache_step_{{chunk_idx}}"',
                filter_code
            )
            filter_code += "\\nimport os\\n"
            filter_code += "os._exit(0)"

            # 动态生成当前分片的执行脚本
            tmp_file = f"tmp_exec_chunk_{{chunk_idx}}.py"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(filter_code)
            
            print(f"--- [GPU {{gpu_id}}] 启动分片 {{chunk_idx}} 处理: {{chunk_file}} ---")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                subprocess.run([sys.executable, tmp_file], env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! 子进程执行失败: {{e}}")
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        def main():
            # 使用 spawn 模式是多卡处理的强制要求，防止 CUDA 上下文污染
            set_start_method('spawn', force=True)
            
            # 直接从序列化后的字符串加载任务
            tasks = {tasks_data_str}
            
            processes = []
            # 按 GPU 数量分批并行
            for i in range(0, len(tasks), {num_gpus}):
                batch = tasks[i : i + {num_gpus}]
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
    ''').strip()

    script_path = "./simple_parallel_filter.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path
