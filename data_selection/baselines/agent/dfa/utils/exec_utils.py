import os
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

def run_python_file2(file_path, timeout=300):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"  # 添加这一行
    proc = subprocess.Popen(
        [os.sys.executable, "-u", file_path],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        bufsize=1,
        env=env,
        encoding='utf-8' 
    )
    
    out_lines, err_lines = [], []

    def read_stream(stream, collector):
        try:
            for line in iter(stream.readline, ""):
                if line:
                    collector.append(line)
        except Exception as e:
            collector.append(f"\n[Stream Read Error]: {str(e)}")
        finally:
            stream.close()

    t_out = threading.Thread(target=read_stream, args=(proc.stdout, out_lines))
    t_err = threading.Thread(target=read_stream, args=(proc.stderr, err_lines))
    t_out.setDaemon(True) 
    t_out.start(); t_err.start()
    
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
        return_code = -1 
        err_lines.append(f"\n[System Error]: Execution timed out after {timeout}s")
    
    t_out.join(timeout=2)
    t_err.join(timeout=2)
    
    return return_code, "".join(out_lines), "".join(err_lines)

def run_python_file3(file_path, timeout=300):
    import sys
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    proc = subprocess.Popen(
        [os.sys.executable, "-u", file_path],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        bufsize=0,
        env=env,
        encoding='utf-8',
        universal_newlines=True
    )
    
    out_lines, err_lines = [], []

    def read_stream(stream, collector, stream_name):
        try:
            for line in iter(stream.readline, ""):
                if line:
                    collector.append(line)
                    # 实时打印到终端
                    if stream_name == "stdout":
                        print(f"{line}", end='')
                    else:
                        print(f"{line}", end='', file=sys.stderr)
        except Exception as e:
            error_msg = f"\n[Stream Read Error]: {str(e)}"
            collector.append(error_msg)
            print(error_msg, file=sys.stderr)
        finally:
            stream.close()

    t_out = threading.Thread(target=read_stream, args=(proc.stdout, out_lines, "stdout"))
    t_err = threading.Thread(target=read_stream, args=(proc.stderr, err_lines, "stderr"))
    t_out.setDaemon(True) 
    t_err.setDaemon(True)
    t_out.start()
    t_err.start()
    
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
        return_code = -1 
        error_msg = f"\n[System Error]: Execution timed out after {timeout}s"
        err_lines.append(error_msg)
        print(error_msg, file=sys.stderr)
    
    t_out.join(timeout=2)
    t_err.join(timeout=2)
    
    return return_code, "".join(out_lines), "".join(err_lines)

def run_python_file(file_path, timeout=300):
    import sys
    import subprocess
    import threading
    import time
    import os
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 1. 添加 -u 参数，合并 stderr 到 stdout
    proc = subprocess.Popen(
        [os.sys.executable, "-u", file_path],
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,  # 合并到 stdout
        text=True, 
        bufsize=1,  # 行缓冲
        env=env,
        encoding='utf-8'
    )
    
    out_lines = []

    def read_stream():
        # 2. 逐行读取并立即打印
        for line in iter(proc.stdout.readline, ""):
            if line:
                out_lines.append(line)
                print(line, end='', flush=True)  # flush=True 强制刷新

    t = threading.Thread(target=read_stream)
    t.daemon = True
    t.start()
    
    try:
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
        return_code = -1
        error_msg = f"\n[Error]: Timeout after {timeout}s"
        out_lines.append(error_msg)
        print(error_msg, file=sys.stderr)
    
    t.join(timeout=2)
    
    return return_code, "".join(out_lines), ""  # 第三个返回值改为空

def create_simple_parallel_script(tasks, num_gpus: int) -> str:
    tasks_data_str = repr(tasks)

    script_content = textwrap.dedent(f'''
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
            tmp_file = f"tmp_exec_chunk_{{chunk_idx}}.py"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(filter_code)
            
            print(f"--- [GPU {{gpu_id}}] 启动分片 {{chunk_idx}} 处理: {{chunk_file}} ---")
            
            try:
                # 使用 runpy 运行，stdout 会直接流向父进程的终端
                runpy.run_path(tmp_file, run_name="__main__")
            except Exception as e:
                print(f"!!! [分片 {{chunk_idx}}] 执行异常: {{e}}")
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
