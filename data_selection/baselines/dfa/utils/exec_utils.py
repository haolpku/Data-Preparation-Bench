import os
import sys
import textwrap
import pickle
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

def extract_traceback(stderr_text: str):
    traceback_start = stderr_text.find("Traceback (most recent call last):")
    
    if traceback_start == -1:
        return stderr_text, ""
    
    traceback_text = stderr_text[traceback_start:]
    stderr = stderr_text[:traceback_start]
    return stderr.strip(), traceback_text.strip()

def run_python_file(file_path, env=None):
    env = os.environ.copy() if env is None else env
    env["PYTHONUNBUFFERED"] = "1"  
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    cmd = [sys.executable, "-u", str(file_path)]
    
    all_output = []
    stdout_lines = []
    stderr_lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  
            text=True,
            env=env,
            bufsize=1, 
            encoding='utf-8'
        )
        def stream_reader(pipe, container, is_stderr):
            for line in iter(pipe.readline, ''):
                if line:
                    container.append(line)
                    prefix = "[STDERR] " if is_stderr else ""
                    print(f"{prefix}{line}", end="", flush=True)

        t1 = threading.Thread(target=stream_reader, args=(proc.stdout, stdout_lines, False))
        t2 = threading.Thread(target=stream_reader, args=(proc.stderr, stderr_lines, True))
        t1.start()
        t2.start()

        proc.wait()
        t1.join()
        t2.join()

        stderr_text = "".join(stderr_lines)
        stderr_clean, stack_trace = extract_traceback(stderr_text)
        
        return proc.returncode, "".join(stdout_lines), stderr_clean, stack_trace

    except Exception as e:
        return -2, "", "", f"[Internal Error] {str(e)}"

def create_simple_parallel_script2(tasks, num_gpus: int) -> str:
    # 确保 tasks 内部的 gpu_id 是按 0,1,2,3... 轮询分配的
    tasks_data_str = repr(tasks)

    script_content = textwrap.dedent(f'''
        import os
        import sys
        import subprocess
        import multiprocessing as mp
        from multiprocessing import set_start_method
        import re
        import traceback

        def run_filter_on_chunk(args):
            chunk_file, gpu_id, filter_code, chunk_idx, _ = args
            
            # 动态替换脚本中的占位符，定位当前分片文件和缓存前缀
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
            
            # 强制退出逻辑，确保子进程释放显存
            filter_code += "\\nimport os\\n"
            filter_code += "os._exit(0)"

            tmp_file = f"tmp_exec_chunk_{{chunk_idx}}.py"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(filter_code)
            
            # 关键：为子进程设置独立的显卡环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TOKENIZERS_PARALLELISM"] = "false"
            
            try:
                # 执行过滤算子
                subprocess.run([sys.executable, tmp_file], env=env, check=True)
                print(f"--- [GPU {{gpu_id}}] 分片 {{chunk_idx}} 处理完成 ---")
            except subprocess.CalledProcessError as e:
                print(f"!!! [GPU {{gpu_id}}] 分片 {{chunk_idx}} 执行失败: {{e}}")
                traceback.print_exc()
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        def main():
            # 使用 spawn 模式防止 CUDA 上下文污染
            try:
                set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            tasks = {tasks_data_str}
            num_total = len(tasks)
            print(f"🚀 启动并行驱动程序，总任务数: {{num_total}}, 并行 GPU 数: {num_gpus}")
            
            # 使用 Pool 替代手动 Process 列表管理
            # processes={num_gpus} 确保同时只有 num_gpus 个进程在跑
            with mp.Pool(processes={num_gpus}) as pool:
                # imap_unordered 会在任务完成后立即返回结果，适合这种流式处理
                # 它会自动管理队列：谁空闲谁领任务
                list(pool.imap_unordered(run_filter_on_chunk, tasks))

        if __name__ == "__main__":
            main()
    ''').strip()

    script_path = "./simple_parallel_filter.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def create_simple_parallel_script(tasks, num_gpus: int):
    # 依然建议使用外部 pickle 存储任务，保持脚本整洁
    tasks_path = "tasks_metadata.pkl"
    with open(tasks_path, 'wb') as f:
        pickle.dump(tasks, f)

    script_content = textwrap.dedent(f'''
        import os
        import sys
        import subprocess
        import pickle
        import multiprocessing as mp
        from multiprocessing import Process, set_start_method

        def run_task(task_args, gpu_id):
            """
            执行单个任务的函数，绑定特定的 gpu_id
            """
            chunk_file, _, filter_code, chunk_idx, _ = task_args
            import re
            
            # 正则替换逻辑
            filter_code = re.sub(r'first_entry_file_name\s*=\s*[^,\\n)]+', f'first_entry_file_name="{{chunk_file}}"', filter_code)
            filter_code = re.sub(r'file_name_prefix\s*=\s*[^,\\n)]+', f'file_name_prefix="dataflow_cache_step_{{chunk_idx}}"', filter_code)
            filter_code += "\\nimport os\\nos._exit(0)"

            tmp_file = f"tmp/tmp_exec_idx_{{chunk_idx}}_gpu{{gpu_id}}.py"
            os.makedirs("tmp", exist_ok=True)
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(filter_code)
            
            # 设置环境变量并执行
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TOKENIZERS_PARALLELISM"] = "false"
            
            print(f"[Batch Run] GPU {{gpu_id}} 正在处理任务 {{chunk_idx}}")
            
            try:
                subprocess.run([sys.executable, tmp_file], env=env, check=True)
            except Exception as e:
                print(f"!!! [GPU {{gpu_id}}] 任务 {{chunk_idx}} 失败: {{e}}")
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        def main():
            set_start_method('spawn', force=True)
            
            with open("{tasks_path}", 'rb') as f:
                tasks = pickle.load(f)

            num_tasks = len(tasks)
            num_gpus = {num_gpus}

            # 按照 GPU 个数作为步长进行循环
            for i in range(0, num_tasks, num_gpus):
                processes = []
                
                # 在当前批次内，遍历每个 GPU
                for offset in range(num_gpus):
                    task_idx = i + offset
                    
                    # 防止最后一批任务不足 num_gpus 个
                    if task_idx < num_tasks:
                        task_args = tasks[task_idx]
                        # 这里的 offset 恰好就是 0 到 num_gpus-1
                        p = Process(target=run_task, args=(task_args, offset))
                        p.start()
                        processes.append(p)
                
                # 等待当前组所有 GPU 完成任务后再进入下一轮
                for p in processes:
                    p.join()
                
                print(f"--- 已完成第 {{i // num_gpus + 1}} 轮任务 (进度: {{min(i + num_gpus, num_tasks)}}/{{num_tasks}}) ---")

        if __name__ == "__main__":
            main()
    ''').strip()

    script_path = "./indexed_batch_filter.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def create_simple_parallel_script(tasks, num_gpus: int) -> str:
    tasks_path = "tasks_metadata.pkl"
    with open(tasks_path, 'wb') as f:
        pickle.dump(tasks, f)

    script_content = textwrap.dedent(f'''
        import os
        import sys
        import subprocess
        import pickle
        import multiprocessing as mp
        from multiprocessing import Process, Queue, set_start_method

        def run_single_task(task_args, gpu_id):
            """
            执行具体逻辑的内部函数
            """
            chunk_file, _, filter_code, chunk_idx, _ = task_args
            import re
            
            # 正则替换逻辑
            filter_code = re.sub(r'first_entry_file_name\s*=\s*[^,\\n)]+', f'first_entry_file_name="{{chunk_file}}"', filter_code)
            filter_code = re.sub(r'file_name_prefix\s*=\s*[^,\\n)]+', f'file_name_prefix="dataflow_cache_step_{{chunk_idx}}"', filter_code)
            filter_code += "\\nimport os\\nos._exit(0)"

            tmp_file = f"tmp/tmp_exec_idx_{{chunk_idx}}_gpu{{gpu_id}}.py"
            os.makedirs("tmp", exist_ok=True)
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(filter_code)
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TOKENIZERS_PARALLELISM"] = "false"
            
            print(f"[Worker GPU {{gpu_id}}] 正在启动任务 {{chunk_idx}}")
            
            try:
                subprocess.run([sys.executable, tmp_file], env=env, check=True)
            except Exception as e:
                print(f"!!! [GPU {{gpu_id}}] 任务 {{chunk_idx}} 失败: {{e}}")
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        def gpu_worker(task_queue, gpu_id):
            """
            每个 GPU 的常驻进程，不断从队列取任务
            """
            while not task_queue.empty():
                try:
                    # 获取任务，如果队列为空则退出
                    task_args = task_queue.get_nowait()
                except:
                    break
                
                run_single_task(task_args, gpu_id)

        def main():
            set_start_method('spawn', force=True)
            
            with open("{tasks_path}", 'rb') as f:
                tasks = pickle.load(f)

            num_tasks = len(tasks)
            num_gpus = {num_gpus}

            # 1. 将所有任务放入多进程队列
            task_queue = Queue()
            for t in tasks:
                task_queue.put(t)

            # 2. 为每个 GPU 开启一个 Worker 进程
            processes = []
            for gpu_id in range(num_gpus):
                p = Process(target=gpu_worker, args=(task_queue, gpu_id))
                p.start()
                processes.append(p)
            
            # 3. 等待所有 Worker 进程完成
            for p in processes:
                p.join()
            
            print(f"--- 所有任务已完成 (共 {{num_tasks}} 个) ---")

        if __name__ == "__main__":
            main()
    ''').strip()

    script_path = "./indexed_batch_filter.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path

def create_simple_parallel_script2(tasks, num_gpus: int) -> str:
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
            tmp_file = f"tmp/tmp_exec_chunk_{{chunk_idx}}.py"
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