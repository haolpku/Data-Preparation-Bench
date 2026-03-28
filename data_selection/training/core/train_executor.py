import subprocess
import os
import sys
import time

import yaml
from pathlib import Path
from utils.logger import init_train_logger

logger = init_train_logger("train_executor")

from pathlib import Path
from loguru import logger  # 建议使用 loguru，比原生 logging 更优雅

def run_llamafactory_train(config_path, log_file=None, enable_checkpoint=True, timeout=None):
    config_path = Path(config_path).resolve()
    
    # 1. 准备配置与路径
    with config_path.open("r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    
    output_dir = Path(train_config.get("output_dir", "./train_output")).resolve()
    train_cmd = f"llamafactory-cli train {str(config_path)}"

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1]) 
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}".strip(":")

    logger.info(f"🚀 开始执行训练：{train_cmd}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            shell=True,
            encoding="utf-8",
            env=env,
            cwd=os.getcwd()
        )

        log_fh = Path(log_file).open("a", encoding="utf-8") if log_file else None

        try:
            for line in process.stdout:
                sys.stdout.write(line) 
                if log_fh:
                    log_fh.write(line) 
            
            process.wait(timeout=timeout)
            
        except subprocess.TimeoutExpired:
            process.kill() 
            logger.error("⏰ 训练超时，已强行终止")
            raise
        finally:
            if log_fh:
                log_fh.close()

        if process.returncode != 0:
            raise RuntimeError(f"❌ 训练失败，退出码: {process.returncode}")

        duration = time.time() - start_time
        logger.success(f"🎉 训练圆满完成！耗时: {duration/3600:.2f}h")

    except Exception as e:
        logger.exception(f"💥 训练运行异常: {e}")
        raise

def get_train_status(output_dir):
    output_dir = Path(output_dir)
    status_info = {
        "status": "pending",
        "trained_steps": 0,
        "train_duration": 0,
        "latest_checkpoint": None
    }

    if not output_dir.exists():
        return status_info

    train_log_file = output_dir / "train.log"
    if train_log_file.exists():
        status_info["status"] = "running"
        with open(train_log_file, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
            for line in reversed(log_lines):
                if "Training completed" in line or "Finished training" in line:
                    status_info["status"] = "finished"
                    break
                if "Exception" in line or "Error" in line and "Traceback" in line:
                    status_info["status"] = "failed"
                    break
            
            for line in log_lines:
                if "steps completed" in line or "global step" in line:
                    try:
                        # 匹配类似 "global step 500" 或 "steps completed: 500"
                        import re
                        step_match = re.search(r"(\d+)\s+(global step|steps completed)", line)
                        if step_match:
                            status_info["trained_steps"] = int(step_match.group(1))
                    except:
                        pass

    checkpoint_dirs = [d for d in output_dir.glob("checkpoint-*") if d.is_dir()]
    if checkpoint_dirs:
        latest_ckpt = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[-1]))
        status_info["latest_checkpoint"] = latest_ckpt.name
        status_info["trained_steps"] = int(latest_ckpt.name.split("-")[-1])

    return status_info
