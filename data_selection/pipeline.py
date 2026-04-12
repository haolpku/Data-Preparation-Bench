import argparse
import yaml
import os
import shlex

import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TaskRunner:
    @staticmethod
    def run_cmd(command: str, cwd: Path, env: Dict, stage_log_path: Path):
        logger.info(f"Running command in {cwd}: {command}")
        env = env.copy() if env else {}
        env["HF_ALLOW_CODE_EVAL"] = "1" 

        if isinstance(command, str):
            cmd_list = shlex.split(command)
        else:
            cmd_list = command

        with open(stage_log_path, "w") as log_f:
            try:
                process = subprocess.Popen(
                    cmd_list,
                    shell=False,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # bufsize=1,
                    bufsize=0,
                    universal_newlines=True,
                )
                
                for line in iter(process.stdout.readline, ''):
                    if line: 
                        print(line, end="", flush=True) 
                    log_f.write(line)
                    log_f.flush()   
                
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
                return True
            except Exception as e:
                logger.error(f"Stage execution failed: {e}")
                return False

class ResearchPlatform:
    def __init__(self, args):
        self.args = args
        self.stage = args.stage
        self.resume_id = args.resume_id or None
        self.filter_cfg = self._load_yaml(args.filter_config_path)
        self.train_cfg = self._load_yaml(args.train_config_path)
        self.eval_cfg = self._load_yaml(args.eval_config_path)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_file = Path(args.train_file).absolute()
        if "args" in self.filter_cfg:
            for key, value in self.filter_cfg["args"].items():
                self.filter_cfg[key] = value
        
        filter_id = self._generate_filter_id()
        self.filter_run_dir = (Path(self.filter_cfg['output_root']) / "data_selection" / filter_id).absolute()
        self.filter_run_dir.mkdir(parents=True, exist_ok=True)
        self.train_and_eval_run_dir = (Path(self.filter_cfg['output_root']) / "train_and_eval" / filter_id).absolute()
        self.train_and_eval_run_dir.mkdir(parents=True, exist_ok=True)
        self.env = self._prepare_env()

    def _generate_filter_id(self) -> str:
        if self.resume_id is not None: return self.resume_id
        data = self.train_file.parent.stem
        method = self.filter_cfg['method']
        return f"{data}_{method}_{self.timestamp}"

    def _load_yaml(self, path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _prepare_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) 
        env["DEBUG_MODE"] = "1"
        return env

    def start(self):
        # 1. Filtering
        if "filter" in self.stage:
            logger.info("🔍 Starting data filtering stage...")
            if not self._run_filter(): return

        # 2. Training & Evaluation
        if any(s in self.stage for s in ["train", "eval"]):
            if not self._run_train_and_eval(): return
            
    def add_args(self, args_list, prefix, data):
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self.add_args(args_list, new_prefix, value)
        elif isinstance(data, list):
            value_str = " ".join(str(item) for item in data)
            args_list.append(f"--{prefix} {shlex.quote(value_str)}")
        else:
            args_list.append(f"--{prefix} {shlex.quote(str(data))}")

    def build_filter_command(self):
        args_list = []
        for key, value in self.filter_cfg["args"].items():
            self.add_args(args_list, key, value)
        self.add_args(args_list, "output_root", self.filter_run_dir)
        self.add_args(args_list, "train_file", self.train_file)
        cmd = (
            f"python start.py "
            + " ".join(args_list)
        )
        return cmd

    def _run_filter(self):
        cwd = PROJECT_ROOT / "baselines" / self.filter_cfg["method"]
        log_p = self.filter_run_dir / "filter.log"
        
        cmd = self.build_filter_command()
        return TaskRunner.run_cmd(cmd, cwd, self.env, log_p)

    def get_conda_python(self, env_name: str):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            base_path = os.path.dirname(conda_prefix)
            target_python = os.path.join(base_path, env_name, "bin", "python")
            if os.path.exists(target_python):
                return target_python
        
        return f"conda run -n {env_name} --no-capture-output python"
        
    def build_train_command(self, **kwargs):
        args_list = []
        for key, value in kwargs.items():
            self.add_args(args_list, key, value)

        python_exec = self.get_conda_python("bench")
        cmd = (
            f"{python_exec} train_eval.py "
            + " ".join(args_list)
        )
        mode = []
        if "train" in self.args.stage:
            mode.append("train")
        if "eval" in self.args.stage:
            mode.append("eval")
        mode = ",".join(mode)
        cmd += (
            f" --train_config_path {self.args.train_config_path}"
            f" --eval_config_path {self.args.eval_config_path}"
            f" --mode {mode}"
        )
        return cmd

    def _run_train_and_eval(self):
        f_cfg = self.filter_cfg
        cwd = PROJECT_ROOT
        filtered_file = None
        if "filter" in self.stage or self.resume_id is not None:
            filtered_file = self.filter_run_dir / f_cfg['filtered_file'] 

            if  f_cfg["method"] == "dfa":
                # 如果是dfa方法，需要选择过滤的第x步后数据集
                output_file = self.filter_run_dir / f_cfg['filtered_file']
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        if line:
                            data = json.loads(line)
                            filtered_file = data[str(f_cfg["step"])]
                            filtered_file = self.filter_run_dir / filtered_file
        
        log_p = self.train_and_eval_run_dir / "train_eval.log"
        if filtered_file is None or not os.path.exists(filtered_file):
            filtered_file = self.train_file
        
        cmd = self.build_train_command(
            train_file=str(filtered_file),
            output_root=str(self.train_and_eval_run_dir)
        )
        return TaskRunner.run_cmd(cmd, cwd, self.env, log_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_config_path", default="configs/dfa.yaml")
    parser.add_argument("--train_config_path", default="configs/train_qwen2.5.yaml")
    parser.add_argument("--eval_config_path", default="configs/eval.yaml")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--stage", default="all", help="filter, train, eval")
    parser.add_argument("--resume_id", default=None, help="previous experiment ID for connecting the subsequent steps.")
    args = parser.parse_args()
    
    platform = ResearchPlatform(args)
    platform.start()