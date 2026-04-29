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
    def run_cmd(command: str, cwd: Path, stage_log_path: Path):
        logger.info(f"Running command in {cwd}: {command}")
        env = os.environ.copy()
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
        self.filter_cfg = self._load_yaml(args.filter_config_path)
        self.train_cfg = self._load_yaml(args.train_config_path)
        self.eval_cfg = self._load_yaml(args.eval_config_path)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_files = args.train_files
        if "args" in self.filter_cfg:
            for key, value in self.filter_cfg["args"].items():
                self.filter_cfg[key] = value
        
        if "filter" in self.args.stage:
            self.filter_id = self._generate_filter_id()
            self.filter_run_dir = (Path(self.args.output_root) / "data" / self.filter_id).absolute()
            self.filter_run_dir.mkdir(parents=True, exist_ok=True)

        self.train_eval_id = self._generate_train_eval_id()
        self.train_and_eval_run_dir = (Path(self.args.output_root) / "experiments" / self.train_eval_id).absolute()
        self.train_and_eval_run_dir.mkdir(parents=True, exist_ok=True)

    def _generate_train_eval_id(self) :
        model_name = self.train_cfg.get("model_name_or_path", "")
        model_name = f"{model_name.replace('/', '-')}"
        
        if "train" in self.args.stage:
            if "filter" in self.args.stage:
                method = self.filter_cfg['method']
                exp_id = f"exp_{method}_{model_name}_{self.timestamp}"
            else:
                exp_id = f"exp_{model_name}_{self.timestamp}"
        else:
            exp_id = f"exp_{self.timestamp}"
        return exp_id

    def _generate_filter_id(self) -> str:
        dataset = Path(self.filter_cfg['train_file']).stem
        method = self.filter_cfg['method']
        return f"{method}_{dataset}_{self.timestamp}"

    def _load_yaml(self, path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

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
            args_list.append(f"--{prefix}") 
            for item in data:
                args_list.append(shlex.quote(str(item)))
        else:
            args_list.append(f"--{prefix} {shlex.quote(str(data))}")

    def build_filter_command(self):
        args_list = []
        for key, value in self.filter_cfg["args"].items():
            if key in ("train_file", "test_train_file"): value = str(Path(value).absolute())
            self.add_args(args_list, key, value)
        self.add_args(args_list, "output_root", self.filter_run_dir)

        cmd = (
            f"python start.py "
            + " ".join(args_list)
        )
        return cmd

    def _run_filter(self):
        cwd = PROJECT_ROOT / "baselines" / self.filter_cfg["method"]
        log_p = self.filter_run_dir / "filter.log"
        
        cmd = self.build_filter_command()
        return TaskRunner.run_cmd(cmd, cwd, log_p)

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

        python_exec = self.get_conda_python(self.args.env_name)
        cmd = (
            f"{python_exec} train_eval.py "
            + " ".join(args_list)
        )
        if "train" in self.args.stage:
            cmd += f" --train_config_path {self.args.train_config_path}"
            
        if "eval" in self.args.stage:
            cmd += f" --eval_config_path {self.args.eval_config_path}"

        return cmd

    def _run_train_and_eval(self):
        f_cfg = self.filter_cfg
        cwd = PROJECT_ROOT
        filtered_file = None
        if "filter" in self.stage:
            filtered_file = self.filter_run_dir / f_cfg['filtered_file'] 

            if  f_cfg["method"] == "dfa":
                # If it is the DFA method, the dataset after the nth step of filtering needs to be selected.
                output_file = self.filter_run_dir / f_cfg['filtered_file']
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        line = f.readline()
                        if line:
                            data = json.loads(line)
                            sorted_keys = sorted(data.keys(), key=int)
                            step_key = sorted_keys[f_cfg["step"]]
                            filtered_file = data[step_key]
                            filtered_file = self.filter_run_dir / filtered_file
        
        log_p = self.train_and_eval_run_dir / "train_eval.log"
        if filtered_file is None or not os.path.exists(filtered_file):
            filtered_file = self.train_files
        
        cmd = self.build_train_command(
            train_files=filtered_file,
            exp_id=str(self.train_eval_id),
        )
        return TaskRunner.run_cmd(cmd, cwd, log_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_config_path", default=None)
    parser.add_argument("--train_config_path", default=None)
    parser.add_argument("--eval_config_path", default=None)
    parser.add_argument("--train_files", default=None, nargs='+',)
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--stage", default="all", help="filter, train, eval")
    parser.add_argument("--env_name", default="bench")
    args = parser.parse_args()
    
    platform = ResearchPlatform(args)
    platform.start()