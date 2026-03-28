import argparse
import yaml
import os
import shlex

import subprocess
import logging
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from utils.preprocess import DataPreprocessor

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ExperimentManager:
    def __init__(self, args, base_cfg: Dict[str, Any], resume_id: str = None):
        self.args = args
        self.cfg = base_cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if resume_id:
            self.exp_id = resume_id
            logger.info(f"🔄 Resuming existing experiment: {self.exp_id}")
        else:
            self.exp_id = self._generate_exp_id()
            logger.info(f"🆕 Starting new experiment: {self.exp_id}")

        # self.exp_id = "my_filter_task"
        self.run_dir = (Path(base_cfg['output_root']) / self.exp_id).absolute()
        self._setup_dir()

    def _generate_exp_id(self) -> str:
        model = self.cfg['train']['model_path'].split('/')[-1]
        data = Path(self.cfg['filter']['train_file']).parent.stem
        method = self.cfg['filter']['method']
        return f"{method}_{data}_{model}_{self.timestamp}"

    def _setup_dir(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env_data = {**vars(self.args), **self.cfg}
        config_file = self.run_dir / "experiment_config.yaml"
        if not config_file.exists():
            with open(config_file, "w") as f:
                yaml.dump(env_data, f)

class TaskRunner:
    @staticmethod
    def run_cmd(command: str, cwd: Path, env: Dict, stage_log_path: Path):
        logger.info(f"Running command in {cwd}: {command}")
        
        with open(stage_log_path, "w") as log_f:
            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                for line in process.stdout:
                    print(line, end="") 
                    log_f.write(line)   
                
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
        resume_id = args.resume_id or None
        self.config_path = args.config
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        train_file = self.preprocess_data(self.cfg['filter']["args"])
        self.cfg['filter']["args"]["train_file"]  = Path(train_file).absolute()

        if "args" in self.cfg["filter"]:
            for key, value in self.cfg["filter"]["args"].items():
                self.cfg["filter"][key] = value
        self.data_preprocessor = DataPreprocessor()
        self.exp_manager = ExperimentManager(args, self.cfg, resume_id=resume_id)
        self.env = self._prepare_env()

    def _prepare_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) 
        return env

    def _save_latest_id(self):
        record_file = PROJECT_ROOT / f"{self.args.latest_run_path}"
        with open(record_file, "w") as f:
            f.write(self.exp_manager.exp_id)
        logger.info(f"💾 Experiment ID saved to {record_file}")

    def start(self):
        # 1. Filtering
        if any(s in self.stage for s in ["all", "filter"]):
            logger.info("🔍 Starting data filtering stage...")
            if self._run_filter():
                self._save_latest_id()

        # 2. Training
        if any(s in self.stage for s in ["all", "train"]):
            logger.info("🚀 Starting model training stage...")
            if not self._run_train(): return
        
        # 3. Evaluation
        if any(s in self.stage for s in ["all", "eval"]):
            logger.info("📊 Starting model evaluation stage...")
            if not self._run_eval(): return

    def preprocess_data(self, f_cfg):
        input_file = Path(f_cfg['train_file'])
        output_file = input_file.parent / f"{input_file.stem}_alpaca.jsonl"
        if output_file.exists(): return str(output_file)
        self.data_preprocessor.preprocess(input_file, output_file)
        return str(output_file)

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
        filter_cfg = self.cfg.get("filter", {})
        env_name = filter_cfg.get("env", "default_env")
        
        args_list = []
        for key, value in self.cfg["filter"]["args"].items():
            self.add_args(args_list, key, value)
        self.add_args(args_list, "output_root", self.exp_manager.run_dir)

        cmd = (
            f"python -m debugpy --listen 5679 --wait-for-client start.py "
            + " ".join(args_list)
        )
        return cmd

    def _run_filter(self):
        f_cfg = self.cfg['filter']
        cwd = PROJECT_ROOT / "baselines" / f_cfg["type"] / f_cfg["method"]
        log_p = self.exp_manager.run_dir / "filter.log"
        
        cmd = self.build_filter_command()
        return TaskRunner.run_cmd(cmd, cwd, self.env, log_p)

    def build_train_command(self, **kwargs):
        train_cfg = self.cfg.get("train", {})
        
        args_list = []
        for key, value in self.cfg["train"].items():
            self.add_args(args_list, key, value)
        for key, value in kwargs.items():
            self.add_args(args_list, key, value)
        self.add_args(args_list, "output_root", self.exp_manager.run_dir / "adapter_model")

        cmd = (
            f"python train_entry.py "
            + " ".join(args_list)
        )
        return cmd

    def _run_train(self):
        t_cfg = self.cfg['train']
        f_cfg = self.cfg['filter']
        cwd = PROJECT_ROOT / "training"
        log_p = self.exp_manager.run_dir / "train.log"
        
        filtered_file = self.exp_manager.run_dir / f_cfg['filtered_file'] 
        if not filtered_file.exists():
            if "train_file" in t_cfg:
                filtered_file = t_cfg["filtered_file"]
            else:
                filtered_file = None

        if any(s in self.stage for s in ["filter"]) and f_cfg["method"] == "dfa":
            output_file = self.exp_manager.run_dir / f_cfg['filtered_file']
            with open(output_file, 'r', encoding='utf-8') as f:
                line = f.readline()
                if line:
                    data = json.loads(line)
                    filtered_file = data[str(f_cfg["step"])]

        if filtered_file is None or not os.path.exists(filtered_file):
            logger.error(f"❌ Filtered file not found: {filtered_file}. Did you run 'filter' stage?")
            return False

        dataset_name = Path(f_cfg['train_file']).parent.name
        cmd = self.build_train_command(
            train_file=str(filtered_file),
            train_dataset_name=str(dataset_name) 
        )
        return TaskRunner.run_cmd(cmd, cwd, self.env, log_p)

    def _run_eval(self):
        e_cfg = self.cfg['eval']
        cwd = PROJECT_ROOT #/ "eval"
        log_p = self.exp_manager.run_dir / "eval.log"
        output_dir = self.exp_manager.run_dir / 'eval_results'
        os.makedirs(output_dir, exist_ok=True)
        cmd = (
            f"python eval/run_eval.py "
            f"--model_path {self.cfg['train']['model_path']} "
            f"--cfg {self.config_path} "
            f"--output_dir {output_dir}"
        )
        if any(s in self.stage for s in ["filter"]):
            cmd += f"--peft_path {self.exp_manager.run_dir / 'adapter_model'} "

        return TaskRunner.run_cmd(cmd, cwd, self.env, log_p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_exp.yaml")
    parser.add_argument("--stage", default="all", help="all, filter, train, eval")
    parser.add_argument("--resume_id", help="previous experiment ID for connecting the subsequent steps.")
    parser.add_argument("--latest_run_path", default="./.latest_run")
    args = parser.parse_args()
    
    platform = ResearchPlatform(args)
    platform.start()