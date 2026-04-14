import argparse
import os
import yaml
import json
import subprocess
import sys
import torch
from pathlib import Path
from datetime import datetime

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union


def register_dataset_to_train_dir(
    source_data_path: Union[str, Path],
    dataset_tmp_dir: Union[str, Path],
    column_mapping: Dict[str, str]
):
    """
    将过滤后的数据集注册到 LLaMA Factory 的训练临时目录中
    完全符合你的最终目录规范：
    1.  将数据集文件复制
    2.  重命名
    3.  自动生成/更新 dataset_info.json
    4.  返回 LF 可用的 ds_name
    """
    source_path = Path(source_data_path)
    filter_id = f"{source_path.parent.stem}_{source_path.stem}"
    info_path = dataset_tmp_dir / "dataset_info.json"
    
    # 执行复制并重命名
    target_filename = f"{filter_id}.jsonl" 
    target_path = dataset_tmp_dir / target_filename
    
    shutil.copy2(source_path, target_path)
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            ds_registry = json.load(f)
    else:
        ds_registry = {}

    # 写入新的数据集注册信息
    ds_registry[filter_id] = {
        "file_name": target_filename,
        "formatting": "alpaca",
        "columns": column_mapping,
        "register_time": datetime.now().isoformat()
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(ds_registry, f, indent=2, ensure_ascii=False)

    return filter_id

def register_datasets_batch(
    source_data_paths: List[Union[str, Path]],
    train_exp_dir: Union[str, Path],
    column_mapping: Dict[str, str]
):
    """
    【批量版】支持同时注册多个数据集，用于混合训练
    返回最终的 dataset_dir 路径
    """
    ds_name_list = []
    # 训练临时数据集目录
    dataset_tmp_dir = Path(train_exp_dir) / "dataset"
    dataset_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    for path in source_data_paths:
        ds_name = register_dataset_to_train_dir(path, dataset_tmp_dir, column_mapping)
        ds_name_list.append(ds_name)   
         
    if len(ds_name_list) == 0:
        print("没有传入任何有效数据集路径, 需要在config中手动指定dataset和dataset_dir！")

    ds_name = ",".join(ds_name_list)  
    return dataset_tmp_dir, ds_name

def delete_train_temp_dataset(train_exp_dir: Union[str, Path]):
    """
    删除训练生成的临时 dataset 目录
    训练完成后调用，清理所有复制的 jsonl 和 dataset_info.json
    """
    dataset_tmp_dir = Path(train_exp_dir)  / "dataset"

    if dataset_tmp_dir.exists() and dataset_tmp_dir.is_dir():
        shutil.rmtree(dataset_tmp_dir)

def _build_train_config(config, ds_name, dataset_dir, output_dir):
    """Reads the base YAML template and injects runtime parameters."""
    # Override critical dynamic parameters
    config.update({
        "dataset": ds_name,
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
    })
    return config

def _run_llamafactory(config_path, gpu_id=0):
    """Calls llamafactory-cli to execute the training process."""
    env = os.environ.copy()
    has_cuda = torch.cuda.is_available()
    use_cpu = not has_cuda or str(gpu_id).lower() == "cpu"
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        train_cmd = ["llamafactory-cli", "train", str(config_path)]
        print("🔵 Running on CPU mode")
    else:
        # train_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} llamafactory-cli train {config_path}"
        if "," in str(gpu_id):
            nproc = len(str(gpu_id).split(","))
            train_cmd = [
                "torchrun", 
                f"--nproc_per_node={nproc}", 
                "third_party/LlamaFactory/src/train.py", 
                str(config_path)
            ]
        else:
            train_cmd = ["llamafactory-cli", "train", str(config_path)]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure current directory is in PYTHONPATH for custom module discovery
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}".strip(":")
    env_name = "bench"
    full_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + train_cmd
    print(f"🚀 Launching training: {train_cmd}")
    try:
        process = subprocess.Popen(
            full_cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env
        )
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"❌ Training exited with error code: {process.returncode}")
    except Exception as e:
        print(f"💥 Execution failed: {e}")
        raise

def run_training(train_config_path, train_files, train_config, train_exp_dir, gpu_id=0):
    try:
        # Step 1: Register Dataset
        dataset_dir, ds_name = register_datasets_batch(
            source_data_paths=train_files,
            train_exp_dir=train_exp_dir,
            column_mapping={"prompt": "instruction", "query": "input", "response": "output"}
        )
        if not dataset_dir:
            assert train_config.get("dataset") and train_config.get("dataset_dir"), \
                "❌ Error: 'dataset' and 'dataset_dir' must be configured in train_config!"
            ds_name = train_config.get("dataset")
            dataset_dir = train_config.get("dataset_dir")

        # Step 2: Build Runtime Config
        print(f"📝 Generating training configuration...")
        model_save_dir = train_exp_dir / "model"
        run_config = _build_train_config(
            config=train_config,
            ds_name=ds_name,
            dataset_dir=dataset_dir,
            output_dir=model_save_dir,
        )

        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, indent=2, sort_keys=False, allow_unicode=True)
        
        # Step 3: Execute Training
        _run_llamafactory(config_path=train_config_path, gpu_id=gpu_id)
        
        # Step 4: Cleanup Temporary Dataset
        delete_train_temp_dataset(train_exp_dir)
        print(f"🎉 Training complete! Model saved at: {model_save_dir}")

    except Exception as e:
        print(f"❌ Process interrupted: {e}")
        sys.exit(1)

def convert_lm_eval_results(lm_eval_results):
    eval_tasks = list(lm_eval_results.get("results", {}).keys())
    
    config = lm_eval_results.get("config", {})
    model_name = config.get("model") or config.get("model_args") or "unknown_model"

    custom_results = {
        "model_name": model_name,
        "eval_tasks": eval_tasks,
        "core_metrics": {},
        "eval_time": lm_eval_results.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    }

    for task, metrics in lm_eval_results.get("results", {}).items():
        core_metric = (
            metrics.get("acc_norm,none") or 
            metrics.get("acc,none") or 
            metrics.get("exact_match,none") or
            metrics.get("acc_norm") or 
            metrics.get("acc")
        )
        
        if core_metric is None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    core_metric = v
                    break
        
        custom_results["core_metrics"][task] = round(core_metric, 4) if core_metric else 0

    return custom_results

def run_evaluation(eval_exp_dir, eval_config_path, model_name, model_save_dir):
    with open(eval_config_path, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)
    if not model_name:
        assert eval_config["model_name_or_path"], \
                "❌ Error: 'model_name_or_path' must be configured in eval_config!"
        model_name = eval_config["model_name_or_path"]

    if model_save_dir is None:
        model_path = eval_config["model_path"]
    else:
        model_path = str(model_save_dir)

    print(f"📊 Starting Evaluation...")
    
    eval_tasks = eval_config["tasks"]
    num_fewshot = eval_config.get("num_fewshot", 0)
    batch_size = eval_config.get("batch_size", 1)
    device = eval_config.get("device", "cuda:0")

    model = HFLM(
        pretrained=model_name,
        peft=model_path,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=eval_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        confirm_run_unsafe_code=True,
    )
    custom_results = convert_lm_eval_results(results)
    custom_save_path = f"{eval_exp_dir}/eval_results.jsonl"

    with open(custom_save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(custom_results, ensure_ascii=False) + "\n")

    with open(eval_exp_dir / "report.txt", "w") as f:
        f.write(f"# 实验评估报告\n\n")
        f.write(f"| 任务 | 分数 |\n| :--- | :--- |\n")
        for task, score in custom_results["core_metrics"].items():
            f.write(f"| {task} | {score:.4f} |\n")

def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory One-Click Training Script")
    parser.add_argument("--train_config_path", default=None, help="Path to YAML template of training")
    parser.add_argument("--eval_config_path", default=None, help="Path to YAML template of evaluation")
    parser.add_argument(
        "--train_files", 
        default=None, 
        nargs='+',        
        help="Input train files, can pass multiple files: --train_files a.jsonl b.jsonl"
    )
    parser.add_argument("--output_root", type=str, default="./output/experiments")
    parser.add_argument("--gpu_id", default="0", help="gpu id, e.g., 0 or 0,1,2")
    parser.add_argument("--exp_id", default=None)
    args = parser.parse_args()

    # Setup environment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_id = args.exp_id
    model_save_dir = None
    model_name = None
    if args.train_files is not None and args.train_config_path is not None:
        print("🛠️  正在执行训练逻辑...")
        with open(args.train_config_path, "r", encoding="utf-8") as f:
            train_config = yaml.safe_load(f)
        model_name = train_config["model_name_or_path"]
        if not exp_id:
            exp_id = f"exp_{model_name.replace('/', '-')}_{timestamp}"
        
        train_exp_dir = Path(args.output_root) / exp_id / "train"
        train_exp_dir.mkdir(parents=True, exist_ok=True)   
        model_save_dir = train_exp_dir / "model"
        run_training(args.train_config_path, args.train_files, train_config, train_exp_dir, args.gpu_id)

    if args.eval_config_path is not None:
        print("🛠️  正在执行评估逻辑...")
        if not exp_id:
            exp_id = f"exp_{timestamp}"
        eval_exp_dir = Path(args.output_root) / exp_id / "eval"
        eval_exp_dir.mkdir(parents=True, exist_ok=True)   

        run_evaluation(eval_exp_dir, args.eval_config_path, model_name, model_save_dir)
    

if __name__ == "__main__":
    main()