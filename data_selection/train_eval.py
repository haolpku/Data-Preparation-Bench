import argparse
import os
import yaml
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

def _register_dataset(ds_name, file_path, column_mapping):
    """Automatically generates/updates dataset_info.json in the input data directory."""
    dataset_dir = Path(file_path).parent  
    info_path = dataset_dir / "dataset_info.json"
    
    if not info_path.exists():
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
    
    with open(info_path, "r", encoding="utf-8") as f:
        ds_registry = json.load(f)
    
    ds_registry[ds_name] = {
        "file_name": str(Path(file_path).name),
        "columns": column_mapping,
        "register_time": datetime.now().isoformat()
    }
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(ds_registry, f, indent=2, ensure_ascii=False)
    
    return str(dataset_dir)

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
    # train_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} llamafactory-cli train {config_path}"
    if "," in str(gpu_id):
        nproc = len(str(gpu_id).split(","))
        train_cmd = [
            "torchrun", 
            f"--nproc_per_node={nproc}", 
            "third_party/LlamaFactory/src/train.py", # 指向子模块路径
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

def run_training(train_config_path, train_file, train_config, filter_id, save_dir, gpu_id=0):
    try:
        # Step 1: Register Dataset
        print(f"📦 Registering dataset: {filter_id}...")
        if filter_id is not None:
            ds_name = filter_id
            dataset_dir = _register_dataset(
                ds_name=filter_id,
                file_path=train_file,
                column_mapping={"prompt": "instruction", "query": "input", "response": "output"}
            )
        else:
            assert train_config.get("dataset") and train_config.get("dataset_dir"), \
                "❌ Error: 'dataset' and 'dataset_dir' must be configured in train_config!"
            ds_name = train_config.get("dataset")
            dataset_dir = train_config.get("dataset_dir")

        # Step 2: Build Runtime Config
        print(f"📝 Generating training configuration...")
        run_config = _build_train_config(
            config=train_config,
            ds_name=ds_name,
            dataset_dir=dataset_dir,
            output_dir=save_dir,
        )

        with open(train_config_path, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, indent=2, sort_keys=False, allow_unicode=True)
        
        # Step 3: Execute Training
        _run_llamafactory(config_path=train_config_path, gpu_id=gpu_id)
        
        print(f"🎉 Training complete! Model saved at: {save_dir}")

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

def run_evaluation(args, config_path, model_name, model_save_dir):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_save_dir is None and model_name is None:
        model_path = config["model_path"]
        model_name = config["model_name"]

        model_id = Path(model_path).stem
        save_dir =  Path(args.output_root) / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_path = model_save_dir
        save_dir = model_save_dir.parent / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Starting Evaluation...")
    
    eval_tasks = config["tasks"]
    num_fewshot = config.get("num_fewshot", 0)
    batch_size = config.get("batch_size", 1)
    device = config.get("device", "cuda:0")

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
    custom_save_path = f"{save_dir}/eval_results.jsonl"

    with open(custom_save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(custom_results, ensure_ascii=False) + "\n")

    with open(save_dir / "report.txt", "w") as f:
        f.write(f"# 实验评估报告\n\n")
        f.write(f"| 任务 | 分数 |\n| :--- | :--- |\n")
        for task, score in custom_results["core_metrics"].items():
            f.write(f"| {task} | {score:.4f} |\n")

def main():
    parser = argparse.ArgumentParser(description="LLaMA-Factory One-Click Training Script")
    parser.add_argument("--train_config_path", default="configs/train_qwen2.5.yaml", help="Path to YAML template of training")
    parser.add_argument("--eval_config_path", default="configs/eval.yaml", help="Path to YAML template of evaluation")
    parser.add_argument("--train_file", default=None, help="Root directory for outputs")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--mode", choices=["train,eval", "train", "eval"], default="train,eval")
    
    args = parser.parse_args()

    # Setup environment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = Path(args.train_config_path).stem
    if args.train_file is not None:
        filter_id = Path(args.train_file).parent.stem
    else:
        filter_id = None

    model_save_dir = None
    model_name = None
    if "train" in args.mode:
        print("🛠️  正在执行训练逻辑...")
        with open(args.train_config_path, "r", encoding="utf-8") as f:
            train_config = yaml.safe_load(f)

        model_save_dir = Path(args.output_root) / "model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model_name = train_config["model_name_or_path"]
        # run_training(args, train_config, filter_id, model_save_dir)
        run_training(args.train_config_path, args.train_file, train_config, filter_id, model_save_dir, args.gpu_id)

    if "eval" in args.mode:
        print("🛠️  正在执行评估逻辑...")
        run_evaluation(args, args.eval_config_path, model_name, model_save_dir)
    

if __name__ == "__main__":
    main()