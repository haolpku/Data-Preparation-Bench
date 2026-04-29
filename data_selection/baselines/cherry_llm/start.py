import os
import json
import subprocess
import sys
import yaml
import shutil
import argparse
from pathlib import Path

from huggingface_hub import constants, try_to_load_from_cache

CHERRY_LLM_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Cherry_LLM"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent

def run_command(cmd, env_update=None, cwd=None):
    current_env = os.environ.copy()
    if env_update:
        current_env.update(env_update)

    import shlex
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    result = subprocess.run(cmd_list, shell=False, check=True, cwd=cwd, env=current_env)
    return result

def transfer_json(data_path):
    input_path = Path(data_path)
    output_dir = Path(data_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / input_path.with_suffix('.json').name
    
    all_alpaca_data = []
    if os.path.exists(output_path):
        print(f"💡 The converted JSON file has been detected. Use directly: {output_path}")
        return str(output_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.suffix == '.jsonl':
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    all_alpaca_data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_alpaca_data, f_out, indent=4, ensure_ascii=False)
        
    return str(output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Cherry Data (IFD) 数据筛选自动化流水线")
    parser.add_argument("--train_file", required=True, help="原始训练数据集路径 (.json)")
    parser.add_argument("--model_path", required=True, help="基座模型路径 (如 Llama-2-7b-hf)")
    parser.add_argument("--output_dir", default="./cherry_results", help="结果输出目录")
    parser.add_argument("--filtered_file", required=True, help="结果输出目录")
    parser.add_argument("--sample_rate", default="0.1", help="最终筛选比例 (默认 0.1 即 10%)")
    parser.add_argument("--prompt_type", default="alpaca", choices=["alpaca"], help="Prompt 模板类型")
    parser.add_argument("--output_root", default="", required=True)
    parser.add_argument("--train_config_path", default="", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")
    BASE_NAME = str(Path(args.train_file).parent.stem)
    
    output_dir = Path(args.output_root)
    PRE_PT = (output_dir / "pre.pt").absolute()
    PRE_JSON = (output_dir / "pre_experience.json").absolute()
    CHERRY_PT = (output_dir / "cherry_analysis.pt").absolute()
    PRE_MODEL_DIR = (output_dir / "pre_experienced_model").absolute()
    MERGE_MODEL_DIR = (output_dir / "pre_experienced_model" / "merge").absolute()

    data_path = args.train_file 
    data_path = transfer_json(data_path)

    # Stage 1: Pre-Analysis
    if not os.path.exists(PRE_PT):
        exec_file = str(BASE_DIR / "data_analysis.py")
        ray_cmd = (
            f"{os.path.basename(sys.executable)} {exec_file} "
            f"--data_path {data_path} "
            f"--save_path {PRE_PT} "
            f"--model_name_or_path {args.model_path} "
            f"--mod pre "
            f"--prompt {args.prompt_type} "
        )

        run_command(ray_cmd, cwd=CHERRY_LLM_ROOT) 

    # # Stage 2: Clustering
    if not os.path.exists(PRE_JSON):
        ray_cmd = (
            f"{os.path.basename(sys.executable)} cherry_seletion/data_by_cluster.py "
            f"--pt_data_path {PRE_PT} "
            f"--json_data_path {data_path} "
            f"--json_save_path {PRE_JSON} "
            f"--sample_num 10 "
            f"--kmeans_num_clusters 5 "
        )
        run_command(ray_cmd, cwd=CHERRY_LLM_ROOT)

    train_entry = Path(PROJECT_ROOT) / "train_eval.py"
    # Stage 3: Training Pre-experienced Model
    if not os.path.exists(PRE_MODEL_DIR):
        run_command(
            f"{os.path.basename(sys.executable)} {train_entry} "
            f"--train_config_path {args.train_config_path} "
            f"--train_files {PRE_JSON} "
            f"--output_root {PRE_MODEL_DIR} "
            f"--exp_id cherry_llm ",
            cwd=PROJECT_ROOT
        )
    
    # Stage 4: Export/Merge Model
    train_config_path = Path(PROJECT_ROOT) / args.train_config_path
    with open(train_config_path, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    model_name_or_path = train_config["model_name_or_path"]

    PRE_MODEL_DIR = Path(PRE_MODEL_DIR) / "cherry_llm/train/model"
    run_command(    
        "llamafactory-cli export "
        f"--model_name_or_path {model_name_or_path} "
        f"--adapter_name_or_path {PRE_MODEL_DIR} "
        f"--export_dir {MERGE_MODEL_DIR} "
        , cwd=PROJECT_ROOT
    )

    # Stage 5: Final Cherry Analysis
    eval_model = MERGE_MODEL_DIR if os.path.exists(MERGE_MODEL_DIR) else args.model_path
    if not os.path.exists(CHERRY_PT):
        exec_file = str(BASE_DIR / "data_analysis.py")
        ray_cmd = (
            f"{os.path.basename(sys.executable)} {exec_file} "
            f"--data_path {data_path} "
            f"--save_path {CHERRY_PT} "
            f"--model_name_or_path {eval_model} "
            f"--mod cherry "
            f"--prompt {args.prompt_type} "
        )
        run_command(ray_cmd, cwd=CHERRY_LLM_ROOT)

    # # Stage 6: IFD Selection
    FINAL_OUTPUT = os.path.join(args.output_root, args.filtered_file)
    ray_cmd = (
        f"{os.path.basename(sys.executable)} cherry_seletion/data_by_IFD.py "
        f"--pt_data_path {CHERRY_PT} "
        f"--model_name_or_path {eval_model} "
        f"--json_data_path {data_path} "
        f"--json_save_path {FINAL_OUTPUT} "
        f"--sample_rate {args.sample_rate} "
        f"--prompt {args.prompt_type} "
    )
    run_command(ray_cmd, cwd=CHERRY_LLM_ROOT)

    print(f"🍒 The cherry dataset has been saved to:: {os.path.abspath(FINAL_OUTPUT)}")

if __name__ == "__main__":
    main()