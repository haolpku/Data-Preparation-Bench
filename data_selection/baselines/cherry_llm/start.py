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
def run_command(cmd, env_update=None, cwd=None):
    current_env = os.environ.copy()
    if env_update:
        current_env.update(env_update)

    import shlex
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    try:
        subprocess.run(cmd_list, shell=False, check=True, env=current_env, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)

def convert_sharegpt_to_alpaca(sharegpt_list):
    alpaca_data = []
    for entry in sharegpt_list:
        convs = entry.get("conversations", [])
        if not convs:
            continue
            
        system = entry.get("system", "")
        start_idx = 0
        
        if convs[0].get("from") == "system":
            system = convs[0].get("value", "")
            start_idx = 1 
            
        history = ""
        
        for i in range(start_idx, len(convs) - 1, 2):
            if convs[i]["from"] not in ["human", "user"]:
                continue
            
            user_msg = convs[i]["value"]
            assistant_msg = convs[i+1]["value"]
            
            current_instruction = f"{system}\n{history}User: {user_msg}".strip()
            alpaca_data.append({
                "instruction": current_instruction,
                "input": "",
                "output": assistant_msg
            })
            
            history += f"User: {user_msg}\nAssistant: {assistant_msg}\n"

   
    return alpaca_data

def judge_and_convert(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list) and len(raw_data) > 0 and "conversations" in raw_data[0]:
        print("💡 检测到 ShareGPT 格式（含多轮/System）")
        output_path = Path(file_path).parent / f"{Path(file_path).stem}_alpaca.json"
        if os.path.exists(output_path): return output_path
        alpaca_data = convert_sharegpt_to_alpaca(raw_data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=4)
        return output_path
    else:
        print("💡 检测到 Alpaca 格式")
        return file_path

def transfer_json(data_path):
    input_path = Path(data_path)
    output_dir = Path(data_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / input_path.with_suffix('.json').name
    
    all_alpaca_data = []
    if os.path.exists(output_path):
        print(f"💡 已检测到转换后的 JSON 文件，直接使用: {output_path}")
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
    parser.add_argument("--prompt_type", default="alpaca", choices=["alpaca", "wiz"], help="Prompt 模板类型")
    parser.add_argument("--output_root", default="", required=True)
    parser.add_argument("--train_config_path", default="", required=True)
    parser.add_argument("--template", default="", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")
    BASE_NAME = str(Path(args.train_file).parent.stem)

    # output_dir = os.path.join(args.output_root, f"{BASE_NAME}_{timestamp}")
    # os.makedirs(output_dir, exist_ok=True)
    
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
        ray_cmd = (
            f"{os.path.basename(sys.executable)} cherry_seletion/data_analysis.py "
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
            f"conda run -n bench python {train_entry} "
            f"--train_config_path {args.train_config_path} "
            f"--train_file {PRE_JSON} "
            f"--output_root {PRE_MODEL_DIR} "
            f"--mode train ",
            cwd=PROJECT_ROOT
        )
    
    # Stage 4: Export/Merge Model
    train_config_path = Path(PROJECT_ROOT) / args.train_config_path
    with open(train_config_path, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    model_name_or_path = train_config["model_name_or_path"]

    PRE_MODEL_DIR = Path(PRE_MODEL_DIR) / "model"
    run_command(    
        "conda run -n bench llamafactory-cli export "
        f"--model_name_or_path {model_name_or_path} "
        f"--adapter_name_or_path {PRE_MODEL_DIR} "
        f"--export_dir {MERGE_MODEL_DIR} "
        f"--template {args.template} ",
        cwd=PROJECT_ROOT
    )

    current_cache_dir = constants.HF_HUB_CACHE
    tokenizer_files = ["vocab.json", "tokenizer.json", "tokenizer_config.json"]
    repo_folder = f"models--{args.model_path.replace('/', '--')}"
    snapshots_path = os.path.join(current_cache_dir, repo_folder, "snapshots")
    
    if os.path.exists(snapshots_path):
        subdirs = [os.path.join(snapshots_path, d) for d in os.listdir(snapshots_path) 
                if os.path.isdir(os.path.join(snapshots_path, d))]
        if subdirs:
            real_model_path = max(subdirs, key=os.path.getmtime)
    # print(real_model_path)
    # for file in tokenizer_files:
    #     src = os.path.join(real_model_path, file)
    #     dst = os.path.join(MERGE_MODEL_DIR)
    #     os.makedirs(dst, exist_ok=True)
    #     if os.path.exists(src):
    #         shutil.copy2(src, dst)

    # Stage 5: Final Cherry Analysis
    eval_model = MERGE_MODEL_DIR if os.path.exists(MERGE_MODEL_DIR) else args.model_path
    if not os.path.exists(CHERRY_PT):
        ray_cmd = (
            f"{os.path.basename(sys.executable)} cherry_seletion/data_analysis.py "
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

    print(f"🍒 樱桃数据集已保存至: {os.path.abspath(FINAL_OUTPUT)}")

if __name__ == "__main__":
    main()