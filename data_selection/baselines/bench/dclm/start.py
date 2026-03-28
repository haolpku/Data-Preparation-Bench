import os
import re
import sys
import orjson
import json
import uuid
import shutil
import yaml
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.name

def run_command(command, shell=True, cwd=None):
    print(f"执行命令: {command}")
    result = subprocess.run(command, shell=shell, check=True, cwd=cwd)
    return result

def register_source(input_files, source_ref_dir, dataset_name):
    source_ref_paths = []
    for data_path in input_files:
        source_ref_path = source_ref_dir / f"{dataset_name}.json"
        entry = {
            "uuid": str(uuid.uuid4()),
            "name": dataset_name,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tokenized": False,
            "tokenizer": None,
            "size": "Unknown", 
            "dataset_url": str(source_ref_dir), 
            "sources": "custom_mixing_track"
        }

        os.makedirs(os.path.dirname(source_ref_path), exist_ok=True)
        with open(source_ref_path, 'w') as f:
            json.dump(entry, f, indent=4)
        source_ref_paths.append(str(source_ref_path))
    
    return source_ref_paths

def create_share_file(share_dir, converted_file):
    shard_files = [converted_file.name]
    shard_list_file = share_dir / "shard_list_file.txt"
    with open(shard_list_file, "w") as f:
        for s in shard_files:
            f.write(str(s) + "\n")
    return shard_list_file

def convert_to_dclm(input_file, output_file):
    input_path = Path(input_file)
    with open(input_file, "r", encoding="utf-8") as f_in, \
        open(output_file, "w", encoding="utf-8") as f_out:
        
        if input_path.suffix.lower() == ".json":
            try:
                items = json.load(f_in)
                if not isinstance(items, list):
                    items = [items] 
            except json.JSONDecodeError as e:
                print(f"❌ JSON 格式错误: {e}")
                return
        elif input_path.suffix.lower() == ".jsonl":
            items = (json.loads(line) for line in f_in if line.strip())
        else:
            print(f"⚠️ 无法识别的后缀: {input_path.suffix}，跳过该文件。")
            return

        count = 0
        for data in items:
            full_text = ""
            
            if "instruction" in data:
                inst = data.get("instruction", "")
                inp = data.get("input", "")
                out = data.get("output", "")
                
                if inp:
                    full_text = f"Instruction: {inst}\nInput: {inp}\nAnswer: {out}"
                else:
                    full_text = f"Instruction: {inst}\nAnswer: {out}"

            new_item = {
                "text": full_text,
                "metadata": {"url": data.get("url", "local_data")}, 
                "source": data.get("source", "unknown"),
                # "conversations": data.get("conversations") # 保留原始对话，方便后续还原
            }
            
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            
def preprocess_data(input_file):
    input_file = Path(input_file)
    share_dir = input_file.parent.parent.absolute() / input_file.stem / BASE_DIR
    converted_file = share_dir / "dclm.jsonl"
    os.makedirs(share_dir, exist_ok=True)
    import json

    if not os.path.exists(converted_file):
        convert_to_dclm(input_file, converted_file)
    
    shard_list_file = create_share_file(share_dir, converted_file)
    return shard_list_file, converted_file

def ensure_ray():
    try:
        subprocess.run("ray stop", shell=True, check=True)
        subprocess.run("ray start --head --port=6379 --dashboard-port=8265", shell=True, check=True)
    except:
        subprocess.run("ray start --head --port=6379 --dashboard-port=8265", shell=True, check=True)

def merge_jsonl_files(input_paths, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for file_path in input_paths:
            if not os.path.exists(file_path):
                print(f"⚠️ 警告: 找不到文件 {file_path}，跳过")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line.strip() + "\n")

def ray_processing(args, shard_list_file, converted_input_file):
    source_ref_paths = register_source(
        input_files=[converted_input_file], 
        source_ref_dir=shard_list_file.parent,
        dataset_name=args.name
    )
    data_dir = Path(args.output_dir) / args.name
    os.makedirs(data_dir, exist_ok=True)

    ensure_ray()
    source_ref_paths = ",".join(source_ref_paths)
    ray_cmd = (
        # f"{os.path.basename(sys.executable)} -m debugpy --listen 5679 --wait-for-client -m ray_processing.process "
        f"{os.path.basename(sys.executable)} -m ray_processing.process "
        f"--source_ref_paths {source_ref_paths} "
        f"--readable_name {args.name} "
        f"--output_dir {data_dir} "
        f"--config_path {args.config} "
        f"--source_name {args.name} "
        f"--shard_list_file {shard_list_file} "
        f"--overwrite "
    )
    ROOT_DIR = Path(__file__).parent.resolve()

    run_command(ray_cmd, cwd=ROOT_DIR)
    run_command("ray stop")

    config_name = os.path.basename(args.config).split(".")[0]
    file_ext = converted_input_file.suffix
    jsonl_relpath = converted_input_file.with_name(
        converted_input_file.name.replace("_processed.jsonl", ".jsonl")
    )    # To allow for overwrite if continuning from intermediate
    shard_name = jsonl_relpath.stem

    processed_file = data_dir / config_name / 'processed_data' / (str(shard_name) + f'_processed{file_ext}')
    return processed_file

def dedup(ray_processed_dir, output_dir):
    bff_binary = "./dedup/bff/target/release/bff"
    bff_dir = "./dedup/bff"
    if not os.path.exists(bff_binary):
        run_command("cd dedup/bff && cargo build --release")

    bff_cmd = (
        f"cargo run --release bff "
        f"--inputs {ray_processed_dir} "
        f"--output-directory {output_dir} "
        f"--expected-ngram-count 2000000000 "
        f"--fp-rate 0.01 "
        f"--min-ngram-size 13 "
        f"--max-ngram-size 13 "
        f"--filtering-threshold 0.8 "
        f"--remove-type old-both"
    )
    run_command(bff_cmd, cwd=bff_dir)

def update_yaml_source(file_path, new_source_name):
    file_path = Path(file_path).absolute()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if isinstance(data, list) and len(data) > 0:
        data[0]['source'] = new_source_name
    elif isinstance(data, dict):
        data['source'] = new_source_name
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

def preprocess_dclm_to_sft(input_file, output_file):
    new_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            raw_text = item.get("text", "")
            
            match = re.search(r"Instruction:(.*?)\nAnswer:(.*)", raw_text, re.DOTALL)
            if match:
                instruction = match.group(1).strip()
                output = match.group(2).strip()
                new_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output
                })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in new_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return len(new_data)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--filtered_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--name", default="my_dataset")
    parser.add_argument("--config", default="baselines/baselines_configs/dclm.yaml")

    return parser.parse_args()

def main():
    args = parse_args()
    args.name = Path(args.train_file).stem
    shard_list_file, converted_input_file = preprocess_data(args.train_file)
    # Refineweb filter
    update_yaml_source(args.config, args.name)
    processed_file = ray_processing(args, shard_list_file, converted_input_file)

    # Dedup
    dedup_input_dir = processed_file.parent.absolute()
    dedup_output_dir = (Path(args.output_dir) / args.name / "dedup").absolute()
    dedup(str(dedup_input_dir), str(dedup_output_dir))
    dedup_processed_file = dedup_output_dir / processed_file.name

    # Fasttext
    args.config = str(Path(args.config).parent /  "fasttext_filter.yaml")
    update_yaml_source(args.config, args.name)
    shard_list_file = create_share_file(dedup_output_dir, dedup_processed_file)
    processed_file = ray_processing(args, shard_list_file, dedup_processed_file)

    # convert to alpaca format
    filtered_file = Path(args.output_root) / args.filtered_file
    preprocess_dclm_to_sft(str(processed_file), str(filtered_file))

if __name__ == "__main__":
    main()