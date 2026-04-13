import os
import re
import json
import math
import glob
import ast
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def save_as_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def split_dataset(file_path, num_chunks):
    """智能切分数据集 (支持 jsonl/csv/json)"""
    p = Path(file_path)
    if not p.exists(): return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    chunk_size = math.ceil(len(lines) / num_chunks)
    chunk_paths = []
    for i in range(num_chunks):
        chunk_lines = lines[i*chunk_size : (i+1)*chunk_size]
        if not chunk_lines: continue
        c_path = p.parent / f"{p.stem}_chunk_{i}{p.suffix}"
        with open(c_path, 'w', encoding='utf-8') as f:
            f.writelines(chunk_lines)
        chunk_paths.append(str(c_path))
    return chunk_paths

def merge_jsonl_results(cache_dir, output_file_pattern, dataset_name):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Warning: Cache directory {cache_dir} does not exist.")
        return {}

    all_files = [f.name for f in cache_path.glob("*.jsonl")]
    steps = sorted(list(set(re.findall(r'_step(\d+)', " ".join(all_files)))))
    
    merged_paths = {}
    for step in steps:
        final_out_path = Path(output_file_pattern.format(name=dataset_name, step=step))
        final_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Merging Step {step} results to {final_out_path}...")
        with open(final_out_path, 'w', encoding='utf-8') as outfile:
            # 只合并属于当前 step 的分片文件
            chunk_files = cache_path.glob(f"*_step{step}*.jsonl")
            for f in sorted(chunk_files):
                # 排除掉已经是合并后的文件（如果有的话）
                if f.name == final_out_path.name:
                    continue
                with open(f, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
        
        merged_paths[step] = str(final_out_path.name)
    return merged_paths

def extract_conversations_to_json(dataframe, output_path, sample_path=None, max_char_limit=30000):
    alpaca_data = []
    is_sharegpt = 'conversations' in dataframe.columns and isinstance(dataframe['conversations'].iloc[0], list)
    has_alpaca_keys = 'instruction' in dataframe.columns and \
                  any(k in dataframe.columns for k in ['output', 'response', 'answer'])
    skipped_long = 0
    if is_sharegpt:
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Converting"):
            convs = row.get("conversations", []) or row.get("conversation", [])
            if not convs: continue
            
            system = row.get("system", "")
            start_idx = 0
            if isinstance(convs, str):
                fixed = re.sub(r'\}\s*\{', '},{', convs)
                fixed = fixed.replace('\n', '')
                convs = ast.literal_eval(fixed)

            if convs[0].get("from") == "system":
                system = convs[0].get("value", "")
                start_idx = 1 
                
            history = ""
            from_key = ["from", "role"]
            content_key = ["content", "value"]
            for i in range(start_idx, len(convs) - 1, 2):
                user_conv = convs[i]
                assistant_conv = convs[i+1]
                
                user_role = next((user_conv.get(k) for k in from_key if k in user_conv), None)
                if user_role not in ["human", "user"]:
                    continue
                
                user_msg = next((user_conv.get(k) for k in content_key if user_conv.get(k)), None)
                assistant_msg = next((assistant_conv.get(k) for k in content_key if assistant_conv.get(k)), None)
                
                if not user_msg or not assistant_msg:
                    continue
    
                current_instruction = f"{system}\n{history}User: {user_msg}".strip()
                current_total_len = len(current_instruction) + len(assistant_msg)
                
                if current_total_len > max_char_limit:
                    skipped_long += 1
                    continue
                alpaca_data.append({
                    "instruction": current_instruction,
                    "input": "",
                    "output": assistant_msg
                })
                
                history += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
                
            # convs = row.get('conversations', [])
            # system_prompt = row.get('system', "")
            
            # for i in range(0, len(convs) - 1, 2):
            #     if convs[i]['from'] == 'human' and convs[i+1]['from'] == 'gpt':
            #         inst = str(convs[i]['value'])
            #         out = str(convs[i+1]['value'])
            #         current_total_len = len(inst) + len(out) + len(system_prompt)
                
            #         if current_total_len > max_char_limit:
            #             skipped_long += 1
            #             continue
            #         entry = {
            #             "instruction": inst,
            #             "input": system_prompt if i == 0 else "", # 仅在第一轮保留 system
            #             "output": out
            #         }
            #         # if 'id' in row: entry['id'] = f"{row['id']}_t{i//2}"
            #         alpaca_data.append(entry)
    elif has_alpaca_keys:
        for _, row in dataframe.iterrows():
            instruction_keys = ['instruction', 'prompt', 'question']
            input_keys = ['input', 'context']
            output_keys = ['output', 'response', 'answer', 'target']

            instruction = next((str(row[k]) for k in instruction_keys if k in row and pd.notna(row[k])), "")
            input_data = next((str(row[k]) for k in input_keys if k in row and pd.notna(row[k])), "")
            output = next((str(row[k]) for k in output_keys if k in row and pd.notna(row[k])), "")

            current_total_len = len(instruction) + len(input_data) + len(output)
            
            if current_total_len > max_char_limit:
                skipped_long += 1
                continue
            alpaca_data.append({
                "instruction": instruction,
                "input": input_data,
                "output": output
            })
    else:
        raise ValueError("无法识别的数据格式：既没有 'conversations' 列，也没有 'instruction'/'output' 列。")

    print(f"{skipped_long}条数据超过了最大限制")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_as_jsonl(alpaca_data, output_path)
    save_as_jsonl(alpaca_data[:10], sample_path)
    print(f"处理完成！输出文件：{output_path}，总条数：{len(alpaca_data)}")