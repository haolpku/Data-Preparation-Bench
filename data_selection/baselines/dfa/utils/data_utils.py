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

def split_dataset(file_path, num_chunks, min_lines_per_chunk=100):
    p = Path(file_path)
    if not p.exists(): return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    if total_lines <= min_lines_per_chunk:
        print(f"The dataset is relatively small (with {total_lines} lines), so the split will be skipped.")
        return [str(file_path)]
    
    actual_chunks = min(num_chunks, math.ceil(total_lines / min_lines_per_chunk))
    
    if actual_chunks <= 1:
        return [str(file_path)]
    
    chunk_size = math.ceil(total_lines / actual_chunks)
    chunk_paths = []
    
    for i in range(actual_chunks):
        chunk_lines = lines[i*chunk_size : (i+1)*chunk_size]
        if not chunk_lines: continue
        
        c_path = p.parent / f"{p.stem}_chunk_{i}{p.suffix}"
        with open(c_path, 'w', encoding='utf-8') as f:
            f.writelines(chunk_lines)
        chunk_paths.append(str(c_path))
    
    print(f"The dataset has been divided into {len(chunk_paths)} chunks (each chunk containing approximately {chunk_size} lines).")
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
            chunk_files = cache_path.glob(f"*_[0-9]*_step{step}*.jsonl")
            for f in sorted(chunk_files):
                if f.name == final_out_path.name:
                    continue
                with open(f, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
        
        merged_paths[int(step)] = str(final_out_path.name)
    return merged_paths
