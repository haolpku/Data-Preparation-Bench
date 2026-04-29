import json
import os
import re
import glob
import argparse
import ast
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AlpacaExample:
    instruction: str
    input: str
    output: str

class DatasetConverter:
    MAX_LENGTH = 30000

    @staticmethod
    def clean_text(s: Any) -> str:
        if s is None:
            return ""
        if isinstance(s, (int, float)):
            return str(s)
        s = re.sub(r'\s+', ' ', str(s)).strip()
        return s

    @staticmethod
    def parse_any(data: Any) -> List[AlpacaExample]:
        if isinstance(data, dict):
            data = [data]
        examples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if DatasetConverter._is_sharegpt_like(item):
                examples.extend(DatasetConverter._parse_sharegpt(item))
            else:
                examples.extend(DatasetConverter._parse_instruct(item))
        return examples

    @staticmethod
    def _is_sharegpt_like(item: Dict) -> bool:
        return any(k in item for k in ["conversations", "conversation", "messages", "chat", "dialog", "turns"])

    @staticmethod
    def _parse_instruct(item: Dict[str, Any]) -> List[AlpacaExample]:
        instruction = DatasetConverter.clean_text(
            item.get("instruction") or item.get("prompt") or item.get("question") or item.get("user") or ""
        )
        input_text = DatasetConverter.clean_text(item.get("input") or item.get("context") or "")
        output_text = DatasetConverter.clean_text(
            item.get("output") or item.get("response") or item.get("answer") or item.get("assistant") or ""
        )
        if not instruction or not output_text:
            return []
        if len(instruction + input_text + output_text) > DatasetConverter.MAX_LENGTH:
            return []
        return [AlpacaExample(instruction=instruction, input=input_text, output=output_text)]

    @staticmethod
    def _parse_sharegpt(item: Dict[str, Any]) -> List[AlpacaExample]:
        # convs = item.get("conversations") or item.get("conversation") or item.get("messages") or item.get("chat") or []
        convs = (item.get("conversations") if item.get("conversations") is not None else
         item.get("conversation") if item.get("conversation") is not None else
         item.get("messages") if item.get("messages") is not None else
         item.get("chat") if item.get("chat") is not None else [])
        system = DatasetConverter.clean_text(item.get("system", ""))

        if isinstance(convs, str):
            try:
                convs = ast.literal_eval(re.sub(r'}\s*{', '},{', convs))
            except:
                return []
        if hasattr(convs, 'tolist'):
            convs = convs.tolist()
        if not isinstance(convs, list):
            return []

        examples = []
        history, system = "", ""
        start_idx = 0

        if convs[0].get("from") == "system":
            system = convs[0].get("value", "")
            start_idx = 1 
                
        valid_convs = convs[start_idx:]
        for i in range(0, len(valid_convs) - 1, 2):
            user = valid_convs[i]
            assistant = valid_convs[i+1]
            role = (user.get("from") or user.get("role") or "").lower()
            if role not in ["human", "user"]:
                continue
            u = DatasetConverter.clean_text(user.get("content") or user.get("value"))
            a = DatasetConverter.clean_text(assistant.get("content") or assistant.get("value"))
            if not u or not a:
                continue
            instr = f"{system}\n{history}\n{u}".strip()
            examples.append(AlpacaExample(instruction=instr, input="", output=a))
            history += f"User: {u}\nAssistant: {a}\n"
        return [e for e in examples if len(e.instruction + e.output) < DatasetConverter.MAX_LENGTH]

class DataPreprocessor:
    @staticmethod
    def split_dataset(file_path, num_chunks, min_lines_per_chunk=100):
        """
        智能切分数据集
        :param min_lines_per_chunk: 每个分块至少要包含的行数，防止分块过细
        """
        p = Path(file_path)
        if not p.exists(): return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # --- 核心改进逻辑 ---
        # 1. 如果总行数还没达到一个分块的最小要求，直接返回原文件路径，不分割
        if total_lines <= min_lines_per_chunk:
            print(f"数据集较小 ({total_lines} 行)，跳过分割。")
            return [str(file_path)]
        
        # 2. 重新计算实际需要的分块数量
        # 比如你想分 32 个块，但总共只有 320 行，按阈值 100 行算，实际只需分 3 块
        actual_chunks = min(num_chunks, math.ceil(total_lines / min_lines_per_chunk))
        
        # 如果算出来只需 1 个块，也直接返回原路径
        if actual_chunks <= 1:
            return [str(file_path)]
        
        # --- 开始切分 ---
        chunk_size = math.ceil(total_lines / actual_chunks)
        chunk_paths = []
        
        for i in range(actual_chunks):
            chunk_lines = lines[i*chunk_size : (i+1)*chunk_size]
            if not chunk_lines: continue
            
            c_path = p.parent / f"{p.stem}_chunk_{i}{p.suffix}"
            with open(c_path, 'w', encoding='utf-8') as f:
                f.writelines(chunk_lines)
            chunk_paths.append(str(c_path))
        
        print(f"数据集已切分为 {len(chunk_paths)} 个分块 (每块约 {chunk_size} 行)。")
        return chunk_paths
  
    @staticmethod
    def load_file(path: Path) -> List[Dict]:
        try:
            if path.suffix == ".json":
                return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            elif path.suffix == ".jsonl":
                return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            elif path.suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
                return df.to_dict('records')
            elif path.suffix == ".parquet":
                try:
                    import pandas as pd
                except ImportError:
                    print(f"❌ 缺少依赖：请运行 'pip install pandas pyarrow' 以处理 parquet 文件")
                    return []
                df = pd.read_parquet(path)
                return df.to_dict('records')
        except Exception as e:
            print(f"⚠️ 读取失败 {path.name}: {str(e)[:40]}")
        return []

    @staticmethod
    def save_results(examples: List[AlpacaExample], out_path: Path):
        out = [json.dumps({"instruction": e.instruction, "input": e.input, "output": e.output}, ensure_ascii=False) for e in examples]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out), encoding="utf-8")
        sample = out_path.parent / f"{out_path.stem}_sample.jsonl"
        sample.write_text("\n".join(out[:10]), encoding="utf-8")
        print(f"✅ 保存：{out_path.name} ({len(examples)} 条)")

    @staticmethod
    def save_results_chunk(examples1: List[AlpacaExample], out_path1: Path):
        import torch
        num_gpus = torch.cuda.device_count() or 1
        num_samples = len(examples1)
        chunk_size = (num_samples + num_gpus - 1) // num_gpus
        for i in range(num_gpus):
            out_path = out_path1.parent / f"{out_path1.stem}_{i}.jsonl"
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            examples = examples1[start_idx:end_idx]
            out = [json.dumps({"instruction": e.instruction, "input": e.input, "output": e.output}, ensure_ascii=False) for e in examples]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(out), encoding="utf-8")
            sample = out_path.parent / f"{out_path.stem}_sample.jsonl"
            sample.write_text("\n".join(out[:10]), encoding="utf-8")
            print(f"✅ 保存：{out_path.name} ({len(examples)} 条)")

    @staticmethod
    def process(input_path: Path):
        files = [input_path] if input_path.is_file() else []
        if input_path.is_dir():
            for ext in ["*.json", "*.jsonl", "*.csv", "*.parquet"]:
                files.extend(input_path.rglob(ext))

        for f in files:
            rel = f.relative_to(input_path.parent if input_path.is_file() else input_path)
            if input_path.is_dir():
                out_path = input_path / "processed" / rel.with_name(f"{f.stem}_extracted.jsonl")
            else:
                out_path = input_path.parent / "processed" / rel.with_name(f"{f.stem}_extracted.jsonl")
                
            print(f"\n处理：{f}")
            data = DataPreprocessor.load_file(f)
            examples = DatasetConverter.parse_any(data)
            DataPreprocessor.save_results_chunk(examples, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量转 Alpaca 格式，输出独立目录，不污染原始数据")
    parser.add_argument("--train_file", type=str, required=True, help="单个文件 或 整个文件夹")
    args = parser.parse_args()

    DataPreprocessor.process(Path(args.train_file))
