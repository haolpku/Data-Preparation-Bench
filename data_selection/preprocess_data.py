import json
import os
import re
import glob
import argparse
import yaml
from pathlib import Path
import ast

def parse_messy_string(conv_str):
    fixed_str = re.sub(r'\}\s*\{', '}, {', conv_str)
    fixed_str = fixed_str.replace('\n', ' ').strip()
    
    if not fixed_str.startswith('['):
        fixed_str = '[' + fixed_str + ']'
    try:
        return ast.literal_eval(fixed_str)
    except (ValueError, SyntaxError) as e:
        print(f"解析失败: {e}")
        return None

class DataPreprocessor:
    DATASET_STRATEGIES = {
        "dolly": {
            "format_type": "mapping",
            "instruction": "instruction", "input": "context", "output": "response"
        },
        "evol_instruct": {
            "format_type": "mapping",
            "instruction": "instruction", "input": None, "output": "output"
        },
        "openhermes": {
            "format_type": "sharegpt",  
        },
        "alpaca": {
            "format_type": "mapping",
            "instruction": "instruction", "input": "input", "output": "output"
        },
        "wildchat": {
            "format_type": "sharegpt",  
        },
        "lmsys": {
            "format_type": "sharegpt",  
        },
    }
    MAX_LEN = 30000
    @staticmethod
    def preprocess_unified(input_path, output_path):
        raw_data = []
        # output_paths = [str(output_path), str(sample_path)]
        all_data = None
        path_lower = str(input_path).lower()
        sample_path = Path(output_path).parent / f"{Path(output_path).stem}_sample{Path(output_path).suffix}"
        # if os.path.exists(output_path) and os.path.exists(sample_path): return 
        if os.path.isdir(input_path):
            all_files = glob.glob(os.path.join(input_path, "*.*"))
            all_files = [f for f in all_files if 'extracted' not in f]
        else:
            all_files = [input_path]

        if len(all_files) > 1:
            all_data = []
            output_paths = []
            for idx, _ in enumerate(all_files):
                output_path_i = os.path.splitext(output_path)[0] + f"_{idx}" + os.path.splitext(output_path)[1]
                sample_path_i = os.path.splitext(output_path)[0] + f"_{idx}_sample" + os.path.splitext(output_path)[1]
                output_paths.append([str(output_path_i), str(sample_path_i)])
                all_data.append([])
            
        for i, f_path in enumerate(all_files):
            ext = os.path.splitext(f_path)[1].lower()
            
            if ext == '.csv':
                import csv
                with open(f_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        conv = row.get("conversation") or row.get("conversations")
                        if isinstance(conv, str):
                            try:
                                raw_data.append({
                                    "conversations": parse_messy_string(conv),
                                    "system": row.get("system_prompt", "")
                                })
                                if all_data is not None:
                                    all_data[i].append({
                                        "conversations": parse_messy_string(conv),
                                        "system": row.get("system_prompt", "")
                                    })
                            except: continue
                        elif isinstance(conv, list):
                            raw_data.append({"conversations": conv})
                            if all_data is not None:
                                all_data[i].append({"conversations": conv})

            elif ext in ['.json', '.jsonl']:
                with open(f_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content: continue
                    
                    if content.startswith('['):
                        raw_data.extend(json.loads(content))
                    else:
                        for line in content.splitlines():
                            if line.strip():
                                raw_data.append(json.loads(line))
            
            else:
                print(f"跳过不受支持的文件后缀: {ext} ({f_path})")

        strategy = None
        for key, val in DataPreprocessor.DATASET_STRATEGIES.items():
            if key in path_lower:
                strategy = val
                break
        if len(raw_data) > 0:  
            if not strategy:
                strategy = {"format_type": "mapping", "instruction": "instruction", "input": "input", "output": "output"}

            if strategy.get("format_type") == "sharegpt" or "conversations" in raw_data[0]:
                processed_data = DataPreprocessor._convert_sharegpt_to_alpaca(raw_data)
            else:
                processed_data = DataPreprocessor._convert_mapping_to_alpaca(raw_data, strategy)
            DataPreprocessor._save_data(processed_data, output_path, sample_path)
            
        if all_data is not None:
            for idx, data in enumerate(all_data):
                output_path_i = output_paths[idx][0]
                sample_path_i = output_paths[idx][1]
                if os.path.exists(output_path_i) and os.path.exists(sample_path_i): continue    
                if data:
                    if strategy.get("format_type") == "sharegpt" or "conversations" in data[0]:
                        processed_data_i = DataPreprocessor._convert_sharegpt_to_alpaca(data)
                    else:
                        processed_data_i = DataPreprocessor._convert_mapping_to_alpaca(data, strategy)
                    DataPreprocessor._save_data(processed_data_i, output_path_i, sample_path_i)
                    output_paths.append([str(output_path_i), str(sample_path_i)])
    
    @staticmethod
    def _save_data(processed_data, output_path, sample_path):
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in processed_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(sample_path, 'w', encoding='utf-8') as f_out:
            for item in processed_data[:10]:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Done! Total processed: {len(processed_data)}")
        
    @staticmethod
    def preprocess(cls, input_path, output_path, sample_path):
        path_lower = str(input_path).lower()
        for key, strategy_name in DataPreprocessor.DATASET_STRATEGIES.items():
            if key in path_lower:
                strategy = strategy_name
                break
            
        if not strategy:
            raise ValueError(f"Unknown dataset type")

        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                raw_data = json.loads(content)
            else:
                raw_data = [json.loads(line) for line in content.splitlines()]
                strategy = {
                    "format_type": "mapping",
                    "instruction": "instruction", "input": "input", "output": "output"
                }

        if strategy["format_type"] == "sharegpt":
            processed_data = DataPreprocessor._convert_sharegpt_to_alpaca(raw_data)
        else:
            processed_data = DataPreprocessor._convert_mapping_to_alpaca(raw_data, strategy)

        cls._save_data(processed_data, output_path, sample_path)

    @staticmethod
    def _convert_mapping_to_alpaca(raw_data, strategy):
        results = []
        for item in raw_data:
            instruction = item.get(strategy["instruction"], "")
            input_field = strategy["input"]
            user_input = item.get(input_field, "") if input_field else ""
            output = item.get(strategy["output"], "")
            context_len = len(instruction) + len(user_input) + len(output)
            if context_len > DataPreprocessor.MAX_LEN: continue
            if instruction and output:
                results.append({"instruction": instruction, "input": user_input, "output": output})
        return results

    @staticmethod
    def _convert_sharegpt_to_alpaca(sharegpt_list):
        alpaca_data = []
        for entry in sharegpt_list:
            convs = entry.get("conversations", []) or entry.get("conversation", [])
            if not convs: continue
            
            system = entry.get("system", "")
            start_idx = 0
            if isinstance(convs, str):
                # convs = re.sub(r'\}\s*\n\s*\{', '},\n{', convs)
                # convs = ast.literal_eval(convs)

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
                context_len = len(current_instruction) + len(assistant_msg)
                if context_len > DataPreprocessor.MAX_LEN: continue
                alpaca_data.append({
                    "instruction": current_instruction,
                    "input": "",
                    "output": assistant_msg
                })
                
                history += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        return alpaca_data
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess dataset to Alpaca format")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the input dataset(file or directory)")
    args = parser.parse_args()
 
    train_file = args.train_file
    input_file = Path(train_file)
    if not input_file.is_dir():
        output_file = input_file.parent / f"{input_file.stem}_extracted.jsonl"
    else:
        output_file = input_file / f"{input_file.stem}_extracted.jsonl"
    DataPreprocessor.preprocess_unified(input_file, output_file)
    print()