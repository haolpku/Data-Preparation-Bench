import json
import os
import re

import ast
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

    @staticmethod
    def preprocess(input_path, output_path):
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

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in processed_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return len(processed_data)

    @staticmethod
    def _convert_mapping_to_alpaca(raw_data, strategy):
        results = []
        for item in raw_data:
            instruction = item.get(strategy["instruction"], "")
            input_field = strategy["input"]
            user_input = item.get(input_field, "") if input_field else ""
            output = item.get(strategy["output"], "")
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
                alpaca_data.append({
                    "instruction": current_instruction,
                    "input": "",
                    "output": assistant_msg
                })
                
                history += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        return alpaca_data