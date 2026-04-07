import re
import json
import time
import random
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# ========================================================
NEW_API_KEY = "sk-dummy"
NEW_BASE_URL = "http://XXX/v1"
NEW_MODEL_NAME = "gpt-4o"
IS_FULL_EVAL = False 
REFER_MODEL_NAME = "gpt-4o"                                           
REFER_API_BASE_URL = "http://XXX/v1"                  
REFER_API_KEY = "sk-dummy"  
# ========================================================

# Path constants
DATA_PATH = '../../data/MedRBench/treatment_496_cases_with_rare_disease_165.json'
TREATMENT_PROMPT_PATH = './instructions/treatment_plan_prompt.txt'
OUTPUT_DIR = "oracle_treatment_plan"

# Default settings
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"

# ======================
# Utility Functions
# ======================

def load_instruction(txt_path):
    """Load prompt template from file"""
    try:
        with open(txt_path, encoding='utf-8') as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading prompt template from {txt_path}: {e}")
        return None

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created output directory: {directory}")

def save_results(data, filename):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def load_data():
    """Load data"""
    # Load full dataset
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        json_datas = json.load(f)
    return json_datas
    

# ======================
# Model API Interfaces
# ======================

    
def query_my_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):

    client = OpenAI(
        base_url=NEW_BASE_URL,
        api_key= NEW_API_KEY
    )
    try:
        response = client.chat.completions.create(
            model= NEW_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        
   
        reasoning = ""
        if "</think>" in content:
            parts = content.split("</think>")
            reasoning = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
        else:
            answer = content
        return answer, reasoning
    except Exception as e:
        print(f"API Error: {e}")
        return None, None    
    
    
# ======================
# Data Processing Functions
# ======================

    
def process_my_model_treatment_data(data_id, data_item, prompt_template):

    try:
        result = {}

        patient_case = data_item.get('generate_case', {}).get('case_summary', "")
        
        if not patient_case:
            return data_id, None
            
        prompt = prompt_template.format(case=patient_case)
        answer, reasoning = query_my_model(prompt)
        
        if answer is not None:
            result['content'] = answer
            result['reasoning'] = reasoning
            return data_id, result
    except Exception as e:
        print(f"Error processing data {data_id}: {e}")
    return data_id, None

# ======================
# Main Inference Functions
# ======================

def run_inference_with_model(model_name, process_func, max_workers=8):
    """Generic function to run inference with any model"""
    print(f"Running treatment inference with {model_name}")
    
    # Load prompt template
    prompt_template = load_instruction(TREATMENT_PROMPT_PATH)
    if not prompt_template:
        print(f"Error: Failed to load prompt template")
        return
    
    # Load data
    datas = load_data()
    if not datas:
        print(f"Error: No data to process")
        return
    
    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)
    
    # Process data with the specified model using concurrent processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_id, data in datas.items():
            futures.append(executor.submit(process_func, data_id, data, prompt_template))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing with {model_name}"):
            data_id, result = future.result()
            if result is not None:
                datas[data_id][model_name.lower()] = result

    # Save results
    filename = f"{OUTPUT_DIR}/{model_name.lower()}_all_output.json"
    save_results(datas, filename)

def run_oracle_treatment_inference(max_workers=4):

    print(f"Running Oracle Treatment Planning with {NEW_MODEL_NAME}")
    

    prompt_template = load_instruction(TREATMENT_PROMPT_PATH)
    

    data = load_data()
    

    if not IS_FULL_EVAL:
        items = list(data.items())[:5]
    else:
        items = list(data.items())

    

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_my_model_treatment_data, d_id, d_item, prompt_template): d_id 
            for d_id, d_item in items
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(items), desc="Treatment Inference"):
            data_id, result = future.result()
            if result:

                data[data_id][f'result'] = result


    ensure_output_dir(OUTPUT_DIR)
    output_file = f"../../data/InferenceResults/{NEW_MODEL_NAME}_oracle_treatment.json"
    save_results(data, output_file)
    print(f"Successfully saved treatment results to {output_file}")


if __name__ == "__main__":
    # Run inference with all models
    run_oracle_treatment_inference(max_workers=4)