import re
import json
import time
import random
import concurrent.futures
import numpy as np
import requests
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


DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
DATA_PATH = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
PROMPT_TEMPLATE_PATH = './instructions/oracle_diagnose.txt'

# =====================
# MODEL API INTERFACES
# =====================



def query_my_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    client = OpenAI(
        base_url=NEW_BASE_URL,
        api_key= NEW_API_KEY
    )
    
    try:
        response = client.chat.completions.create(
            model= NEW_MODEL_NAME ,
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
        if "rate limit" in str(e).lower():
            time.sleep(30)
        return None, None
    

# =====================
# DATA PROCESSING
# =====================

def load_instruction(template_path):
    """Load instruction template from file"""
    try:
        with open(template_path, encoding='utf-8') as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading instruction template: {e}")
        return None

def load_data(data_path):
    """Load and parse case data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def save_results(data, filename):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

# =====================
# INFERENCE PROCESSORS
# =====================


    
def process_my_model_data(data_id, data_item, prompt_template):

    try:
        result = {}

        patient_case = data_item.get('generate_case', {}).get('case_summary', "")
        
        if not patient_case:
            return data_id, None
            
        prompt = prompt_template.format(case=patient_case)
        result['input'] = prompt
        
        answer, reasoning = query_my_model(prompt)
        
        if answer is not None:
            result['out_answer'] = answer
            result['out_reasoning'] = reasoning
            return data_id, result
        else:
            return data_id, None
            
    except Exception as e:
        print(f"Error processing data {data_id}: {e}")
        return data_id, None

# =====================
# MAIN INFERENCE RUNNERS
# =====================

def run_inference_with_model(
    process_func, 
    model_name, 
    output_filename, 
    max_workers=8
):
    """Generic function to run inference with any model"""
    print(f"Running inference with {model_name} model")
    
    prompt_template = load_instruction(PROMPT_TEMPLATE_PATH)
    data = load_data(DATA_PATH)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_id, data_item in data.items():
            futures.append(executor.submit(process_func, data_id, data_item, prompt_template))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing with {model_name}"):
            data_id, result = future.result()
            if result is not None:
                data[data_id][model_name.lower()] = result
    save_results(data, output_filename)


def run_oracle_inference(max_workers=4):

    print(f"Running Oracle Diagnosis with {NEW_MODEL_NAME}")
    

    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not IS_FULL_EVAL:
        items = list(data.items())[:5]
    else:
        items = list(data.items())
    
    results_map = {}
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_my_model_data, d_id, d_item, prompt_template): d_id 
            for d_id, d_item in items
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(items), desc="Inference"):
            data_id, result = future.result()
            if result:
  
                data[data_id][f'result'] = result


    output_file = f"../../data/InferenceResults/{NEW_MODEL_NAME}_oracle_diagnosis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved results to {output_file}")

if __name__ == "__main__":

    run_oracle_inference(max_workers=4)