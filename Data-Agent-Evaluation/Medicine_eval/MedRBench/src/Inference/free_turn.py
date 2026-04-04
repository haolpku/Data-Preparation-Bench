import os
import json
import re
import time
import random
import requests
import numpy as np
import tqdm
from multiprocessing import Pool
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
DATA_PATH = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
INSTRUCTION_DIR = "instructions"
INITIAL_TEMPLATE_PATH = f'{INSTRUCTION_DIR}/free_turn_first_turn_prompt.txt'
PROCESS_TEMPLATE_PATH = f'{INSTRUCTION_DIR}/free_turn_following_turn_prompt.txt'
GPT_PROMPT_PATH = f'{INSTRUCTION_DIR}/patient_agent_prompt.txt'

# Default settings
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
VERBOSE = False
MAX_TURNS = 5  # Maximum number of diagnostic turns

# ======================
# Utility Functions
# ======================

def load_instruction(txt_path):
    """Load prompt template from file"""
    try:
        with open(txt_path) as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading instruction from {txt_path}: {e}")
        return None

def parse_answer(answer_text):
    """Extract additional info request and conclusion from model response"""
    print(answer_text)  # Debug output
    pattern = r'### Additional Information Required:\s*(.*?)\s*### Conclusion:\s*(.*)'
    matches = re.search(pattern, answer_text, re.DOTALL)
    if matches:
        additional_info_required = matches.group(1).strip()
        preliminary_conclusion = matches.group(2).strip()
        return preliminary_conclusion, additional_info_required
    else:
        raise ValueError("Could not parse answer format - missing expected sections")

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created output directory: {directory}")

# ======================
# Model API Interfaces
# ======================

def gpt4o_workflow(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query GPT-4o model for additional information retrieval"""
    max_retry = 5
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            client = OpenAI(
                base_url= REFER_API_BASE_URL ,
                api_key= REFER_API_KEY
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            curr_retry += 1
            print(f"Error ({curr_retry}/{max_retry}): {e}")
            time.sleep(5)
    
    return None






def my_model_workflow(messages):

    client = OpenAI(
        base_url=NEW_BASE_URL,
        api_key= NEW_API_KEY
    )

    try:
        response = client.chat.completions.create(
            model=NEW_MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=8192 
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
        print(f"Free-turn API Error: {e}")
        return None, None

# ======================
# Core Multi-Turn Inference Process
# ======================

def process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs):
    """
    Process a single case with multi-turn interaction using the specified model
    
    Parameters:
    -----------
    key : str
        Case identifier
    json_data : dict
        Dictionary containing all case data
    gpt_prompt : str
        Template for GPT-4o prompt
    initial_template : str
        Template for initial query to primary model
    process_template : str
        Template for subsequent queries to primary model
    model_name : str
        Name of the primary model to use
    **kwargs : dict
        Additional model-specific parameters
    """
    # Define output path based on model name
    output_dir = f'../../data/InferenceResults/free_turn_{NEW_MODEL_NAME}'
    output_file = f'{output_dir}/log_{key}.json'
    
    # Skip if already processed
    if os.path.exists(output_file):
        return
    
    model_workflow = my_model_workflow
    
    try:
        # Get case data
        one_instance = json_data[key]
        case_summary = one_instance['generate_case']['case_summary']
        
        if "Ancillary Tests" in case_summary:
            case_summary_paragrapgh = case_summary.strip().split('\n')
            for idx in range(len(case_summary_paragrapgh)):
                if "Ancillary Tests" in case_summary_paragrapgh[idx]:
                    case_summary_without_ancillary_test = "\n".join(case_summary_paragrapgh[:idx])
                    ancillary_test = "\n".join(case_summary_paragrapgh[idx:])
                    break
    
        # Prepare prompts
        gpt_instruction = gpt_prompt.format(
            case=case_summary_without_ancillary_test, 
            ancillary_test_results=ancillary_test
        )
        
        # Initial messages
        primary_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_template.format(case=case_summary_without_ancillary_test)}
        ]
        
        # Log messages with reasoning separately
        messages_log = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_template.format(case=case_summary_without_ancillary_test)}
        ]
        
        # Multi-turn interaction loop
        turn_idx = 1
        while turn_idx <= MAX_TURNS:
            # Get model response for current state
            primary_answer, primary_reasoning = model_workflow(primary_messages)
            
            if VERBOSE:
                print(f"Turn {turn_idx} - Primary model reasoning:\n{primary_reasoning}")
                print(f"Turn {turn_idx} - Primary model answer:\n{primary_answer}")
                
            if not primary_answer:
                print(f"Error: No response from primary model on turn {turn_idx}")
                break
                
            # Clean up response and extract additional information request
            primary_answer = primary_answer.replace('```', '').strip()
            
            # Update message history
            primary_messages.append({"role": "assistant", "content": primary_answer})
            messages_log.append({"role": "assistant", "content": {
                'reasoning': primary_reasoning, 
                'answer': primary_answer
            }})
            
            # Parse response to get conclusion and additional info request
            try:
                preliminary_conclusion, additional_info_required = parse_answer(primary_answer)
            except ValueError as e:
                print(f"Error parsing model response: {e}")
                break
                
            # Check if we're done with information gathering
            if "Not required" in additional_info_required or turn_idx == MAX_TURNS:
                break
                
            # Request additional information from GPT-4o
            gpt_input = f"The junior physician wants the following information:\n{additional_info_required}"
            gpt_response = gpt4o_workflow(gpt_input, gpt_instruction)
            
            if VERBOSE:
                print(f"Turn {turn_idx} - GPT-4o response:\n{gpt_response}")
                
            if not gpt_response:
                print(f"Error: No response from GPT-4o on turn {turn_idx}")
                break
                
            # Format response for next turn
            formatted_response = process_template.format(additional_information=gpt_response)
            
            # Add final turn warning if needed
            if turn_idx == MAX_TURNS - 1:
                formatted_response = "In the next turn, you cannot ask any additional infomation and must make a final diagnoisis.\n" + formatted_response
            
            # Update message history
            primary_messages.append({"role": "user", "content": formatted_response})
            messages_log.append({"role": "user", "content": formatted_response})
            
            # Increment turn counter
            turn_idx += 1
            
        # Prepare output messages
        output_messages = []
        for msg in messages_log:
            output_messages.append({
                'role': msg['role'],
                'content': msg['content'],
            })
        gen_case = one_instance.get('generate_case', {})
        
        ground_truth = gen_case.get('final_diagnosis') or one_instance.get('final_diagnosis', "Unknown")

        level2_data = one_instance.get('level2', {})
        if isinstance(level2_data, dict):
            ancillary_res = level2_data.get('ancillary_test', "")
        else:
            ancillary_res = ""

        # Prepare output data
        log_data = {
            f'{NEW_MODEL_NAME}_messages': output_messages, 
            'ground_truth': ground_truth,
            'ancillary_test_results': ancillary_res,
            'turns': turn_idx,
        }

        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as fp:
            json.dump(log_data, fp, ensure_ascii=False, indent=4)
            
        print(f"Successfully processed {key} with {model_name} in {turn_idx} turns")
        
    except Exception as e:
        print(f"Error processing {key} with {model_name}: {e}")
        error_log = f'level3/{model_name.lower()}.log'
        with open(error_log, 'a') as fp:
            fp.write(f"Error: {e}, {key}\n")

def safe_process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs):
    """Wrapper function with retry logic for robustness"""
    output_dir = f'../../data/InferenceResults/free_turn_{NEW_MODEL_NAME}'
    output_file = f'{output_dir}/log_{key}.json'
    
    # Skip if already processed
    if os.path.exists(output_file):
        return
    
    # Try up to 3 times to process the instance
    for try_idx in range(3):
        try:
            process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs)
            return  # Success, exit the retry loop
        except Exception as e:
            if try_idx == 2:  # If final retry
                print(f"Error processing {key} with {model_name} after 3 attempts: {e}")
                error_log = f'level3/{model_name.lower()}.log'
                with open(error_log, 'a') as fp:
                    fp.write(f"Error: {e}, {key}\n")

# ======================
# Main Inference Functions
# ======================

def run_inference(model_name, max_workers=8, **kwargs):
    """
    Run multi-turn inference for a specific model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    max_workers : int
        Number of parallel workers
    **kwargs : dict
        Additional model-specific parameters
    """
    print(f"Running Level 3 (multi-turn) inference with {model_name}")
    
    # Load templates
    initial_template = load_instruction(INITIAL_TEMPLATE_PATH)
    process_template = load_instruction(PROCESS_TEMPLATE_PATH)
    gpt_prompt = load_instruction(GPT_PROMPT_PATH)
    
    if not all([initial_template, process_template, gpt_prompt]):
        print("Error: Failed to load required templates")
        return
    
    # Load case data
    with open(DATA_PATH, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)

    if not IS_FULL_EVAL:
        keys = list(json_data.keys())[:5]
    else:
        keys = list(json_data.keys())
    print(f"Processing {len(keys)} cases with {model_name}")
    
    # Create output directory
    output_dir = f'../../data/InferenceResults/free_turn_{NEW_MODEL_NAME}'
    ensure_output_dir(output_dir)
    
    # Special handling for Baichuan which doesn't use multiprocessing
    if model_name.lower() == "baichuan":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load model
            model_path = kwargs.get('model_path', "Baichuan-M1-14B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='cuda:0',
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Process cases sequentially
            for key in tqdm.tqdm(keys, desc=f"Processing with {model_name}"):
                safe_process_instance(
                    key, 
                    json_data, 
                    gpt_prompt, 
                    initial_template, 
                    process_template, 
                    model_name, 
                    model=model, 
                    tokenizer=tokenizer
                )
        except Exception as e:
            print(f"Error initializing Baichuan model: {e}")
        return
    
    # Process cases with multiprocessing for other models
    with Pool(processes=max_workers) as pool:
        results = pool.starmap(
            safe_process_instance, 
            [(key, json_data, gpt_prompt, initial_template, process_template, model_name) for key in keys]
        )
        
        # Show progress with tqdm
        list(tqdm.tqdm(results, total=len(keys), desc=f"Processing with {model_name}"))


if __name__ == '__main__':
    run_inference("my_test_model", max_workers=4)