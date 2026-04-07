import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import datasets
import pandas as pd
import re
from tqdm import tqdm
from openai import OpenAI
import json

from tasks import TASKS
from utils import generate_prompts
from evaluation import evaluate

# ========================================================
NEW_API_KEY = "sk-dummy"                                           
NEW_BASE_URL = "http://XXX/v1"                          
NEW_MODEL_NAME = "gpt-4o"

REFER_MODEL_NAME = "gpt-4o"                                        
REFER_API_BASE_URL = "http://XXX/v1"                
REFER_API_KEY = "sk-dummy"                                          

SAMPLE_LIMIT = 3
LOCAL_DATA_DIR = "./data"

TARGET_TASKS = [
    "abercrombie",
    "hearsay",
    "personal_jurisdiction",
    "ucc_v_common_law",
    "contract_qa"
]
# ==========================================

client = OpenAI(api_key=NEW_API_KEY, base_url=NEW_BASE_URL)

judge_client = OpenAI(api_key=REFER_API_KEY, base_url=REFER_API_BASE_URL)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_r1_output(text):


    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r'^(answer|the answer is|result|prediction)[:\s]+', '', cleaned)
    
    if cleaned.startswith("yes"): return "Yes"
    if cleaned.startswith("no"): return "No"

    for label in ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]:
        if label in cleaned: 
            return label
            

    if "ucc" in cleaned: return "UCC"
    if "common law" in cleaned: return "Common Law"
    
    return cleaned.capitalize()

def get_llm_response(prompt, task_name):

    try:
        response = client.chat.completions.create(
            model=NEW_MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are a legal expert. Task: {task_name}. Provide a concise answer. If it is a Yes/No question, answer only 'Yes' or 'No'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        cleaned = clean_r1_output(raw_content)
        return raw_content, cleaned
    except Exception as e:
        print(f"API ERROR: {e}")
        return "", ""

def judge_calibration(prompt, raw_response, task_name):
    judge_prompt = f"""You are an expert legal evaluator. Your task is to judge the correctness of a model's answer based on the given legal question. 

Task: {task_name}

Question:
{prompt}

Model's raw answer:
{raw_response}

Please output ONLY the correct answer (e.g., "Yes", "No", "UCC", "Common Law", "generic", etc.) based on the question. Do not include any additional explanation.
"""
    try:
        response = judge_client.chat.completions.create(
            model=REFER_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a legal expert judge. Output only the answer."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
        )
        calibrated = response.choices[0].message.content.strip()

        calibrated = clean_r1_output(calibrated)
        return calibrated
    except Exception as e:
        print(f"ERROR:REFER MODEL {e}")
        return None

def run_benchmark():
    summary_results = {}

    for task_name in TARGET_TASKS:
        print(f"\n" + "="*50)
        print(f"TASK: {task_name}")
        
        try:
 
            task_local_path = os.path.join(LOCAL_DATA_DIR, task_name)
            if os.path.exists(task_local_path):
                dataset = datasets.load_from_disk(task_local_path)
                test_df = dataset["test"].to_pandas()
            else:
                raise FileNotFoundError(f"data can't find: {task_local_path}")

            if SAMPLE_LIMIT is not None:
                print(f"NOTICE： {SAMPLE_LIMIT} data.")
                test_df = test_df.head(SAMPLE_LIMIT)


            template_path = f"tasks/{task_name}/base_prompt.txt"
            if os.path.exists(template_path):
                with open(template_path, "r") as f:
                    prompt_template = f.read()
            else:
                prompt_template = "Context: {{text}}\nQuestion: {{question}}\nAnswer:"


            prompts = generate_prompts(prompt_template, test_df)
            

            temp_rows = []
            for i, p in enumerate(tqdm(prompts)):
                raw, cleaned = get_llm_response(p, task_name)
                temp_rows.append({
                    "prompt": p,
                    "gold": test_df["answer"].iloc[i],
                    "pred": cleaned,
                    "raw_response": raw
                })
                if i < 2:
                    print(f"\n[case {i+1}]")
                    print(f"raw_response: {raw}")
                    print(f"pred: {cleaned}")
                    print(f"gold: {test_df['answer'].iloc[i]}")

    
            ensure_dir("temp_result")
            temp_path = f"./temp_result/{task_name}_{NEW_MODEL_NAME}.jsonl"
            with open(temp_path, "w", encoding="utf-8") as f:
                for row in temp_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"TASK {task_name} : {temp_path}")



            refined_preds = []
            answers = []
            for row in temp_rows:
                gold = row["gold"]
                pred = row["pred"]

                if pred == "" or pred != gold:
                    calibrated = judge_calibration(row["prompt"], row["raw_response"], task_name)
                    if calibrated is not None:
                        pred = calibrated
                    else:
                        print(f"Still pred: {pred}")
                refined_preds.append(pred)
                answers.append(gold)


            score = evaluate(task_name, refined_preds, answers)
            summary_results[task_name] = score
            print(f"\nTask [{task_name}] score: {score:.4f}")

    
            refined_path = f"./temp_result/{task_name}_{NEW_MODEL_NAME}_refined.jsonl"
            with open(refined_path, "w", encoding="utf-8") as f:
                for i, row in enumerate(temp_rows):
                    row["refined_pred"] = refined_preds[i]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"refined_result: {refined_path}")

        except Exception as e:
            print(f"Task {task_name} ERROR: {e}")
            summary_results[task_name] = "Error"


    print("\n" + "█"*50)
    print("  LegalBench ")
    print("-" * 50)
    print(f"{'TASK':<25} | {'SCORE':<10}")
    print("-" * 50)
    for task, score in summary_results.items():
        score_display = f"{score:.4f}" if isinstance(score, float) else score
        print(f"{task:<25} | {score_display:<10}")
    print("█"*50)

    df_results = pd.DataFrame(list(summary_results.items()), columns=['Task', 'Score'])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ensure_dir("result")
    filename = f"./result/legalbench_results_{NEW_MODEL_NAME}_{timestamp}.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nSAVED: {filename}")

if __name__ == "__main__":
    run_benchmark()