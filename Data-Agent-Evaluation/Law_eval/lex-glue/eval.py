import os
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

API_KEY = "sk-dummy"
BASE_URL = "http://localhost:8000/v1" 
MODEL_NAME = "qwen2.5-7b-law-full-claude-simply-dolly"
SAMPLE_LIMIT = None
LOCAL_DATA_DIR = "./data"  


TARGET_TASKS = [
    "abercrombie",            
    "hearsay",                
    "personal_jurisdiction",  
    "ucc_v_common_law",       
    "contract_qa"             
]


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_r1_output(text):

    import re

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
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"You are a legal expert. Task: {task_name}. Provide a concise answer. If it is a Yes/No question, answer only 'Yes' or 'No'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        raw_content = response.choices[0].message.content
        return clean_r1_output(raw_content)
    except Exception as e:
        print(f"API Failed: {e}")
        return ""

def run_benchmark():
    summary_results = {}
    import re 

    for task_name in TARGET_TASKS:
        print(f"\n" + "="*50)
        print(f"Task: {task_name}")
        
        try:
            
            task_local_path = os.path.join(LOCAL_DATA_DIR, task_name)
            if os.path.exists(task_local_path):
                print(f"Loading data : {task_local_path}")
                dataset = datasets.load_from_disk(task_local_path)
                test_df = dataset["test"].to_pandas()

            # dataset = datasets.load_dataset("nguha/legalbench", task_name, trust_remote_code=True)
            # test_df = dataset["test"].to_pandas()


            if SAMPLE_LIMIT is not None:
                print(f"SAMPLE: {SAMPLE_LIMIT} 。")
                test_df = test_df.head(SAMPLE_LIMIT)


 
            template_path = f"tasks/{task_name}/base_prompt.txt"
            if os.path.exists(template_path):
                with open(template_path, "r") as f:
                    prompt_template = f.read()
            else:
                prompt_template = "Context: {{text}}\nQuestion: {{question}}\nAnswer:"

            prompts = generate_prompts(prompt_template, test_df)

            generations = []
            temp_rows = []
            print(f" Using {MODEL_NAME} ...")
            for i, p in enumerate(tqdm(prompts)):
                res = get_llm_response(p, task_name)
                generations.append(res)
                temp_rows.append(
                    {
                        "prompt": p,
                        "gold": test_df["answer"].iloc[i],
                        "pred": res,
                    }
                )

                if i < 2:
                    print(f"\n[id: {i+1}]")
                    print(f"Predicted answer: {res}")
                    print(f"Gold answer: {test_df['answer'].iloc[i]}")

            answers = test_df["answer"].tolist()
            score = evaluate(task_name, generations, answers)
            summary_results[task_name] = score
            print(f"\nTask [{task_name}] score: {score:.4f}")
            ensure_dir("temp_result")
            temp_path = f"./temp_result/{task_name}_{MODEL_NAME}.jsonl"
            with open(temp_path, "w", encoding="utf-8") as f:
                for row in temp_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"The temporary results for {task_name}: {temp_path}")
            
        except Exception as e:
            print(f" ERROR {task_name} : {e}")
            summary_results[task_name] = "Error"


    print("\n" + "█"*50)
    print("      LegalBench ")
    print("-" * 50)
    print(f"{'TASK':<25} | {'SCORE':<10}")
    print("-" * 50)
    for task, score in summary_results.items():
        score_display = f"{score:.4f}" if isinstance(score, float) else score
        print(f"{task:<25} | {score_display:<10}")
    print("█"*50)

    df_results = pd.DataFrame(list(summary_results.items()), columns=['Task', 'Score'])
    

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"./result/legalbench_results_{MODEL_NAME}_{timestamp}.csv"
    
    df_results.to_csv(filename, index=False)
    print(f"\nRESULT SAVED: {filename}")

if __name__ == "__main__":
    run_benchmark()
