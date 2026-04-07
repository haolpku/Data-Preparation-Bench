import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def parse_args():
    parser = argparse.ArgumentParser(description="MedCaseReasoning LLM ")
    parser.add_argument("--input_path", type=str, default="./data/test.jsonl")
    parser.add_argument("--output_path", type=str, default=f"./results/{NEW_MODEL_NAME}_medcase_results.jsonl")
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()

PROMPT_TEMPLATE = """Read the following case presentation and give the most likely diagnosis.
First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.
Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.

----------------------------------------
CASE PRESENTATION
----------------------------------------
%s
----------------------------------------
OUTPUT TEMPLATE
----------------------------------------
<think>
...your internal reasoning for the diagnosis...
</think>
<answer>
...the name of the disease/entity...
</answer>"""

def process_item(item, client, args):
    case_input = item.get("case_prompt", "")
    full_prompt = PROMPT_TEMPLATE % case_input

    try:
        response = client.chat.completions.create(
            model=NEW_MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
        )
        
        
        message = response.choices[0].message
        content = message.content or ""
        
        
        reasoning = getattr(message, 'reasoning_content', None)
        
        
        if reasoning:
            full_response = f"<think>\n{reasoning}\n</think>\n{content}"
        else:
            
            if "<think>" not in content and content.strip():
                full_response = f"<think>\n{content}\n</think>\n<answer>\n{content}\n</answer>"
            else:
                full_response = content

        result = {
            "pmcid": item.get("pmcid", "N/A"),
            "final_diagnosis": item.get("final_diagnosis", ""),
            "diagnostic_reasoning": item.get("diagnostic_reasoning", ""),
            "llm_response": full_response
        }
        
        with open(args.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"\nERROR [PMCID: {item.get('pmcid')}]: {e}")

def main():
    args = parse_args()
    client = OpenAI(api_key=NEW_API_KEY , base_url=NEW_BASE_URL)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    if not os.path.exists(args.input_path):
        print(f"ERROR:can't find file: {args.input_path}")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    if not IS_FULL_EVAL:
        all_data = all_data[:5]
    
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            done_pmcids = {json.loads(line).get("pmcid") for line in f if line.strip()}
        all_data = [d for d in all_data if d.get("pmcid") not in done_pmcids]
        if done_pmcids:
            print(f"Skip {len(done_pmcids)} completed data")

    if not all_data:
        print("ALL Finished")
        return

    print(f"Begin to test, the num of threads: {args.threads}")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        list(tqdm(executor.map(lambda x: process_item(x, client, args), all_data), total=len(all_data)))

if __name__ == "__main__":
    main()