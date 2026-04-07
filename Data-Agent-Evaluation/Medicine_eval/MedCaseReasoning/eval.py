import json
import os
import re
import argparse
import logging
import threading
import traceback
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
    parser = argparse.ArgumentParser(description="MedCaseReasoning")
    parser.add_argument("--input_path", type=str, default=f"./results/{NEW_MODEL_NAME}_medcase_results.jsonl")
    parser.add_argument("--output_path", type=str, default=f"./results/{NEW_MODEL_NAME}_medcase_evaluated.jsonl")
    parser.add_argument("--trace_path", type=str, default=f"./results/{NEW_MODEL_NAME}_medcase_traces.json")
    parser.add_argument("--score_path", type=str, default=f"./results/{NEW_MODEL_NAME}_medcase_score.json")
    parser.add_argument("--log_path", type=str, default=f"./results/{NEW_MODEL_NAME}_eval_threads.log")
    parser.add_argument("--model_name", type=str, default=REFER_MODEL_NAME)
    parser.add_argument("--api_key", type=str, default=REFER_API_KEY)
    parser.add_argument("--base_url", type=str, default=REFER_API_BASE_URL)
    parser.add_argument("--threads", type=int, default=5)
    return parser.parse_args()




ACCURACY_PROMPT = "Is our predicted diagnosis correct (y/n)?\nPredicted diagnosis: {pred}, True diagnosis: {true}\nAnswer [y/n]."


REASONING_MATCH_PROMPT = """Analyze if the model's reasoning covers the following clinician's point.
Clinician's point: {gold_point}
Model's reasoning: {model_reasoning}
Does the model mention or imply this specific point? Answer [y/n]."""

args = parse_args()
client = OpenAI(api_key=args.api_key, base_url=args.base_url)

os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.FileHandler(args.log_path, encoding="utf-8")]
)
logger = logging.getLogger("eval")

os.makedirs(os.path.dirname(args.trace_path) or ".", exist_ok=True)

def extract_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def get_llm_decision(prompt):
    request_payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5
    }
    try:
        response = client.chat.completions.create(**request_payload)
        ans = response.choices[0].message.content.lower()
        trace = {
            "prompt": prompt,
            "request": request_payload,
            "response_text": response.choices[0].message.content,
            "response_raw": response.model_dump() if hasattr(response, "model_dump") else str(response)
        }
        return 'y' in ans, trace
    except Exception as e:
        logger.error("ERROR：LLM REQUEST: %s\n%s", e, traceback.format_exc())
        trace = {
            "prompt": prompt,
            "request": request_payload,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return False, trace

def evaluate_item(args_tuple):
    idx, item = args_tuple
    llm_output = item.get("llm_response", "")
    pred_diag = extract_tag(llm_output, "answer")
    model_think = extract_tag(llm_output, "think")
    true_diag = item.get("final_diagnosis", "")
    gold_reasoning = item.get("diagnostic_reasoning", "")

 
    acc_prompt = ACCURACY_PROMPT.format(pred=pred_diag, true=true_diag)
    is_correct, acc_trace = get_llm_decision(acc_prompt)

  
    gold_points = re.split(r'\d+\.', gold_reasoning)
    gold_points = [p.strip() for p in gold_points if p.strip()]
    
    hits = 0
    recall_traces = []
    if gold_points:
        for point in gold_points:
            match_prompt = REASONING_MATCH_PROMPT.format(gold_point=point, model_reasoning=model_think)
            is_match, match_trace = get_llm_decision(match_prompt)
            recall_traces.append({
                "gold_point": point,
                "decision": is_match,
                "trace": match_trace
            })
            if is_match:
                hits += 1
        recall = hits / len(gold_points)
    else:
        recall = 0.0

    trace_payload = {
        "pmcid": item.get("pmcid"),
        "idx": idx,
        "accuracy": {
            "decision": is_correct,
            "trace": acc_trace
        },
        "reasoning_recall": {
            "num_points": len(gold_points),
            "hits": hits,
            "traces": recall_traces
        }
    }

    return {
        "pmcid": item.get("pmcid"),
        "is_correct": is_correct,
        "recall": recall,
        "num_points": len(gold_points),
        "trace": trace_payload
    }

def main():
    if not os.path.exists(args.input_path):
        print(f"ERROR:can't find file: {args.input_path}")
        return

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"Using {args.model_name} eval {len(data)} data...")
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(tqdm(executor.map(evaluate_item, enumerate(data)), total=len(data)))

    
    total = len(results)
    avg_accuracy = sum(1 for r in results if r['is_correct']) / total
    avg_recall = sum(r['recall'] for r in results) / total

    print("\n" + "="*30)
    print(f"REPORT ({args.model_name})")
    print("-" * 30)
    print(f"SAMPLE NUM: {total}")
    print(f"Accuracy: {avg_accuracy:.2%}")
    print(f"Reasoning Recall: {avg_recall:.2%}")
    print("="*30)

    
    try:
        os.makedirs(os.path.dirname(args.score_path) or ".", exist_ok=True)
        score_payload = {
            "model_name": args.model_name,
            "input_path": args.input_path,
            "output_path": args.output_path,
            "total": total,
            "accuracy": avg_accuracy,
            "recall": avg_recall
        }
        with open(args.score_path, "w", encoding="utf-8") as sf:
            json.dump(score_payload, sf, ensure_ascii=False)
    except Exception as e:
        logger.error("ERROR：write to score file (%s): %s\n%s", args.score_path, e, traceback.format_exc())

   
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    
    try:
        traces = [r.get("trace") for r in results]
        with open(args.trace_path, "w", encoding="utf-8") as tf:
            json.dump(traces, tf, ensure_ascii=False)
    except Exception as e:
        logger.error("ERROR：write to trace file(%s): %s\n%s", args.trace_path, e, traceback.format_exc())

if __name__ == "__main__":
    main()
