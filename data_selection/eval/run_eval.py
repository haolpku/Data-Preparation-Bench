import argparse
import yaml
import os
import json
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
# from custom_tasks.my_business_task import MyBusinessTask 
from utils.result_converter import convert_lm_eval_results 

# tasks.register_task("my_business_task", MyBusinessTask)

def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="基于lm_eval的定制评估模块")
    parser.add_argument("--cfg", type=str, default="eval/configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径（覆盖配置文件）")
    parser.add_argument("--peft_path", type=str, default=None,
                        help="模型路径（覆盖配置文件）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="模型路径（覆盖配置文件）")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    output_root = args.peft_path
    model_path = args.model_path or cfg["model"]["path"]
    eval_tasks = cfg["eval"]["tasks"]  
    num_fewshot = cfg["eval"]["num_fewshot"]  
    batch_size = cfg["eval"]["batch_size"]
    device = cfg["eval"]["device"]

    model = HFLM(
        pretrained=model_path,
        peft=args.peft_path,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True  # 适配自定义模型
    )

    print(f"===== Task: {eval_tasks} =====")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=eval_tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=None 
    )

    custom_results = convert_lm_eval_results(results)
    custom_save_path = f"{args.output_dir}/eval_results.json"
    with open(custom_save_path, "w", encoding="utf-8") as f:
        json.dump(custom_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
