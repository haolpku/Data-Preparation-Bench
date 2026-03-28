from datetime import datetime
def convert_lm_eval_results(lm_eval_results):
    eval_tasks = list(lm_eval_results.get("results", {}).keys())
    
    config = lm_eval_results.get("config", {})
    model_name = config.get("model") or config.get("model_args") or "unknown_model"

    custom_results = {
        "model_name": model_name,
        "eval_tasks": eval_tasks,
        "core_metrics": {},
        "eval_time": lm_eval_results.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 
    }

    for task, metrics in lm_eval_results.get("results", {}).items():
        core_metric = (
            metrics.get("acc_norm,none") or 
            metrics.get("acc,none") or 
            metrics.get("exact_match,none") or
            metrics.get("acc_norm") or 
            metrics.get("acc")
        )
        
        if core_metric is None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    core_metric = v
                    break
        
        custom_results["core_metrics"][task] = round(core_metric, 4) if core_metric else 0

    return custom_results