import os

import yaml
from pathlib import Path

def build_train_config(dataset_dir, base_config_profile, gpu_config, model_path, dataset_name, output_dir, template):
    base_config_path = base_config_profile
    if not os.path.exists(base_config_path):
        raise ValueError(f"基础配置不存在：{base_config_path}")
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # runtime_config_path = Path("training/configs/runtime") / f"{gpu_config}.yaml"
    # with open(runtime_config_path, "r") as f:
    #     runtime_config = yaml.safe_load(f)
    # config.update(runtime_config)

    config.update({
        "model_name_or_path": str(model_path),
        "dataset": dataset_name,
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "template": template,
    })

    return yaml.safe_load(yaml.dump(config))  # 标准化配置格式
