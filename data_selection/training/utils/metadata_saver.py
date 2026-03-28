# training/utils/metadata_saver.py
import json
import os
import sys
import torch
import platform
from pathlib import Path
from datetime import datetime
from .logger import init_train_logger

logger = init_train_logger("metadata_saver")

def save_training_metadata(save_dir, base_model, train_data, config_path, run_id, timestamp):
    save_dir = Path(save_dir)
    metadata_path = save_dir / "training_metadata.json"

    basic_info = {
        "run_id": run_id,
        "timestamp": timestamp,
        "train_start_time": datetime.now().isoformat(),
        "base_model": base_model,
        "train_data_path": train_data,
        "train_config_path": config_path,
        "output_dir": str(save_dir.absolute()),
        "working_dir": os.getcwd()
    }

    config_snapshot = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config_snapshot = yaml.safe_load(f)

    metadata = {
        "basic_info": basic_info,
        "config_snapshot": config_snapshot,
        "metadata_version": "1.0"
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"✅ 训练元数据已保存至：{metadata_path}")
    return metadata_path
