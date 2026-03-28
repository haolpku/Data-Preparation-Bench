import json
from datetime import datetime

from pathlib import Path

def register_dataset2(ds_id, file_path, column_mapping):
    dataset_dir = DATASET_INFO_PATH.parent  
    target_file_path = dataset_dir / file_path.name  # 目标路径（和原文件同名）
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
    if not DATASET_INFO_PATH.exists():
        with open(DATASET_INFO_PATH, "w") as f:
            json.dump({}, f, indent=2)
    
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Create dataset directory: {dataset_dir}")
    
    if not target_file_path.exists():
        try:
            shutil.copy2(file_path, target_file_path)
            logger.info(f"The dataset file does not exist. It has been automatically copied: \n  Source path: {file_path}\n  Target path: {target_file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to copy the dataset file: {str(e)}")
    else:
        logger.info(f"The dataset file already exists: {target_file_path}. No need to copy.")

    with open(DATASET_INFO_PATH, "r") as f:
        ds_registry = json.load(f)
        
    ds_registry[ds_id] = {
        "file_name": str(file_path),
        "columns": column_mapping,
        "register_time": datetime.now().isoformat()
    }
    with open(DATASET_INFO_PATH, "w") as f:
        json.dump(ds_registry, f, indent=2, ensure_ascii=False)

    return ds_id

def register_dataset(ds_name, file_path, column_mapping):
    dataset_dir = file_path.parent  
    DATASET_INFO_PATH = Path(f"{dataset_dir}/dataset_info.json")
    if not DATASET_INFO_PATH.exists():
        with open(DATASET_INFO_PATH, "w") as f:
            json.dump({}, f, indent=2)
    
    with open(DATASET_INFO_PATH, "r") as f:
        ds_registry = json.load(f)
    
    ds_registry[ds_name] = {
        "file_name": str(file_path.name),
        "columns": column_mapping,
        "register_time": datetime.now().isoformat()
    }
    with open(DATASET_INFO_PATH, "w") as f:
        json.dump(ds_registry, f, indent=2, ensure_ascii=False)

    return DATASET_INFO_PATH.parent