import argparse
import os
import yaml

from pathlib import Path
from datetime import datetime
from core.config_builder import build_train_config
from core.dataset_manager import register_dataset
from core.train_executor import run_llamafactory_train
from utils.metadata_saver import save_training_metadata
from utils.logger import init_train_logger

logger = init_train_logger("training_module")

def run_train(train_file, train_dataset_name, model_path, output_root, template="qwen3", config_profile="llama3_lora", gpu_config="single_gpu"):
    train_file_abs = Path(train_file).absolute()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{train_file_abs.stem}_{model_path.split('/')[-1]}_{timestamp}"
    save_dir = Path(output_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # dataset register
        dataset_dir = register_dataset(
            ds_name=train_dataset_name,
            file_path=train_file_abs,
            column_mapping={"prompt": "instruction", "query": "input", "response": "output"}
        )

        run_config = build_train_config(
            dataset_dir=dataset_dir,
            base_config_profile=config_profile,  # 如llama3_lora
            gpu_config=gpu_config,
            model_path=model_path,
            dataset_name=train_dataset_name,
            output_dir=save_dir,
            template=template,
        )
        with open(config_profile, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, indent=2, sort_keys=False, allow_unicode=True)

        run_llamafactory_train(config_profile, log_file=Path("logs") / f"{run_id}.log")
        logger.info(f"Training completed! Model saved to: {save_dir}")

        save_training_metadata(
            save_dir=save_dir,
            base_model=str(model_path),
            train_data=str(train_file_abs),
            config_path=str(config_profile),
            run_id=run_id,
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Training failed! Error: {str(e)}", exc_info=True)
        raise  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Training data file path")
    parser.add_argument("--train_dataset_name", required=True, help="Training dataset name")
    parser.add_argument("--model_path", required=True, help="Base model path")
    parser.add_argument("--output_root", default="../training/outputs", help="Training output root directory")
    parser.add_argument("--template", default="qwen3", help="Prompt Template Name")
    parser.add_argument("--config_profile", default="llama3_lora", help="Basic configuration template (located in configs/base/)")
    parser.add_argument("--gpu_config", default="single_gpu")
    args = parser.parse_args()
    
    run_train(
        train_file=args.train_file,
        train_dataset_name=args.train_dataset_name,
        model_path=args.model_path,
        output_root=args.output_root,
        template=args.template,
        config_profile=args.config_profile,
        gpu_config=args.gpu_config
    )
