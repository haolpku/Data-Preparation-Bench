# training/utils/logger.py
import logging
import os
from pathlib import Path
from datetime import datetime

def init_train_logger(name):
    log_root = Path("training/logs")
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"train_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
