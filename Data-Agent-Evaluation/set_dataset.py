#!/usr/bin/env python3


import shutil
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SRC_BASE = BASE_DIR / "Data-Agent-Evaluation-Dataset"


COPY_MAP = [
    # 1. FinCDM -> Business_eval/FinCDM/data_en
    ("FinCDM", "Business_eval/FinCDM/data_en"),
    # 2. XFINBENCH/dataset -> Business_eval/XFinBench/dataset
    ("XFINBENCH/dataset", "Business_eval/XFinBench/dataset"),
    # 3. legalbench/data -> Law_eval/legalbench/data
    ("legalbench/data", "Law_eval/legalbench/data"),
    # 4. Lex-glue -> Law_eval/lex-glue/data_filtered_48k
    ("Lex-glue", "Law_eval/lex-glue/data_filtered_48k"),
    # 5. Medcaseresoning/data -> Medicine_eval/MedCaseReasoning/data
    ("Medcaseresoning/data", "Medicine_eval/MedCaseReasoning/data"),
    # 6. MedmcQA/medmcqa -> Medicine_eval/MedmcQA/eval/data/medmcqa
    ("MedmcQA/medmcqa", "Medicine_eval/MedmcQA/eval/data/medmcqa"),
    # 7. MedRBench -> Medicine_eval/MedRBench/data/MedRBench
    ("MedRBench", "Medicine_eval/MedRBench/data/MedRBench"),
]

def copy_dataset(src_rel: str, dst_rel: str) -> None:
    src = SRC_BASE / src_rel
    dst = BASE_DIR / dst_rel

    if not src.exists():
        print(f"Warning: source path does not exist, skipping: {src}")
        return


    dst.parent.mkdir(parents=True, exist_ok=True)

    try:

        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Success: {src_rel} -> {dst_rel}")
    except Exception as e:
        print(f"Error: failed to copy {src_rel}: {e}")

def main():
    print("Starting to copy datasets...")
    for src_rel, dst_rel in COPY_MAP:
        copy_dataset(src_rel, dst_rel)
    print("All copy tasks completed!")

if __name__ == "__main__":
    main()