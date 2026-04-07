import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd


def resolve_path(project_root: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(project_root, path_value)


def main() -> None:
    default_dataset = "dataset/validation_set.csv"
    default_output = "dataset/validation_bool_questions.json"

    parser = argparse.ArgumentParser(description="Export all bool-task questions to JSON.")
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Path to the CSV dataset (default: dataset/validation_set.csv)",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path for the JSON output file (default: dataset/validation_bool_questions.json)",
    )
    args = parser.parse_args()

    project_root = os.environ.get(
        "PROJECT_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dataset_path = resolve_path(project_root, args.dataset)
    output_path = resolve_path(project_root, args.output)

    df = pd.read_csv(dataset_path)
    bool_df = df[df["task"] == "bool"].copy()

    records: List[Dict[str, Any]] = bool_df.to_dict(orient="records")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)

    print(
        f"Exported {len(records)} bool questions to {output_path}. "
        f"Source: {dataset_path}"
    )


if __name__ == "__main__":
    main()
