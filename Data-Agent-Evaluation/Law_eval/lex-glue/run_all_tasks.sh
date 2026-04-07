#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="$(python - <<'PY'
import ast
from pathlib import Path

path = Path("experiments/eurlex.py")
model_name = "model"
for line in path.read_text(encoding="utf-8").splitlines():
    if line.startswith("NEW_MODEL_NAME"):
        _, value = line.split("=", 1)
        model_name = ast.literal_eval(value.strip())
        break
print(model_name)
PY
)"

TASKS=(
  "eurlex"
  "ledgar"
  "unfair_tos"
)

run_task() {
  local task="$1"
  echo "=== Running ${task} ==="
  python "experiments/${task}.py"
  local result_file="result/${task}_${MODEL_NAME}.json"
  if [[ -f "$result_file" ]]; then
    echo "--- Score (${task}) ---"
    python - <<PY
import json
p = "${result_file}"
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
  else
    echo "Result file not found: ${result_file}"
  fi
  echo
}

for t in "${TASKS[@]}"; do
  run_task "$t"
done

echo "=== Computing accuracy ==="
python scripts/compute_accuracy.py \
  --input_dir temp_result \
  --output_dir temp_result_min \
  --output score_result
