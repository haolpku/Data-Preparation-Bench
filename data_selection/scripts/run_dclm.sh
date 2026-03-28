#!/bin/bash

# Configuration
CONFIG_PATH="./configs/dclm.yaml"
FILTER_ENV="dclm"
BENCH_ENV="bench"
FILTER_YAML="env_config/dclm.yaml"
BENCH_YAML="env_config/bench.yaml"
ID_FILE="./.latest_run"

# Source Conda functions for script compatibility
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

set -e # Exit on error

echo "=== Stage 1: Data Filtering ==="

# Check/Create Filter Env
if ! conda env list | grep -q "^$FILTER_ENV\s"; then
    echo "Creating environment: $FILTER_ENV..."
    conda env create -f "$FILTER_YAML"
fi

conda activate "$FILTER_ENV"
python ./pipeline.py --config "$CONFIG_PATH" --stage "filter"

# Retrieve Resume ID
if [ -f "$ID_FILE" ]; then
    RESUME_ID=$(cat "$ID_FILE")
    mv "$ID_FILE" "${ID_FILE}_${RESUME_ID}"
    echo "✅ Success. Experiment ID: $RESUME_ID"
else
    echo "❌ Error: .latest_run file not found. Filtering may have failed."
    exit 1
fi

echo "=== Stage 2: Training and Evaluation ==="

# Check/Create Bench Env
if ! conda env list | grep -q "^$BENCH_ENV\s"; then
    echo "Creating environment: $BENCH_ENV..."
    conda env create -f "$BENCH_YAML"
fi

conda activate "$BENCH_ENV"
python ./pipeline.py \
    --config "$CONFIG_PATH" \
    --stage train,eval \
    --resume_id "$RESUME_ID"

echo "🎉 Full DCLM pipeline completed successfully."