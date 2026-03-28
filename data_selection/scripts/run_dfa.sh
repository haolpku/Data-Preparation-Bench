#!/bin/bash

CONFIG_PATH="./configs/dfa.yaml"
FILTER_ENV="dfa"
BENCH_ENV="bench"
BENCH_YAML="./env_list.bench_env.yaml"
ID_FILE="./.latest_run"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

set -e # Exit on error

echo "=== Stage 1: Data Filtering ==="

if ! conda info --envs | grep -q "$FILTER_ENV"; then
    echo "📦 Creating environment $FILTER_ENV (Python 3.11)..."
    conda create -n "$FILTER_ENV" python=3.11 -y
fi
conda activate "$FILTER_ENV"

python ./pipeline.py --config "$CONFIG_PATH" --stage "filter"

if [ -f "$ID_FILE" ]; then
    RESUME_ID=$(cat "$ID_FILE")
    mv "$ID_FILE" "${ID_FILE}_${RESUME_ID}"
    echo "✅ Successfully captured Experiment ID: $RESUME_ID"
else
    echo "❌ Error: $ID_FILE not found. Filtering stage might have failed."
    exit 1
fi

echo "=== Stage 2: Training and Evaluation ==="

if ! conda info --envs | grep -q "$BENCH_ENV"; then
    echo "📦 Creating environment from $BENCH_YAML..."
    conda env create -f "$BENCH_YAML"
fi
conda activate "$BENCH_ENV"

python ./pipeline.py \
    --config "$CONFIG_PATH" \
    --stage "train,eval" \
    --resume_id "$RESUME_ID"

echo "🎉 Full DFA pipeline completed successfully."