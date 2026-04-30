#!/bin/bash

set -euo pipefail

# ============================================================================
# Configurable parameters (mirroring reference.sh)
# ============================================================================

# vLLM settings
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DTYPE="${DTYPE:-bfloat16}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
BASE_PORT="${BASE_PORT:-8002}"

# Evaluation settings
MAX_CONCURRENT="${MAX_CONCURRENT:-256}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"

# Judge API settings (fill in if LLM-as-judge is needed)
JUDGE_URL="${JUDGE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"

# GPU selection (default: all visible GPUs)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# ============================================================================
# Argument parsing
# ============================================================================

usage() {
    echo "Usage: $0 <model_path> <domain> <output_dir>"
    echo "  model_path: path to the model directory"
    echo "  domain:     business | medicine | law"
    echo "  output_dir: directory to save all results"
    echo ""
    echo "Environment variables (optional):"
    echo "  TP_SIZE, GPU_MEMORY_UTIL, MAX_MODEL_LEN, DTYPE, BASE_PORT"
    echo "  HEALTH_TIMEOUT, MAX_CONCURRENT, JUDGE_CONCURRENCY"
    echo "  CUDA_VISIBLE_DEVICES, JUDGE_URL, JUDGE_API_KEY, JUDGE_MODEL"
    exit 1
}

if [ $# -ne 3 ]; then
    usage
fi

MODEL_PATH="$1"
DOMAIN="$2"
OUTPUT_DIR="$3"

# Validate domain
if [[ "$DOMAIN" != "business" && "$DOMAIN" != "medicine" && "$DOMAIN" != "law" ]]; then
    echo "Error: domain must be one of business, medicine, law"
    exit 1
fi

INPUT_JSONL="./data/${DOMAIN}.jsonl"
if [ ! -f "$INPUT_JSONL" ]; then
    echo "Error: input file not found: $INPUT_JSONL"
    exit 1
fi

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check for safetensors (from reference.sh)
if ! ls "${MODEL_PATH}"/model*.safetensors >/dev/null 2>&1; then
    echo "Error: no .safetensors found in ${MODEL_PATH}"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
mkdir -p "$OUTPUT_DIR"

VLLM_PORT=$BASE_PORT
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/health"

LOG_DIR="${OUTPUT_DIR}/vllm_logs"
mkdir -p "$LOG_DIR"
VLLM_LOG_FILE="${LOG_DIR}/vllm_${MODEL_NAME}.log"

# ============================================================================
# Cleanup helper (from reference.sh)
# ============================================================================

cleanup_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup_vllm EXIT

# ============================================================================
# Wait for health (from reference.sh)
# ============================================================================

wait_for_health() {
    local health_url=$1
    local timeout_secs=$2
    local start_ts
    start_ts=$(date +%s)

    # Give vLLM some time to start before polling
    sleep 10

    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "$health_url" | grep -q "^200$"; then
            return 0
        fi

        local now_ts
        now_ts=$(date +%s)
        if [ $((now_ts - start_ts)) -ge "$timeout_secs" ]; then
            return 1
        fi

        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Error: vLLM process exited unexpectedly"
            return 1
        fi

        sleep 10
    done
}

# ============================================================================
# Start vLLM server (from reference.sh)
# ============================================================================

echo "========================================"
echo "Starting vLLM server"
echo "  Model:   $MODEL_PATH"
echo "  Port:    $VLLM_PORT"
echo "  Log:     $VLLM_LOG_FILE"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
fi
echo "========================================"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES
fi

vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    > "$VLLM_LOG_FILE" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM health check (timeout: ${HEALTH_TIMEOUT}s)..."
if ! wait_for_health "$VLLM_HEALTH_URL" "$HEALTH_TIMEOUT"; then
    echo "Error: vLLM health check failed. See log: $VLLM_LOG_FILE"
    cleanup_vllm
    exit 1
fi
echo "vLLM is ready."

# ============================================================================
# Step 1: Inference
# ============================================================================

echo "========================================"
echo "Step 1: Running inference"
echo "  Input:  $INPUT_JSONL"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

EVAL_LOG="${OUTPUT_DIR}/eval.log"

uv run inference.py \
    --input-jsonl "$INPUT_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --url "$VLLM_BASE_URL" \
    --api-key "key-123" \
    --concurrency "$MAX_CONCURRENT" 2>&1 | tee "$EVAL_LOG"

# ============================================================================
# Step 2: Judging
# ============================================================================

echo "========================================"
echo "Step 2: Running judge"
echo "========================================"

JUDGE_OUTPUT_DIR="${OUTPUT_DIR}/judge_out"
mkdir -p "$JUDGE_OUTPUT_DIR"

JUDGE_LOG="${JUDGE_OUTPUT_DIR}/judge.log"

uv run judge.py \
    --input-jsonl "${OUTPUT_DIR}/inference_results.jsonl" \
    --output-dir "$JUDGE_OUTPUT_DIR" \
    --judge-url "$JUDGE_URL" \
    --judge-api-key "$JUDGE_API_KEY" \
    --judge-model "$JUDGE_MODEL" \
    --concurrency "$JUDGE_CONCURRENCY" 2>&1 | tee "$JUDGE_LOG"

# ============================================================================
# Step 3: Extract scores
# ============================================================================

echo "========================================"
echo "Step 3: Extracting scores"
echo "========================================"

uv run extract_score.py \
    --input "${JUDGE_OUTPUT_DIR}/judge_results.jsonl" \
    --output "${OUTPUT_DIR}/calculated_scores.json"

# ============================================================================
# Done
# ============================================================================

echo "========================================"
echo "All steps completed successfully."
echo "  Inference results: ${OUTPUT_DIR}/inference_results.jsonl"
echo "  Judge results:     ${JUDGE_OUTPUT_DIR}/judge_results.jsonl"
echo "  Scores:            ${OUTPUT_DIR}/calculated_scores.json"
echo "========================================"
