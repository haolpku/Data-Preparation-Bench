#!/bin/bash
# ==============================================================================
# run_all_bench.sh
# 统一评测脚本：支持 math / general / science / business / law / medicine 六个 domain
# 用法: 在 Data-Preparation-Bench-2 根目录执行
#   bash Data-Agent-Evaluation/scripts/run_all_bench.sh
# ==============================================================================

set -euo pipefail

# ========================= 用户配置区 =========================
# 格式: "domain|model_path"
# domain 可选: math, general, science, business, law, medicine
MODELS=(
    "math|/path/to/your/math-model"
    "general|/path/to/your/general-model"
    "science|/path/to/your/science-model"
    "business|/path/to/your/business-model"
    "law|/path/to/your/law-model"
    "medicine|/path/to/your/medicine-model"
    # 同一个模型可以跑多个 domain，例如:
    # "math|/path/to/model-A"
    # "general|/path/to/model-A"
    # "science|/path/to/model-A"
    # "business|/path/to/model-A"
    # "law|/path/to/model-A"
    # "medicine|/path/to/model-A"
)

# 统一输出根目录 (相对于 Data-Preparation-Bench-2 根目录)
OUTPUT_ROOT="Data-Agent-Evaluation/bench_outputs"
# ==============================================================

# 各 bench 仓库相对路径 (相对于 Data-Preparation-Bench-2 根目录)
MATH_DIR="Data-Agent-Evaluation/Qwen2.5-Math/evaluation"
GENERAL_DIR="Data-Agent-Evaluation/lm-evaluation-harness"
SCIENCE_DIR="Data-Agent-Evaluation/lm-open-science-evaluation"
SIMPLE_EVAL_DIR="Data-Agent-Evaluation/simple-evaluation"

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ---------- helper: 从 model_path 提取模型短名 ----------
get_model_name() {
    basename "$1"
}

# ---------- math 评测 ----------
run_math() {
    local model_path="$1"
    local model_name
    model_name="$(get_model_name "$model_path")"
    local output_dir="${OUTPUT_ROOT}/math/${model_name}"

    echo "======================================================"
    echo "[math] model: ${model_path}"
    echo "[math] output: ${output_dir}"
    echo "======================================================"

    mkdir -p "${output_dir}"

    # conda 环境
    eval "$(conda shell.bash hook)"
    conda activate math

    local DATA_NAME="gsm8k,amc23,aime24,minerva_math,gaokao2024_mix,olympiadbench,math"
    local SPLIT="test"
    local NUM_TEST_SAMPLE=-1
    local TEMPERATURE=0.6
    local N_SAMPLING=1
    local TOP_P=1
    local MAX_TOKENS_PER_CALL=16384
    local SEED=0
    local PROMPT_TYPE="qwen25-math-cot"

    cd "${PROJ_ROOT}/${MATH_DIR}"

    python3 -u math_eval.py \
        --model_name_or_path "$model_path" \
        --data_name "$DATA_NAME" \
        --output_dir "${PROJ_ROOT}/${output_dir}" \
        --split "$SPLIT" \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample "$NUM_TEST_SAMPLE" \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --n_sampling "$N_SAMPLING" \
        --top_p "$TOP_P" \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --apply_chat_template \
        --max_tokens_per_call "$MAX_TOKENS_PER_CALL"

    cd "${PROJ_ROOT}"
    conda deactivate
}

# ---------- general 评测 ----------
run_general() {
    local model_path="$1"
    local model_name
    model_name="$(get_model_name "$model_path")"
    local output_dir="${OUTPUT_ROOT}/general/${model_name}"

    echo "======================================================"
    echo "[general] model: ${model_path}"
    echo "[general] output: ${output_dir}"
    echo "======================================================"

    mkdir -p "${output_dir}/math"

    eval "$(conda shell.bash hook)"
    conda activate general

    cd "${PROJ_ROOT}/${GENERAL_DIR}"

    lm_eval --model vllm \
        --model_args "pretrained=${model_path},dtype=bfloat16,max_model_len=32768" \
        --tasks "mmlu_redux_stem_generative,mmlu_redux_other_generative,mmlu_redux_social_sciences_generative,mmlu_redux_humanities_generative" \
        --batch_size 128 \
        --output_path "${PROJ_ROOT}/${output_dir}" \
        --log_samples \
        --num_fewshot 5 \
        --apply_chat_template

    cd "${PROJ_ROOT}"
    conda deactivate
}

# ---------- science 评测 ----------
run_science() {
    local model_path="$1"
    local model_name
    model_name="$(get_model_name "$model_path")"
    local output_dir="${OUTPUT_ROOT}/science/${model_name}"

    echo "======================================================"
    echo "[science] model: ${model_path}"
    echo "[science] output: ${output_dir}"
    echo "======================================================"

    mkdir -p "${output_dir}"

    eval "$(conda shell.bash hook)"
    conda activate science

    cd "${PROJ_ROOT}/${SCIENCE_DIR}"

    python run_eval.py \
        --model-path "$model_path" \
        --test-conf "configs/science.json" \
        --output-dir "${PROJ_ROOT}/${output_dir}" \
        --prompt-format "sft" \
        --temperature 1.0

    cd "${PROJ_ROOT}"
    conda deactivate
}

# ---------- business / law / medicine 评测 ----------
# 三个 domain 共用 simple-evaluation 流水线
run_domain_eval() {
    local domain="$1"
    local model_path="$2"
    local model_name
    model_name="$(get_model_name "$model_path")"
    local output_dir="${OUTPUT_ROOT}/${domain}/${model_name}"

    echo "======================================================"
    echo "[${domain}] model: ${model_path}"
    echo "[${domain}] output: ${output_dir}"
    echo "======================================================"

    mkdir -p "${output_dir}"

    cd "${PROJ_ROOT}/${SIMPLE_EVAL_DIR}"

    bash run_evaluation.sh \
        "$model_path" \
        "$domain" \
        "${PROJ_ROOT}/${output_dir}"

    cd "${PROJ_ROOT}"
}

# ========================= 主循环 =========================
echo "========== run_all_bench.sh 开始 =========="
echo "项目根目录: ${PROJ_ROOT}"
echo "输出根目录: ${OUTPUT_ROOT}"
echo "模型总数:   ${#MODELS[@]}"
echo ""

for entry in "${MODELS[@]}"; do
    domain="${entry%%|*}"
    model_path="${entry#*|}"

    echo "------------------------------------------------------"
    echo ">>> domain=${domain}  model=${model_path}"
    echo "------------------------------------------------------"

    case "$domain" in
        math)
            run_math "$model_path"
            ;;
        general)
            run_general "$model_path"
            ;;
        science)
            run_science "$model_path"
            ;;
        business|law|medicine)
            run_domain_eval "$domain" "$model_path"
            ;;
        *)
            echo "[ERROR] 未知 domain: ${domain}，跳过。支持: math, general, science, business, law, medicine"
            ;;
    esac

    echo ""
done

echo "========== run_all_bench.sh 全部完成 =========="
