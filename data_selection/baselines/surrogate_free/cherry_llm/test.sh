#!/bin/bash

# 1. 环境变量配置
export HF_HOME="/home/hxy/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0,1  # 指定可见显卡

# 2. 激活环境 (可选)
# source activate bench

# 3. 定义参数
TRAIN_FILE="/home/hxy/filter/dataset/intermediate/openhermes2_5/openhermes.json"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# 4. 后台运行并记录日志
nohup python -u pipeline.py \
    --train_file "$TRAIN_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "../cherry_results" \
    --sample_rate 0.1 \
    > cherry_workflow.log 2>&1 &

echo "🚀 Cherry Data 筛选已启动，请执行 'tail -f cherry_workflow.log' 查看进度。"