#!/bin/bash

PROJECT_PATH='your_project_path'
OPENAI_API_KEY='your_openai_api_key'
ANTHROPIC_API_KEY='your_anthropic_api_key'
GEMINI_API_KEY='your_gemini_api_key'
DEEPSEEK_API_TOKEN='your_deepseek_api_key'
export PROJECT_PATH=${PROJECT_PATH}
export OPENAI_API_KEY=${OPENAI_API_KEY}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export DEEPSEEK_API_TOKEN=${DEEPSEEK_API_TOKEN}

gpu_id=0
sys_msg="On"
dataset="XFinBench"
task="bool"
model="Meta-Llama-3.1-8B-Instruct"
reason_type="CoT"
retri_type="free"
retriever="bm25"
top_k_retr=3
max_token=1024

CUDA_VISIBLE_DEVICES=$gpu_id python ${PROJECT_PATH}/evaluate/main.py \
    --dataset $dataset\
    --task $task\
    --model $model\
    --reason_type $reason_type\
    --sys_msg $sys_msg\
    --retri_type $retri_type\
    --retriever $retriever\
    --top_k_retr $top_k_retr\
    --max_token $max_token