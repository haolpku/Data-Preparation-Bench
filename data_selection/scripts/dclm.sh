conda activate dclm  # 激活任一筛选环境
python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted.jsonl \
    --filter_config configs/dclm.yaml \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench