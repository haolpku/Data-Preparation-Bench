conda activate cherry_llm 
CUDA_VISIBLE_DEVICES=0 nohup python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted.jsonl \
    --filter_config configs/cherry_llm.yaml \
    --stage filter \
    --env_name bench > output1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted_1.jsonl \
    --filter_config configs/cherry_llm.yaml \
    --stage filter \
    --env_name bench > output2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted_2.jsonl \
    --filter_config configs/cherry_llm.yaml \
    --stage filter \
    --env_name bench > output3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted_3.jsonl \
    --filter_config configs/cherry_llm.yaml \
    --stage filter \
    --env_name bench > output4.log 2>&1 &

nohup python pipeline.py \
    --train_files dataset/dolly-15k/processed/databricks-dolly-15k_extracted_0.jsonl \
    --filter_config configs/cherry_llm.yaml \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench > output.log 2>&1 &