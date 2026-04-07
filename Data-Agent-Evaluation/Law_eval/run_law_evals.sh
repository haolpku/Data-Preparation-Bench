#!/bin/bash


OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/law_eval_$(date +%Y%m%d_%H%M%S).log"


exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo " Law_eval "
echo "TIME: $(date)"
echo "LOG: $LOG_FILE"
echo "==============================================="


echo ">>>   legalbench..."
cd legalbench


CONDA_LIB_PATH=$(dirname $(which python))/../lib

HF_DATASETS_TRUST_REMOTE_CODE=true LD_PRELOAD="$CONDA_LIB_PATH/libstdc++.so.6" python eval.py

cd ..

echo ">>> lex-glue..."
cd lex-glue

bash run_all_tasks.sh

cd ..

echo "==============================================="
echo "ALL BENCHMARK FINISH !"
echo "LOG: $LOG_FILE"
echo "==============================================="