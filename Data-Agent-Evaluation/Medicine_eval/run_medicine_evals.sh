#!/bin/bash

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/medicine_eval_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo " Medicine_eval "
echo "TIME: $(date)"
echo "LOG FILE: $LOG_FILE"
echo "==============================================="


CONDA_LIB_PATH=$(dirname $(which python))/../lib


echo ">>>   MedCaseReasoning (Inference & Eval)..."

cd MedCaseReasoning

bash run_all.sh

cd ..


echo ">>>  MedmcQA..."

cd MedmcQA

LD_PRELOAD="$CONDA_LIB_PATH/libstdc++.so.6" python llm_eval_medmcqa.py 

cd ..





echo ">>> MedRBench ..."

cd MedRBench
cd src
cd Inference


python oracle_diagnose.py

cd ..
cd Evaluation


eval_scripts=(
    "oracle_diagnose_accuracy.py"
)

for script in "${eval_scripts[@]}"; do
    echo "Eval: $script"
    python "$script"
done

cd ..
cd ..

python src/Evaluation/diagnose_metrics_summary.py
cd ..

echo "==============================================="
echo "ALL BENCHMARK FINISH !"
echo "LOG: $LOG_FILE"
echo "==============================================="