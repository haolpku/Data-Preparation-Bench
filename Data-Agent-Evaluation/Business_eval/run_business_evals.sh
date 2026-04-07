#!/bin/bash

OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/business_eval_$(date +%Y%m%d_%H%M%S).log"


exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo " Business_eval "
echo "TIME: $(date)"
echo "LOG: $LOG_FILE"
echo "==============================================="


echo ">>>  FinCDM..."


cd FinCDM
python src/run_task_eval.py
python src/referee_review.py
cd ..



echo ">>>  XFinbench..."


cd XFinBench
cd evaluate

python main.py
python rejudge.py


cd ..
cd ..






echo "==============================================="
echo "ALL BENCHMARK FINISH !"
echo "LOG: $LOG_FILE"
echo "==============================================="