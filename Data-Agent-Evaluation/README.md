# Data-Agent-Evaluation

## Project Overview

This project provides an **integrated, automated, end-to-end** evaluation platform for domain-specific large language models (LLMs), currently covering three major domains: **finance, medicine, and law**. It integrates both knowledge assessment and reasoning assessment benchmarks. The platform standardizes the entire evaluation process—from model integration and sample selection to scoring calibration and result output—through a unified configuration interface and automated scripts, ensuring reproducibility and fairness in evaluations.

---

## 1. Domains and Benchmarks

The three major domains include the following benchmarks:

| Domain | Included Evaluations |
|--------|----------------------|
| Finance | FinCDM, XFinBench |
| Medicine | MedCaseReasoning (MedmcQA), MedRBench, MedXpertQA |
| Law | legalbench, lex-glue |

---

## 2. Dataset Download

Please download the dataset **Data-Agent-Evaluation-Dataset** from [Data-Agent-Evaluation](https://www.modelscope.cn/datasets/Dujianzhuo/Data-Agent-Evaluation-Dataset). Extract the package and place it in the root directory of the **Data-Agent-Evaluation** project. Run the script `set_dataset.py` to configure the data files into subfolders.

---

## 3. Evaluation Process and Configuration

### 3.1 Directory Structure
The main folder for each domain contains:
- `configure_{Domain}_evals.py`: Evaluation configuration file
- `run_{Domain}_evals.sh`: Automated execution script

### 3.2 Configuration Instructions
Modify the global variables in `configure_{Domain}_evals.py` and run it:
```python
NEW_API_KEY = "sk-dummy"                   # API key for the model under test
NEW_BASE_URL = "http://localhost:8000/v1"  # API URL for the model under test
NEW_MODEL_NAME = "qwen2.5-7b-law"          # Name of the model under test
REFER_MODEL_NAME = "gpt-4o"                # Name of the scoring model
REFER_API_BASE_URL = "http://xxx/v1"       # API URL for the scoring model
REFER_API_KEY = "sk-dummy"                 # API key for the scoring model
IS_FULL_EVAL = True                        # True for full evaluation, False for small-scale test
```

### 3.3 Execution and Results
After running `run_{Domain}_evals.sh`, the script automatically invokes the evaluation scripts for all benchmarks within that domain. Upon completion, test logs are generated in the main folder, containing evaluation metrics and summary results for each benchmark.

---

## 4. Evaluation Methodology and Rigor Assurance

To ensure scientific rigor, the platform incorporates the following mechanisms into the evaluation pipeline:

1. **Filtering Non-Knowledge Questions**  
   Based on the official task definitions of each benchmark, only questions directly related to knowledge or reasoning capabilities are retained, while subjective or overly open-ended question types are excluded.

2. **Instruction-Following Model Fine-Tuning**  
   Supports lightweight instruction fine-tuning of the model under test to ensure its output format is compatible with the scoring model, reducing evaluation bias caused by format inconsistencies.

3. **Scoring Model Calibration**  
   Employs high-performance general-purpose models (e.g., GPT-4o) as scorers to objectively score model-generated answers, with scoring consistency checks (e.g., manual sampling review) to ensure reliability.

4. **Language Standardization**  
   Non-English questions are either translated to a standard language or excluded to maintain linguistic consistency in the evaluation content, avoiding additional variables introduced by language differences.

---

## 5. Result Recording and Reproducibility

All evaluation results are saved in structured log files in the `output` folder under the main directory, including:
- Evaluation time, model version, and configuration parameters
- Detailed scores and sub-metrics for each benchmark
- Scoring model outputs and sample original answers

By preserving complete configurations and logs, the platform supports subsequent result reproduction and comparative analysis.