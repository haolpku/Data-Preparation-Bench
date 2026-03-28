# DFA Pipeline: Data Selection, Training, and Evaluation

This repository provides a standardized pipeline for the **DFA (DataFlow-Agent)** benchmark. The workflow is split into two stages: **Data Filtering** and **Training/Evaluation**, using isolated environments to avoid dependency conflicts.

## 🛠 Prerequisites
* **Conda**: For environment management.
* **Huggingface CLI**: `pip install -U "huggingface_hub[cli]"`

---

## 🚀 Setup Instructions

### Environment Setup
Separate environments ensure compatibility between `DataFlow-Agent` and `LLaMA-Factory`.

```bash
cd data_selection/

# A. Create Filtering Environment
conda env create -f env_config/dfa.yaml
conda activate dfa
git clone --depth 1 https://github.com/OpenDCAI/DataFlow-Agent.git
cd DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
conda deactivate

# B. Setup LLaMA-Factory and Training Environment
conda env create -f env_config/bench.yaml
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory && pip install -e . && cd ..
conda deactivate
```

---

## 🏃 Execution

### Automated Run
The script handles environment switching and ID retrieval automatically.
```bash
bash scripts/run_dfa_task.sh
```

### Manual Run
1. **Filtering**: 
   ```bash
   conda activate dfa
   python pipeline.py --config ./configs/dfa.yaml --stage filter
   ```
2. **Train/Eval**: 
   ```bash
   conda activate bench
   python pipeline.py --config ./configs/dfa.yaml --stage train,eval --resume_id <YOUR_RUN_ID>
   ```