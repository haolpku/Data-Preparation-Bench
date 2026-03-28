# Cherry_LLM Pipeline: Data Selection, Training, and Evaluation

This repository provides a standardized pipeline for the **Cherry_LLM (Data-Comp for Language Models)** benchmark. The workflow is split into two stages: **Data Filtering** and **Training/Evaluation**, using isolated environments to avoid dependency conflicts.

## 🛠 Prerequisites
* **Conda**: For environment management.
* **Huggingface CLI**: `pip install -U "huggingface_hub[cli]"`

---

## 🚀 Setup Instructions

### Environment Setup
Separate environments ensure compatibility between `kenlm` and `LLaMA-Factory`.

```bash
cd data_selection/

# A. Create Filtering Environment
conda env create -f env_config/cherry_llm.yaml
conda activate cherry_llm
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
bash scripts/run_cherrry_task.sh
```

### Manual Run
1. **Filtering**: 
   ```bash
   conda activate cherry_llm
   python pipeline.py --config ./configs/cherry_llm.yaml --stage filter
   ```
2. **Train/Eval**: 
   ```bash
   conda activate bench
   python pipeline.py --config ./configs/cherry_llm.yaml --stage train,eval --resume_id <YOUR_RUN_ID>
   ```