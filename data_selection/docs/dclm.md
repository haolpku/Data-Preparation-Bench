# DCLM Pipeline: Data Selection, Training, and Evaluation

This repository provides a standardized pipeline for the **DCLM (Data-Comp for Language Models)** benchmark. The workflow is split into two stages: **Data Filtering** and **Training/Evaluation**, using isolated environments to avoid dependency conflicts.

## 🛠 Prerequisites
* **Conda**: For environment management.
* **Huggingface CLI**: `pip install -U "huggingface_hub[cli]"`

---

## 🚀 Setup Instructions

### 1. Download Required Assets
We must place specific models and banlists into the internal structure of the `data_selection` module.

```bash
# 1.0 Create necessary directories
mkdir -p data_selection/baselines/mappers/banlists/
mkdir -p data_selection/baselines/mappers/enrichers/language_id_enrichment_models/
mkdir -p data_selection/baselines/mappers/enrichers/quality_prediction_enrichment_models/

# 1.1 Download Banned Domains List
huggingface-cli download --repo-type dataset mlfoundations/refinedweb_banned_domains_curated \
    --local-dir assets/refinedweb_banned_domains_curated
mv assets/refinedweb_banned_domains_curated/refinedweb_banned_domains_curated.txt \
   data_selection/baselines/mappers/banlists/

# 1.2 Download Language ID Model (FastText)
huggingface-cli download --repo-type dataset julien-c/fasttext-language-id \
    --local-dir assets/fasttext-language-id
mv assets/fasttext-language-id/lid.176.bin \
   data_selection/baselines/mappers/enrichers/language_id_enrichment_models/

# 1.3 Download Quality Prediction Model
huggingface-cli download --repo-type dataset mlfoundations/fasttext-oh-eli5 \
    --local-dir assets/fasttext-oh-eli5
mv assets/fasttext-oh-eli5/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
   data_selection/baselines/mappers/enrichers/quality_prediction_enrichment_models/fasttext_oh_eli5.bin
```

### 2. Environment Setup
Separate environments ensure compatibility between `kenlm` and `LLaMA-Factory`.

```bash
cd data_selection/

# A. Create Filtering Environment
conda env create -f env_config/dclm.yaml
conda activate dclm
pip install https://github.com/kpu/kenlm/archive/master.zip
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
bash scripts/run_dclm_task.sh
```

### Manual Run
1. **Filtering**: 
   ```bash
   conda activate dclm
   python pipeline.py --config ./configs/dclm.yaml --stage filter
   ```
2. **Train/Eval**: 
   ```bash
   conda activate bench
   python pipeline.py --config ./configs/dclm.yaml --stage train,eval --resume_id <YOUR_RUN_ID>
   ```