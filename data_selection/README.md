
# Data-Selection

专注于大模型数据筛选、训练与评估的完整流水线。提供标准化数据处理、多种筛选 Baseline、自动化训练评估流程，助力快速复现实验并对比不同数据筛选策略的效果。

---

## 1. 项目简介

本项目核心目标是提供一套**标准化、可复现、易扩展**的大模型数据准备流水线，涵盖「数据构建 $\rightarrow$ 数据筛选 $\rightarrow$ 模型训练 $\rightarrow$ 模型评估」全流程。项目支持多种主流数据筛选 Baseline，统一数据格式为 Alpaca，旨在降低大模型数据准备与实验复现的门槛。

---

## 2. 环境准备

### 2.1 基础依赖
```bash
# 克隆本项目
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd data_selection
```

### 2.2 数据筛选 Baseline 环境
根据需要使用的算法安装对应环境。

#### 2.2.1 DataFlow-Agent (DFA)
```bash
conda create -n dfa python==3.11 -y
conda activate dfa
git submodule add https://github.com/OpenDCAI/DataFlow-Agent.git third_party/DataFlow-Agent
cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
```

#### 2.2.2 DataComp for LM (DCLM)
```bash
conda create -n dclm python==3.11 -y
conda activate dclm
git submodule add --depth 1 https://github.com/mlfoundations/DCLM.git third_party/DCLM
cd third_party/DCLM
pip install -r requirements.txt

# 系统依赖 (Ubuntu)
sudo apt update
sudo apt install cmake build-essential g++-9 -y
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

python setup.py install
pip install ray
```

#### 2.2.3 Cherry_LLM
```bash
conda create -n cherry_llm python==3.11 -y 
conda activate cherry_llm
git submodule add https://github.com/tianyi-lab/Cherry_LLM.git third_party/Cherry_LLM
cd third_party/Cherry_LLM
pip install -r requirements.txt
cd ../LlamaFactory && pip install -e .
```

### 2.3 训练与评估环境
```bash
conda create -n bench python=3.11 -y
conda activate bench
git submodule add --depth 1 https://github.com/hiyouga/LlamaFactory.git third_party/LlamaFactory
cd third_party/LlamaFactory && pip install -e .
pip install lm-eval
```

---

## 3. 数据构建

将原始数据集统一转为 Alpaca 格式（`instruction` / `input` / `output`）。

### 3.1 准备原始数据
将数据放入 `data_selection/dataset` 目录。
```bash
huggingface-cli download --repo-type dataset --resume-download teknium/OpenHermes-2.5 --local-dir data_selection/dataset/teknium/OpenHermes-2.5
```

### 3.2 预处理
运行脚本将数据转换为 Alpaca 格式，结果保存在 `processed/` 目录下。

* **处理单个文件：**
    ```bash
    conda activate bench
    python preprocess_data.py --train_file dataset/teknium/OpenHermes-2.5/openhermes2_5.json
    ```
* **处理文件夹：**
    ```bash
    python preprocess_data.py --train_file dataset/lmsys-chat-1m/
    ```

---

### 3.3 常用数据集准备示例 (Common Datasets Setup)
项目已验证并支持以下 5 个主流数据集的快速构建。请确保已激活 bench 环境。

#### LMSYS-Chat-1M
这类数据通常包含多轮对话，处理时会自动提取并转换为标准 Alpaca 格式。

```Bash
# 下载数据
huggingface-cli download --repo-type dataset lmsys/lmsys-chat-1m --local-dir dataset/lmsys-chat-1m

# 预处理整个目录
python preprocess_data.py --train_file dataset/lmsys-chat-1m/
```
#### WildChat
```Bash
# 下载数据
hf download --repo-type dataset allenai/WildChat --local-dir dataset/WildChat

# 预处理整个目录
python preprocess_data.py --train_file dataset/WildChat/data
```
#### OpenHermes 2.5
```Bash
hf download --repo-type dataset teknium/OpenHermes-2.5 --local-dir dataset/OpenHermes-2.5

# 处理指定 json 文件
python preprocess_data.py --train_file dataset/OpenHermes-2.5/openhermes2_5.json
```
#### Databricks-Dolly-15K
```Bash
hf download --repo-type dataset databricks/databricks-dolly-15k --local-dir dataset/dolly-15k

python preprocess_data.py --train_file dataset/dolly-15k/databricks-dolly-15k.jsonl
```
#### WizardLM Evol-Instruct 70K
```Bash
hf download --repo-type dataset WizardLMTeam/WizardLM_evol_instruct_70k --local-dir dataset/wizardlm-70k

python preprocess_data.py --train_file dataset/wizardlm-70k/alpaca_evol_instruct_70k.json
```

## 4. 快速开始
### 4.1 方式 1：一键执行（全流程）
适合单个数据集的自动化流水线，程序会自动处理环境切换。
```bash
conda activate dfa  # 激活任一筛选环境
python pipeline.py \
    --filter_config configs/baselinesdfa.yaml \
    --train_config configs/model/qwen2.5_lora_sft.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench
```

### 4.2 方式 2：分步执行
适合多阶段调试或多数据集筛选场景。

1.  **数据筛选：**
只接受过滤单个数据集文件。该模块支持自动化调用不同的 Baseline 环境。系统会根据 --filter_config 中的 method 字段，自动在对应的 Conda 环境中执行 third_party/{method}/start.py。配置文件中的 args 字段会自动转换为命令行参数传递给子进程。不同 Baseline 的特有参数（如 API Key、过滤阈值等）均可通过 YAML 灵活配置，无需修改主程序代码。
    ```bash
    conda activate dfa
    python pipeline.py --filter_config configs/baselines/dfa.yaml --stage filter
    ```
2.  **模型训练：**
可接受多个数据集文件用于训练
    ```bash
    conda activate bench
    python train_eval.py --train_files [FILTERED_FILE1] [FILTERED_FILE2] --train_config configs/model/qwen2.5_lora_sft.yaml 
    ```
3.  **模型评估：**
    ```bash
    python train_eval.py --eval_config configs/eval.yaml
    ```
3.  **模型训练和评估评估：**
    ```bash
    python train_eval.py  --train_files [FILTERED_FILE] [FILTERED_FILE2] --train_config configs/model/qwen2.5_lora_sft.yaml  --eval_config configs/eval.yaml
    ```
---

## 5. 实验输出结构

项目采用「上游数据、下游实验」分离管理模式，确保实验的可追溯性。

```text
output/
├── data/                    # 筛选后的数据集
│   └── exp_abc123/          # 筛选实验 ID
│       ├── data.jsonl       # 筛选后的数据
│       └── filter_config.yaml
└── experiments/             # 训练与评估实验
    └── exp_xyz999/          # 实验 ID
        ├── train/           # 训练产物（Model/LoRA）
        │   ├── dataset/     # LlamaFactory 临时映射数据
        │   └── model/       # 权重文件
        └── eval/            # 评估产物
            └── results.json # 评估指标
```
