
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
git submodule add git@github.com:OpenDCAI/DataFlow-Agent.git third_party/DataFlow-Agent
cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
```

#### 2.2.2 DataComp for LM (DCLM)
```bash
conda create -n dclm python==3.11 -y
conda activate dclm
git submodule add --depth 1 git@github.com:mlfoundations/DCLM.git third_party/DCLM
cd third_party/DCLM
git apply ../../assets/patches/dclm.patch
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
git submodule add git@github.com:tianyi-lab/Cherry_LLM.git third_party/Cherry_LLM
cd third_party/Cherry_LLM
git apply ../../assets/patches/cherry_llm.patch
pip install -r requirements.txt
```

### 2.3 训练与评估环境
```bash
conda create -n bench python=3.11 -y
conda activate bench
git submodule add --depth 1 git@github.com:hiyouga/LlamaFactory.git third_party/LlamaFactory
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

## 4. 快速开始



### 4.1 方式 1：一键执行（全流程）
适合单个数据集的自动化流水线，程序会自动处理环境切换。
```bash
conda activate dfa  # 激活任一筛选环境
python pipeline.py \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --filter_config configs/dfa.yaml \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench
```

### 4.2 方式 2：分步执行
适合多阶段调试或多数据集筛选场景。

1.  **数据筛选：**
只接受过滤单个数据集文件
    ```bash
    conda activate dfa
    python pipeline.py --train_files [FILE_PATH] --filter_config configs/dfa.yaml --stage filter
    ```
2.  **模型训练：**
可接受多个数据集文件用于训练
    ```bash
    conda activate bench
    python train_eval.py --train_files [FILTERED_FILE1] [FILTERED_FILE2] --train_config configs/train_qwen2.5.yaml 
    ```
3.  **模型评估：**
    ```bash
    python train_eval.py --eval_config configs/eval.yaml
    ```
3.  **模型训练和评估评估：**
    ```bash
    python train_eval.py  --train_files [FILTERED_FILE] [FILTERED_FILE2] --train_config configs/train_qwen2.5.yaml  --eval_config configs/eval.yaml
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
