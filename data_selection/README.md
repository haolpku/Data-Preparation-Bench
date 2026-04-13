# Data\-Preparation\-Bench 🚀

专注于大模型数据筛选、训练与评估的完整流水线，提供标准化数据处理、多种筛选Baseline、自动化训练评估流程，助力快速复现实验、对比不同数据筛选策略的效果。

---

## 📋 项目简介

本项目核心目标：提供一套**标准化、可复现、易扩展**的大模型数据准备流水线，涵盖「数据构建→数据筛选→模型训练→模型评估」全流程，支持多种主流数据筛选Baseline，统一数据格式为Alpaca，降低大模型数据准备与实验复现的门槛。



## 🔧 环境准备

请先确保你的环境已安装 `conda` 和 `git`，以下是全流程环境配置，按步骤执行即可。

### 1\. 基础依赖（全局通用）

```bash
# 克隆本项目（若未克隆）
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd Data-Preparation-Bench/data_selection

# 安装git子模块（核心依赖）
git submodule init
git submodule update
```

### 2\. 数据筛选Baseline环境

根据需要使用的数据筛选Baseline，安装对应环境，每个Baseline独立环境，避免依赖冲突。目前只实现三个方法，持续更新中...

#### 2\.1 DataFlow\-Agent

```bash
cd Data-Preparation-Bench/data_selection # 回到该目录
# 创建并激活环境
conda create -n dfa python==3.11 -y
conda activate dfa

# 安装依赖
git submodule add git@github.com:OpenDCAI/DataFlow-Agent.git third_party/DataFlow-Agent
cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
```

#### 2\.2 DataComp for LM

```bash
cd Data-Preparation-Bench/data_selection # 回到该目录
# 创建并激活环境
conda create -n dclm python==3.10 -y
conda activate dclm

# 安装依赖
git submodule add git@github.com:mlfoundations/DCLM.git third_party/DCLM
cd third_party/DCLM
git apply ../../assets/patches/dclm.patch
pip install -r requirements.txt

# 安装系统依赖（Ubuntu）
sudo apt update
sudo apt install cmake build-essential g++-9 -y
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

# 编译安装
python setup.py install
pip install ray
```

#### 2\.3 Cherry\_llm

```bash
cd Data-Preparation-Bench/data_selection # 回到该目录
# 创建并激活环境
conda create -n cherry_llm python==3.10 -y 
conda activate cherry_llm

# 安装依赖
git submodule add git@github.com:tianyi-lab/Cherry_LLM.git third_party/Cherry_LLM
cd third_party/Cherry_LLM
git apply ../../assets/patches/cherry_llm.patch
pip install -r requirements.txt
```

### 3\. 训练与评估环境（必装）

```bash
cd Data-Preparation-Bench/data_selection # 回到该目录
# 创建并激活环境
conda create -n bench python=3.11 -y
conda activate bench

# 安装核心训练依赖（LlamaFactory）
git submodule add --depth 1 git@github.com:hiyouga/LlamaFactory.git third_party/LlamaFactory
cd third_party/LlamaFactory && pip install -e .

# 安装评估库
pip install lm-eval
```

提示：所有环境安装完成后，可通过 `conda env list` 查看已创建的环境，后续操作需根据步骤切换对应环境。

---

## 📥 数据构建（标准化数据集）

将任意数据集统一转为Alpaca格式（`instruction/input/output`），为后续筛选、训练提供标准化输入。

### 1\. 准备原始数据集

将待处理的数据集文件/文件夹，放入 `data_selection/dataset` 目录下。示例（以OpenHermes\-2\.5为例）：

```bash
# 从Hugging Face下载数据集（示例）
huggingface-cli download --repo-type dataset --resume-download teknium/OpenHermes-2.5 --local-dir data_selection/dataset/teknium/OpenHermes-2.5
```

### 2\. 预处理数据集

运行预处理脚本，将原始数据集转为Alpaca格式，输出文件将自动保存至 `原数据集目录/processed/` 下。

#### 2\.1 处理单个文件

```bash
cd data_selection
conda activate bench # 预处理依赖训练环境
python preprocess_data.py --train_file dataset/teknium/OpenHermes-2.5/openhermes2_5.json
```

输出：`dataset/teknium/OpenHermes-2.5/processed/openhermes2_5_extracted.jsonl`

#### 2\.2 处理整个文件夹

```bash
cd data_selection
conda activate bench
python preprocess_data.py --train_file dataset/lmsys-chat-1m/
```

输出：文件夹内所有数据集文件，将分别生成`processed/文件名_extracted.jsonl`（如 train1_extracted.jsonl）。

Alpaca格式示例：
```
{
    instruction: "请介绍Chou Chemical公司",
    input: "POB 241025 Charlotte, NC 28224 United States",
    output: "Chou Chemical Co\. is a chemical company located in Charlotte, NC\.\.\."
}
```
---

## 🚀 快速开始（核心流程）

整个流水线分为「数据筛选→模型训练→模型评估」三个阶段，支持**一键执行**或**分步执行**，按需选择即可。

### 前提条件

- 已完成「环境准备」和「数据构建」步骤

- 预处理后的Alpaca格式数据集路径：`dataset/xxx/processed/yyy_extracted.jsonl`

- 切换到对应环境（筛选用筛选环境，训练/评估用bench环境）

### 方式1：一键执行（筛选\+训练\+评估）

适合单个数据集，一键完成全流程，自动切换环境。

```bash
cd data_selection
conda activate dfa # 切换到你使用的筛选Baseline环境（dfa/dclm/cherry_llm）

python pipeline.py  \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --filter_config configs/dfa.yaml  \ # 筛选配置文件（对应所选Baseline）
    --train_config configs/train_qwen2.5.yaml \ # 训练配置文件
    --eval_config configs/eval.yaml \ # 评估配置文件
    --stage filter,train,eval \ # 执行阶段（全流程）
    --env_name bench # 训练/评估的环境名（默认bench）
```

### 方式2：分步执行（更灵活）

适合多个数据集筛选、或需要单独调试某一步骤的场景。

#### 步骤1：数据筛选

```bash
cd data_selection
conda activate dfa # 切换到筛选Baseline环境（dfa/dclm/cherry_llm）

python pipeline.py  \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --filter_config configs/dfa.yaml  \
    --stage filter
```

筛选结果将保存至 `output/data/exp\_xxx/` 目录。

#### 步骤2：模型训练（可选）

```bash
cd data_selection
conda activate bench # 切换到训练环境

# 方式A：使用train_eval.py
python train_eval.py \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --train_config configs/train_qwen2.5.yaml \
    --env_name bench

# 方式B：使用pipeline.py
python pipeline.py  \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --train_config configs/train_qwen2.5.yaml \
    --stage train
```

训练模型将保存至 `output/experiments/exp\_xxx/train/` 目录。

#### 步骤3：模型评估（可选）

```bash
cd data_selection
conda activate bench # 切换到评估环境

# 方式A：使用train_eval.py（直接评估）
python train_eval.py \
    --eval_config configs/eval.yaml 

# 方式B：使用pipeline.py（直接评估）
python pipeline.py  \
    --eval_config configs/eval.yaml \
    --stage eval

# 方式C：训练后直接评估（推荐）
python train_eval.py \
    --train_files dataset/databricks/databricks-dolly-15k/processed/databricks-dolly-15k_alpaca.jsonl \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml 
```

评估结果将保存至 `output/experiments/exp\_xxx/eval/results\.json`。

注意：

- 只训练不评估：不传入`\-\-eval\_config` 参数

- 只评估不训练：不传入 `\-\-train\_files` 和 `\-\-train\_config` 参数，需在 `eval\_config` 中指定待评估模型路径

- `\-\-filter\_config` 需与所选筛选Baseline对应（dfa/dclm/cherry\_llm）

---

## 📊 实验输出结构（可追溯）

项目采用「上游数据、下游实验」分离管理，所有输出文件按规范结构组织，便于实验复现、结果对比和版本管理。

```text
output/
├── data/                     # 过滤后数据集（只读、可复用，供训练调用）
│   ├── exp_abc123/           # 单次筛选实验ID（自动生成）
│   │   ├── data.jsonl        # 筛选后的Alpaca格式数据集
│   │   └── filter_config.yaml# 筛选时使用的配置文件（复现用）
│   └── ...                   # 多个筛选实验结果
│
└── experiments/              # 训练/评估实验（每个实验一个独立ID）
    └── exp_xyz999/           # 单次训练+评估实验ID（自动生成）
        ├── train/            # 训练相关输出
        │   ├── dataset/      # LlamaFactory专用临时数据集目录
        │   │   ├── exp_abc123.jsonl   # 从data/复制的筛选后数据集
        │   │   └── dataset_info.json  # 自动生成的数据集信息
        │   ├── model/        # 训练输出的LoRA适配器/模型权重
        │   └── merged/       # 合并后的完整模型（可选）
        └── eval/             # 评估相关输出
             └── results.json # 多指标评估结果（可直接用于对比分析）
```
