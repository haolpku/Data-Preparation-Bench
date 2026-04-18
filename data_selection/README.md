# Data-Preparation-Bench

专注于大模型数据筛选、训练与评估的完整流水线。提供标准化数据处理、多种筛选 Baseline、自动化训练评估流程，助力快速复现实验并对比不同数据策略的效果。

## 1\. 环境准备

### 基础环境
```bash
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd Data-Preparation-Bench
git submodule update --init --recursive
cd data_selection
```
本项目依赖以下第三方开源仓库(对baseline持续更新)。执行 `git submodule` 命令会自动根据项目配置拉取对应代码。
| 模块名称 | 所在目录 | 来源仓库
| :--- | :--- | :--- 
| **DataFlow-Agent** | `third_party/DataFlow-Agent` | [OpenDCAI/DataFlow-Agent](https://github.com/OpenDCAI/DataFlow-Agent.git)
| **DCLM** | `third_party/DCLM` | [mlfoundations/DCLM](https://github.com/mlfoundations/DCLM.git) 
| **Cherry_LLM** | `third_party/Cherry_LLM` | [tianyi-lab/Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM.git)
| **LlamaFactory** | `third_party/LlamaFactory` | [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory.git)


### 数据筛选环境
注意: 考虑到 Baseline 之间可能存在底层库冲突，本项目默认为每个方法推荐独立的环境配置。如果您希望在统一环境下运行，请在安装前核对各模块的依赖限制。
#### DFA

```bash
conda create -n dfa python==3.11 -y
conda activate dfa

cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
cd ../.. # 回到 data_selection 根目录
```
#### DCLM
```bash
# DCLM
conda create -n dclm python==3.11 -y
conda activate dclm
cd third_party/DCLM
pip install -r requirements.txt

sudo apt update
sudo apt install cmake build-essential g++-9 -y
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

python setup.py install
pip install ray
cd ../.. # 回到 data_selection 根目录
```

#### Cherry_LLM
```bash
conda create -n cherry_llm python==3.11 -y
conda activate cherry_llm
cd third_party/Cherry_LLM
pip install -r requirements.txt
cd ../LlamaFactory && pip install -e .
cd ../.. # 回到 data_selection 根目录
```

### 训练与评估环境
```bash
conda create -n bench python==3.11 -y
conda activate bench
cd third_party/LlamaFactory
pip install -e .
pip install lm_eval
cd ../.. # 回到 data_selection 根目录
```

-----

## 2\. 数据构建

将原始数据集统一转为 Alpaca 格式（`instruction`, `input`, `output`）。执行预处理前需激活 **`bench`** 环境。

### 2.1 快速准备示例

```bash
conda activate bench

# 下载并转换 OpenHermes 2.5
huggingface-cli download --repo-type dataset teknium/OpenHermes-2.5 --local-dir dataset/OpenHermes-2.5
python preprocess_data.py --train_file dataset/OpenHermes-2.5/openhermes2_5.json
# 或者处理整个数据集文件夹
python preprocess_data.py --train_file dataset/OpenHermes-2.5
```

### 2.2 已验证数据集支持

| 数据集 | HuggingFace 路径  
| :--- | :--- |
| **LMSYS-Chat-1M** | `lmsys/lmsys-chat-1m` 
| **WildChat** | `allenai/WildChat` 
| **OpenHermes 2.5** | `teknium/OpenHermes-2.5` 
| **Dolly-15K** | `databricks/databricks-dolly-15k` 
| **WizardLM 70K** | `WizardLMTeam/WizardLM_evol_instruct_70k` |

-----

## 3\. 运行指南

### 3.1 方式 1:一键自动化流水线

`pipeline.py` 支持跨环境调度。用户需根据所选的筛选方法激活对应环境启动，程序会根据配置自动调用 `--env_name` 执行后续阶段。训练和评估会自动调用筛选阶段得到的数据集。
* **自动化衔接**:训练阶段会自动解析筛选产出的路径索引。对于 **DFA** 等多步筛选方法，程序会根据 `filter_config` 中的 `step` 自动加载对应的中间结果。
* **配置要求**:
    * 待过滤数据集路径需写在 `filter_config` 的 `train_file` 字段中。
    * **DFA 特有**: 需额外提供 `test_train_file`（通常为原文件名加 `_sample` 后缀）。
* **限制**:筛选阶段目前仅支持处理单个数据集文件。
```bash
conda activate dfa 
python pipeline.py \
    --filter_config configs/baselines/dfa.yaml \
    --train_config configs/model/qwen2.5_lora_sft.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench
```

### 3.2 方式 2:分步手动执行
通过 `--stage` 参数可以灵活控制执行流。
#### 1. 数据筛选:
* 须在各 Baseline 指定的 Conda 环境中执行。

* **限制**: 目前仅支持单个文件输入。
```bash
conda activate bench
python pipeline.py \
    --stage filter \
    --filter_config configs/baselines/dfa.yaml 
```
#### 2. **模型训练与评估**:
统一在 **`bench`** 环境下执行。

* **多文件支持**:此模式下支持在 `--train_files` 中同时传入多个数据集文件进行训练。
* **⚠️ DFA 特殊说明**:DFA 产出的 `final_filtered_file.jsonl` 仅为**路径索引文件**。手动执行训练时，请直接在 `--train_files` 中传入具体的数据文件（如 `merge_step1.jsonl`）。
```bash
conda activate bench
python pipeline.py \
    --stage train,eval \
    --train_files [FILTERED_FILE1] [FILTERED_FILE2] \
    --train_config configs/model/qwen2.5_lora_sft.yaml \
    --eval_config configs/eval.yaml
```
-----

## 4\. 实验输出结构

项目通过实验 ID 隔离不同阶段的产物，确保数据与模型的一致性。

```text
output/
├── data/                    # 筛选后的数据集
│   └── exp_abc123/          # 筛选实验 ID
│       └── final_filtered_file.jsonl       # 筛选后的数据
└── experiments/             # 训练与评估实验
    └── exp_xyz999/          # 实验 ID
        ├── train/           # 训练产物
        │   ├── dataset/     # LlamaFactory 临时映射数据
        │   └── model/       # 权重文件
        └── eval/            # 评估产物
            └── results.json # 评估指标
```
