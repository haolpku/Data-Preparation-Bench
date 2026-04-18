# Data-Preparation-Bench

专注于大模型数据筛选、训练与评估的完整流水线。提供标准化数据处理、多种筛选 Baseline、自动化训练评估流程，助力快速复现实验并对比不同数据策略的效果。

## 1\. 环境准备

### 1.1 基础环境定义
```bash
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd Data-Preparation-Bench
git submodule update --init --recursive
cd data_selection
```
本项目依赖以下第三方开源仓库。执行 `git submodule` 命令会自动根据项目配置拉取对应代码。
| 模块名称 | 所在目录 | 来源仓库 | 备注 |
| :--- | :--- | :--- | :--- |
| **DataFlow-Agent** | `third_party/DataFlow-Agent` | [OpenDCAI/DataFlow-Agent](https://github.com/OpenDCAI/DataFlow-Agent.git) | 专用 `dfa` 环境 |
| **DCLM** | `third_party/DCLM` | [mlfoundations/DCLM](https://github.com/mlfoundations/DCLM.git) | 使用 `bench` 环境 |
| **Cherry_LLM** | `third_party/Cherry_LLM` | [tianyi-lab/Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM.git) | 使用 `bench` 环境 |
| **LlamaFactory** | `third_party/LlamaFactory` | [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory.git) | 使用 `bench` 环境 |
环境定义:
  * **`dfa` 环境**：专用于运行 **DataFlow-Agent (DFA)** 筛选方法。
  * **`bench` 环境**：通用环境，适用于 **DCLM**、**Cherry-LLM** 筛选，以及**数据预处理**、**模型训练**与**模型评估**。

### 1.2 DFA 环境 (dfa)

```bash
conda create -n dfa python==3.11 -y
conda activate dfa

cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
cd ../.. # 回到 data_selection 根目录
```


### 1.3 通用实验环境 (bench)

除 DFA 外的所有组件均运行在此环境。

```bash
conda create -n bench python==3.11 -y
conda activate bench

# 系统依赖 (用于 DCLM 编译)
sudo apt update && sudo apt install cmake build-essential g++-9 -y
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

# 核心组件与集成工具
pip install -r third_party/DCLM/requirements.txt
pip install -r third_party/Cherry_LLM/requirements.txt
pip install -e third_party/LlamaFactory/.
pip install lm-eval ray 'packaging<24.2'
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

### 3.1 方式 1：一键自动化流水线

`pipeline.py` 支持跨环境调度。用户需根据所选的筛选方法激活对应环境启动，程序会根据配置自动调用 `--env_name` 执行后续阶段。

  * **使用 DFA 筛选时：**

<!-- end list -->

```bash
conda activate dfa 
python pipeline.py \
    --filter_config configs/baselines/dfa.yaml \
    --train_config configs/model/qwen2.5_lora_sft.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench
```

  * **使用 DCLM 或 Cherry-LLM 筛选时：**

<!-- end list -->

```bash
conda activate bench
python pipeline.py \
    --filter_config configs/baselines/dclm.yaml \
    --train_config configs/model/qwen2.5_lora_sft.yaml \
    --eval_config configs/eval.yaml \
    --stage filter,train,eval \
    --env_name bench
```

### 3.2 方式 2：分步手动执行

1.  **数据筛选**：需进入该方法指定的 Conda 环境执行。
2.  **模型训练与评估**：统一使用 **`bench`** 环境。

<!-- end list -->

```bash
conda activate bench
python train_eval.py \
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
