# Data-Preparation-Bench 🚀

专注于大模型数据筛选、训练与评估的完整流水线。

---

## 数据构建
对数据集统一为Alpaca格式, 方便不同baseline直接处理以及后期用来训练模型. 可以按照下面预处理数据或者自己实现这一步.
### 选择数据
例如选择OpenHermes-2.5数据集作为待过滤数据集，将数据集文件放在dataset文件夹内:
```bash
cd data_selection
mdir dataset
huggingface-cli download --repo-type dataset --resume-download teknium/OpenHermes-2.5 --local-dir teknium/OpenHermes-2.5
```
### 预处理数据集
输出为dataset/xxx/原数据集文件名_extracted.jsonl, 例如
格式内容
{
    "instruction": "",
    "input": "",
    "output": "",
}
```bash
python preprocess_data.py --train_file dataset/teknium/OpenHermes-2.5/openhermes2_5.json
```

## 环境安装
目前实现的baseline包括DataFlow-Agent, DataComp for LM以及Cherry_llm

### DataFlow-Agent
```bash
cd data_selection
conda create -n dfa python==3.11 -y
conda activate dfa

git submodule add git@github.com:OpenDCAI/DataFlow-Agent.git third_party/DataFlow-Agent
cd third_party/DataFlow-Agent
pip install -r requirements-data.txt
pip install -e .
```
### DataComp for LM
```bash
cd data_selection
conda create -n dclm python==3.10 -y
conda activate dclm

git submodule add git@github.com:mlfoundations/DCLM.git third_party/DCLM
cd third_party/DCLM
git apply ../../assets/patches/dclm.patch
pip install -r requirements.txt

sudo apt update
sudo apt install cmake build-essential g++-9 -y
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
python setup.py install
pip install ray

```
### Cherry_llm
```bash
cd data_selection
conda create -n cherry_llm python==3.10 -y 
conda activate cherry_llm

git submodule add git@github.com:tianyi-lab/Cherry_LLM.git third_party/Cherry_LLM
cd third_party/Cherry_LLM
git apply ../../assets/patches/cherry_llm.patch
pip install -r requirements.txt

```

## 🛠️ 训练和评估环境安装 (Installation)
```bash
# 1. 创建并激活环境
conda create -n bench python=3.11 -y
conda activate bench

# 2. 安装核心依赖 LlamaFactory
git submodule add --depth 1 git@github.com:hiyouga/LlamaFactory.git third_party/LlamaFactory
cd third_party/LlamaFactory && pip install -e .

# 3. 安装评估库
pip install lm-eval
```

## 🚀 快速开始 (Quick Start)

整个流水线分为 **数据筛选** 和 **一键训练评估** 两个阶段。
一键筛选和训练评估
```bash
python pipeline.py  \
    --filter_config configs/dfa.yaml  \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
```

分步执行
### 第一阶段：数据筛选 (Data Selection)
运行指定的基线算法从原始数据中选出黄金子集。
```bash
python data_selection.py --config configs/dfa.yaml
```

### 第二阶段：训练与评估 (Train & Eval)
使用筛选后的数据进行自动化的训练和多指标评估。
```bash
python train_eval.py \
    --train_files output/data_selection/dfa_alpaca_10pct/selected_data.jsonl \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
    --stage train,eval 
```

---

## 📊 实验输出结构 (Output Structure)

为了确保实验的可追溯性，本项目采用“上游数据、下游实验”的分离管理逻辑：
```text
output/
├── data/                     # 过滤后数据集（只读、可复用）
│   ├── exp_abc123/           # 数据集 A 过滤结果
│   │   ├── data.jsonl
│   │   └── filter_config.yaml
│   ├── exp_def456/           # 数据集 B 过滤结果
│   │   ├── data.jsonl
│   │   └── filter_config.yaml
│   └── exp_ghi789/           # 数据集 C 过滤结果
│       ├── data.jsonl
│       └── filter_config.yaml
│
└── experiments/              # 训练 / 评估实验
    └── exp_xyz999/           # 一次训练实验 = 一个独立 ID
        ├── train/
        │   ├── dataset/      # LLaMA Factory 专用临时目录
        │   │   ├── exp_abc123.jsonl   # 从 data/exp_abc123 复制并重命名
        │   │   ├── exp_def456.jsonl   # 从 data/exp_def456 复制并重命名
        │   │   └── dataset_info.json  # 自动生成
        │   ├── model/        # 训练输出（LoRA / 适配器）
        │   └── merged/       # 合并后的完整模型
        └── eval/             # 评估结果
             └── results.json # 在数据集A和B训练后的的统一评估指标
```

