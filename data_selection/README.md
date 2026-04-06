# Data-Preparation-Bench 🚀

专注于大模型数据筛选、训练与评估的完整流水线。

---

### 数据构建

## 数据选择环境安装
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
git apply ../../assets/patches/cherry_llm.patch
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
git submodule add --depth 1 https://github.com/hiyouga/LlamaFactory.git third_party/LlamaFactory
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
    --train_file output/data_selection/dfa_alpaca_10pct/selected_data.jsonl \
    --train_config configs/train_qwen2.5.yaml \
    --eval_config configs/eval.yaml \
    --stage train,eval 
```

---

## 📊 实验输出结构 (Output Structure)

为了确保实验的可追溯性，本项目采用“上游数据、下游实验”的分离管理逻辑：

```text
output/
├── data_selection/                    # 【阶段一：数据准备】
│   └── {filter_id}/                   # 示例：dfa_alpaca_10pct
│       ├── selected_data.jsonl        # 过滤后的数据集
│
└── experiments/                       # 【阶段二：训练与验证闭环】
    └── {model}_on_{filter_id}_{time}/ # 唯一实验 ID
        ├── model/                     # 训练产物 (Checkpoints/LoRA weights)
        ├── eval/                      # 该模型在各 Benchmark 上的成绩 (MMLU/GSM8K)
        ├── config/                    # 运行时配置备份 (runtime_config.yaml)
```

