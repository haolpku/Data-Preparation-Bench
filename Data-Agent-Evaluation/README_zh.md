# Data-Agent-Evaluation 

## 项目简介

本项目为领域特定大语言模型（Domain-specific LLMs）提供**一体化、自动化、全流程**的评测平台，当前覆盖**金融、医学、法律**三大垂域，整合知识测评与推理测评两类基准（Benchmark）。平台通过统一配置接口与自动化脚本，实现模型接入、样本筛选、评分校准与结果输出的全流程标准化，确保评测的可复现性与公平性。

---

## 1. 垂域与基准说明

三大垂域包含以下基准：

| 垂域 | 包含测评 | 
|------|---------|
| 金融 | FinCDM, XFinBench |
| 医学 | MedCaseReasoning(MedmcQA), MedRBench, MedXpertQA |
| 法律 | legalbench, lex-glue |


---

## 2. 数据集下载

请前往 [Data-Agent-Evaluation](https://www.modelscope.cn/datasets/Dujianzhuo/Data-Agent-Evaluation-Dataset)下载数据集**Data-Agent-Evaluation-Dataset**,解包并置于**Data-Agent-Evaluation**项目根目录文件夹下。运行脚本`set_dataset.py`配置数据文件到子文件夹。
---

## 3. 评测流程与配置

### 3.1 目录结构
各垂域主文件夹下包含：
- `configure_{Domain}_evals.py`：评测配置文件
- `run_{Domain}_evals.sh`：自动化执行脚本

### 3.2 配置说明
修改 `configure_{Domain}_evals.py` 中的全局变量并运行：
```python
NEW_API_KEY = "sk-dummy"                   # 被测模型API_KEY
NEW_BASE_URL = "http://localhost:8000/v1"  # 被测模型API_URL
NEW_MODEL_NAME = "qwen2.5-7b-law"          # 被测模型名称
REFER_MODEL_NAME = "gpt-4o"                # 评分模型名称
REFER_API_BASE_URL = "http://xxx/v1"       # 评分模型API_URL
REFER_API_KEY = "sk-dummy"                 # 评分模型API_KEY
IS_FULL_EVAL = True                        # True全量测试，False少量测试
```

### 3.3 执行与结果
运行 `run_{Domain}_evals.sh` 后，脚本自动调用该垂域下所有基准的评测脚本，完成后在主文件夹生成测试日志，日志中包含各基准的评估指标与汇总结果。

---

## 4. 评测方法与严谨性保障

为保障评测的科学性，平台在流水线中引入以下机制：

1. **过滤非知识性题目**  
   根据各基准的官方任务定义，仅保留与知识或推理能力直接相关的题目，剔除主观性、开放性过强的题型。

2. **指令跟随模型微调**  
   支持对被测模型进行轻量级指令微调，确保模型输出格式与评分模型兼容，降低因输出格式不一致导致的评估偏差。

3. **评分模型校准**  
   采用高性能通用模型（如GPT-4o）作为评分器，对模型生成答案进行客观打分，并提供评分一致性校验（如人工抽样复核），保证评分的可靠性。

4. **语言统一处理**  
   对非英文题目进行标准化翻译或剔除，确保评测内容语言一致性，避免因语言差异引入额外变量。

---

## 5. 结果记录与复现

所有评测结果以结构化日志形式保存在主文件夹中output文件夹下，包含：
- 评测时间、模型版本、配置参数
- 各Benchmark的详细得分与子项指标
- 评分模型输出及原始回答样例

通过保存完整配置与日志，支持后续结果复现与对比分析。

