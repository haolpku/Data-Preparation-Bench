# Data-Agent-Evaluation

## 项目简介

本项目为领域特定大语言模型（Domain-specific LLMs）提供**一体化、自动化、全流程**的评测平台，覆盖 **math（数学）、general（通用）、science（科学）、business（商业）、medicine（医学）、law（法律）** 六大领域，整合知识测评与推理测评两类基准（Benchmark）。平台通过统一评测流水线，实现模型部署、推理、评分与分数汇总的全程标准化，确保评测的可复现性与公平性。

---

## 1. 领域与基准说明

| 领域 | 包含测评 |
|------|---------|
| 数学 | GSM8K, AMC23, AIME24, Minerva Math, Gaokao2024 Mix, OlympiadBench |
| 通用 | MMLU-Redux（STEM, Other, Social Sciences, Humanities） |
| 科学 | ScienceQA, ARC-Challenge, OpenBookQA 等 |
| 商业 | FinCDM, XFinBench |
| 医学 | MedCaseReasoning(MedmcQA), MedRBench |
| 法律 | legalbench, lex-glue |

---

## 2. 数据准备

### Math / General / Science

这三个领域依赖各自的评测框架，运行时会**自动下载**所需数据集，无需手动准备。

### Business / Law / Medicine

这三个领域共用 `simple-evaluation` 统一流水线。请从 HuggingFace 下载 JSONL 数据：

```
https://huggingface.co/datasets/lhpku20010120/Data-Prep-Bench/tree/main/eval
```

将对应 `.jsonl` 文件放入 `simple-evaluation/data/` 目录：

```
simple-evaluation/data/
├── business.jsonl
├── law.jsonl
└── medicine.jsonl
```

---

## 3. 评测流程与配置

全部六个领域均可通过项目根目录的 `scripts/run_all_bench.sh` 统一批量评测，也可单独运行各领域的评测。

### 通过 `run_all_bench.sh` 批量评测

编辑脚本内的 `MODELS` 数组：

```bash
MODELS=(
    "math|/path/to/math-model"
    "general|/path/to/general-model"
    "science|/path/to/science-model"
    "business|/path/to/business-model"
    "law|/path/to/law-model"
    "medicine|/path/to/medicine-model"
)
```

然后执行：

```bash
bash Data-Agent-Evaluation/scripts/run_all_bench.sh
```

脚本会依次执行各领域评测，结果保存在 `Data-Agent-Evaluation/bench_outputs/<domain>/<model_name>/` 下。

### 单独运行各领域评测

- **Math / General / Science**：这三个领域有各自的评测框架和入口，单独使用方式请分别参考 `Qwen2.5-Math/`、`lm-evaluation-harness/`、`lm-open-science-evaluation/` 目录内的文档。

- **Business / Law / Medicine**：使用 `simple-evaluation` 统一流水线：

  ```bash
  cd simple-evaluation
  chmod +x run_evaluation.sh
  ./run_evaluation.sh <model_path> <domain> <output_dir>
  ```

  | 参数 | 说明 | 示例 |
  |------|------|------|
  | `model_path` | 本地模型目录（需包含 `model*.safetensors`） | `/path/to/Qwen2.5-7B` |
  | `domain` | 评估领域，可选 `business`、`medicine`、`law` | `business` |
  | `output_dir` | 结果输出目录 | `./outputs/business_eval` |

  脚本会依次执行：
  1. **启动 vLLM 服务**：自动在后台启动 vLLM，等待 `/health` 就绪
  2. **推理（inference）**：调用模型生成回复，输出 `inference_results.jsonl`
  3. **评测（judge）**：对推理结果进行 domain-specific 评分，输出 `judge_results.jsonl`
  4. **分数统计（extract_score）**：汇总各 benchmark 指标，输出 `calculated_scores.json`

  可选环境变量：

  | 环境变量 | 默认值 | 说明 |
  |----------|--------|------|
  | `CUDA_VISIBLE_DEVICES` | — | 指定使用的 GPU，如 `0` 或 `0,1` |
  | `BASE_PORT` | `8002` | vLLM 服务端口 |
  | `TP_SIZE` | `1` | Tensor Parallelism 大小 |
  | `GPU_MEMORY_UTIL` | `0.95` | GPU 显存占用上限 |
  | `MAX_MODEL_LEN` | `16384` | 最大序列长度 |
  | `DTYPE` | `bfloat16` | 模型权重数据类型 |
  | `HEALTH_TIMEOUT` | `600` | vLLM 健康检查超时（秒） |
  | `MAX_CONCURRENT` | `256` | 推理并发请求数 |
  | `JUDGE_CONCURRENCY` | `8` | Judge 并发请求数 |
  | `JUDGE_URL` | — | Judge 模型 API 地址（如需 LLM-as-judge） |
  | `JUDGE_API_KEY` | — | Judge 模型 API Key |
  | `JUDGE_MODEL` | — | Judge 模型名称 |

---

## 4. 评测方法与严谨性保障

针对 **business（商业）、medicine（医学）、law（法律）** 三个垂域，`simple-evaluation` 评测流水线在基准定义的基础上引入了以下严谨性保障机制：

1. **过滤非知识性题目**  
   根据各基准的官方任务定义，仅保留与知识或推理能力直接相关的题目，剔除主观性、开放性过强的题型。

2. **指令跟随模型微调**  
   支持对被测模型进行轻量级指令微调，确保模型输出格式与评分模型兼容，降低因输出格式不一致导致的评估偏差。

3. **LLM-as-Judge 评分校准**  
   采用高性能通用模型（如 GPT-4o）作为评分器，对模型生成答案进行客观打分，并提供评分一致性校验（如人工抽样复核），保证评分的可靠性。

4. **语言统一处理**  
   对非英文题目进行标准化翻译或剔除，确保评测内容语言一致性，避免因语言差异引入额外变量。

**math（数学）、general（通用）、science（科学）** 三个领域则分别通过成熟的开源评测框架（`Qwen2.5-Math`、`lm-evaluation-harness`、`lm-open-science-evaluation`）按各自原始协议进行评测。

---

## 5. 结果记录与复现

全部六个领域的评测结果统一保存在 `Data-Agent-Evaluation/bench_outputs/<domain>/<model_name>/` 下。

**Business / Law / Medicine** 的输出目录包含：
- `inference_results.jsonl` —— 模型生成答案
- `calculated_scores.json` —— 各基准汇总得分
- `judge_out/judge_results.jsonl` —— 详细评分结果
- `judge_out/judge.log` —— 评分日志
- `vllm_logs/vllm_<model>.log` —— vLLM 服务日志

**Math / General / Science** 三个领域的结果（指标、样本、日志等）也由各自框架保存在上述统一路径下。

通过保存完整配置与日志，支持后续结果复现与对比分析。
