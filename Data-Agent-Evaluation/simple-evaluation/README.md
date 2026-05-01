# Simple Evaluation

一键式本地模型评估流水线，支持在单个 GPU 上自动部署 vLLM 服务并依次完成推理、评测与分数统计。

## 环境准备

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
uv sync
```

此外需要安装 [vLLM](https://docs.vllm.ai/) 以启动本地推理服务。

## 数据集

评估所需的 JSONL 数据请从以下链接下载：

https://huggingface.co/datasets/lhpku20010120/Data-Prep-Bench/tree/main/eval

下载后将对应 domain 的 `.jsonl` 文件放入本项目的 `./data/` 目录即可，例如：

```
data/
├── business.jsonl
├── medicine.jsonl
└── law.jsonl
```

## 快速开始

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh <model_path> <domain> <output_dir>
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `model_path` | 本地模型目录（需包含 `model*.safetensors`） | `/path/to/Qwen2.5-7B` |
| `domain` | 评估领域，可选 `business`、`medicine`、`law` | `business` |
| `output_dir` | 结果输出目录 | `./outputs/business_eval` |

### 运行示例

```bash
./run_evaluation.sh /mnt/models/Qwen2.5-7B business ./outputs/qwen_business
```

脚本会依次执行：

1. **启动 vLLM 服务**：自动在后台启动 vLLM，等待 `/health` 就绪
2. **推理（inference）**：调用模型生成回复，输出 `inference_results.jsonl`
3. **评测（judge）**：对推理结果进行 domain-specific 评分，输出 `judge_results.jsonl`
4. **分数统计（extract_score）**：汇总各 benchmark 指标，输出 `calculated_scores.json`

无论运行成功还是失败，脚本退出时都会自动关闭 vLLM 进程。

## 可选配置

可通过环境变量调整 vLLM 与评估行为：

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

示例：

```bash
CUDA_VISIBLE_DEVICES=0 BASE_PORT=9000 MAX_CONCURRENT=128 \
  ./run_evaluation.sh /mnt/models/Qwen2.5-7B medicine ./outputs/med_eval
```

## 输出文件

```
<output_dir>/
├── inference_results.jsonl      # 推理结果
├── eval.log                     # 推理日志
├── calculated_scores.json       # 最终分数汇总
├── judge_out/
│   ├── judge_results.jsonl      # 评测结果
│   ├── judge.log                # 评测日志
│   ├── failed_results.jsonl     # 评测失败样本
│   └── error.jsonl              # 错误记录
└── vllm_logs/
    └── vllm_<model_name>.log    # vLLM 服务日志
```
