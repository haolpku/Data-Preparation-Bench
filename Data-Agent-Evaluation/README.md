# Data-Agent-Evaluation

## Project Overview

This project provides an **integrated, automated, end-to-end** evaluation platform for domain-specific large language models (LLMs), covering six major domains: **math, general, science, business, medicine, and law**. It integrates both knowledge assessment and reasoning assessment benchmarks. The platform standardizes the entire evaluation process—from model deployment, inference, judging to score aggregation—through a unified evaluation pipeline, ensuring reproducibility and fairness in evaluations.

---

## 1. Domains and Benchmarks

| Domain | Included Evaluations |
|--------|----------------------|
| Math | GSM8K, AMC23, AIME24, Minerva Math, Gaokao2024 Mix, OlympiadBench |
| General | MMLU-Redux (STEM, Other, Social Sciences, Humanities) |
| Science | ScienceQA, ARC-Challenge, OpenBookQA, etc. |
| Business | FinCDM, XFinBench |
| Medicine | MedCaseReasoning (MedmcQA), MedRBench |
| Law | legalbench, lex-glue |

---

## 2. Data Preparation

### Math / General / Science

These three domains rely on their respective evaluation frameworks, which **automatically download** the required datasets at runtime. No manual preparation is needed.

### Business / Law / Medicine

These three domains share the unified `simple-evaluation` pipeline. Download the JSONL data from HuggingFace:

```
https://huggingface.co/datasets/lhpku20010120/Data-Prep-Bench/tree/main/eval
```

Place the corresponding `.jsonl` files into `simple-evaluation/data/`:

```
simple-evaluation/data/
├── business.jsonl
├── law.jsonl
└── medicine.jsonl
```

---

## 3. Evaluation Process and Configuration

All six domains can be evaluated through the unified script `scripts/run_all_bench.sh` in the project root. Alternatively, each domain can be run individually.

### Batch Evaluation with `run_all_bench.sh`

Edit the `MODELS` array inside `scripts/run_all_bench.sh`:

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

Then run:

```bash
bash Data-Agent-Evaluation/scripts/run_all_bench.sh
```

The script will sequentially run the evaluation for each domain and save results under `Data-Agent-Evaluation/bench_outputs/<domain>/<model_name>/`.

### Individual Domain Evaluation

- **Math / General / Science**: These domains each have their own evaluation frameworks and entry points. Please refer to the documentation in `Qwen2.5-Math/`, `lm-evaluation-harness/`, and `lm-open-science-evaluation/` for standalone usage.

- **Business / Law / Medicine**: Use the unified `simple-evaluation` pipeline:

  ```bash
  cd simple-evaluation
  chmod +x run_evaluation.sh
  ./run_evaluation.sh <model_path> <domain> <output_dir>
  ```

  | Parameter | Description | Example |
  |-----------|-------------|---------|
  | `model_path` | Local model directory (must contain `model*.safetensors`) | `/path/to/Qwen2.5-7B` |
  | `domain` | Evaluation domain: `business`, `medicine`, or `law` | `business` |
  | `output_dir` | Directory to save all results | `./outputs/business_eval` |

  The script will:
  1. Start a vLLM server in the background and wait until `/health` is ready.
  2. Run inference and output `inference_results.jsonl`.
  3. Run domain-specific judging and output `judge_results.jsonl`.
  4. Aggregate scores and output `calculated_scores.json`.

  Optional environment variables:

  | Variable | Default | Description |
  |----------|---------|-------------|
  | `CUDA_VISIBLE_DEVICES` | — | GPU device(s) to use, e.g. `0` or `0,1` |
  | `BASE_PORT` | `8002` | vLLM service port |
  | `TP_SIZE` | `1` | Tensor parallelism size |
  | `GPU_MEMORY_UTIL` | `0.95` | GPU memory utilization limit |
  | `MAX_MODEL_LEN` | `16384` | Maximum sequence length |
  | `DTYPE` | `bfloat16` | Model weight dtype |
  | `HEALTH_TIMEOUT` | `600` | vLLM health-check timeout (seconds) |
  | `MAX_CONCURRENT` | `256` | Inference concurrency |
  | `JUDGE_CONCURRENCY` | `8` | Judge concurrency |
  | `JUDGE_URL` | — | Judge model API URL (for LLM-as-judge) |
  | `JUDGE_API_KEY` | — | Judge model API key |
  | `JUDGE_MODEL` | — | Judge model name |

---

## 4. Evaluation Methodology and Rigor Assurance

For the **business, medicine, and law** domains, the `simple-evaluation` pipeline incorporates the following rigor-assurance mechanisms on top of the benchmark definitions:

1. **Filtering Non-Knowledge Questions**  
   Based on the official task definitions of each benchmark, only questions directly related to knowledge or reasoning capabilities are retained, while subjective or overly open-ended question types are excluded.

2. **Instruction-Following Model Fine-Tuning**  
   Supports lightweight instruction fine-tuning of the model under test to ensure its output format is compatible with the scoring model, reducing evaluation bias caused by format inconsistencies.

3. **LLM-as-Judge Calibration**  
   Employs high-performance general-purpose models (e.g., GPT-4o) as scorers to objectively score model-generated answers, with scoring consistency checks (e.g., manual sampling review) to ensure reliability.

4. **Language Standardization**  
   Non-English questions are either translated to a standard language or excluded to maintain linguistic consistency in the evaluation content, avoiding additional variables introduced by language differences.

The **math, general, and science** domains are evaluated through mature open-source frameworks (`Qwen2.5-Math`, `lm-evaluation-harness`, `lm-open-science-evaluation`) following their original protocols.

---

## 5. Result Recording and Reproducibility

All evaluation results are saved under the unified output root `Data-Agent-Evaluation/bench_outputs/<domain>/<model_name>/`.

For **business / law / medicine**, the output directory contains:
- `inference_results.jsonl` — model-generated answers
- `calculated_scores.json` — aggregated benchmark scores
- `judge_out/judge_results.jsonl` — detailed scoring outputs
- `judge_out/judge.log` — scoring logs
- `vllm_logs/vllm_<model>.log` — vLLM service logs

For **math / general / science**, each framework saves structured results (metrics, samples, and logs) under the same unified path.

By preserving complete configurations and logs, the platform supports subsequent result reproduction and comparative analysis.
