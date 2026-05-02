# distflow

A Python package for computing distributional distances (e.g., MMD) between datasets, designed for evaluating data preparation quality in LLM training pipelines.

## Installation

The package is published on PyPI and can be installed via pip:

```bash
pip install distflow
```

For vLLM embedding support, install the optional dependency:

```bash
pip install distflow[vllm]
```

### Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To get started:

```bash
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd Data-Preparation-Bench
uv sync
```

To set up the development environment:

```bash
uv sync --extra dev
uv run pre-commit install
```

Before committing, format and lint the code:

```bash
uv run pre-commit run --all-files
```

## Quick Start

### Computing MMD Distance

The example script [compute_mmd.py](./examples/compute_mmd.py) demonstrates how to compute MMD distance between two datasets using the vLLM OpenAI-compatible embedding API.

1. **Start a vLLM embedding server** (e.g., serving `Qwen/Qwen3-Embedding-8B`):

   ```bash
   vllm serve Qwen/Qwen3-Embedding-8B --runner pooling --trust-remote-code --max-model-len 40960 --served-model-name Qwen/Qwen3-Embedding-8B --dtype bfloat16 --seed 42
   ```

2. **Configure the datasets** in `compute_mmd.py`:

   ```python
   from distflow.data.dataset import DistflowDataset
   from distflow.data.data_formatter import AlpacaFormatter, ShareGptFormatter

   dataset_1 = DistflowDataset(
       dataset_name="oda-math",
       data_path="OpenDataArena/ODA-Math-460k",
       load_type="datasets",
       formatter=AlpacaFormatter(
           user_key="question",
           assistant_key="response",
       ),
       data_size=5000,
       split="train",
       shuffle_seed=42,
   )

   dataset_2 = DistflowDataset(
       dataset_name="infinity-instruct",
       data_path="BAAI/Infinity-Instruct",
       load_type="datasets",
       formatter=ShareGptFormatter(
           conversations_key="conversations",
       ),
       data_size=5000,
       split="train",
       shuffle_seed=42,
   )
   ```

   Typically, you only need to update `data_path` with your HuggingFace dataset identifier and define a formatter that converts raw items to the required chat format. Two built-in formatters are available:

   - **`AlpacaFormatter`**: For datasets with separate `user_key` / `assistant_key` fields.
   - **`ShareGptFormatter`**: For datasets with a `conversations` field containing multi-turn messages.

3. **Run the computation**:

   ```bash
   uv run examples/compute_mmd.py
   ```

   To save results to a JSON file:

   ```bash
   uv run examples/compute_mmd.py --output results/
   ```

### Running Quality Benchmark

The example script [run_benchmark.py](./examples/run_benchmark.py) shows how to evaluate a custom data-quality metric by measuring its correlation with downstream task accuracy.

1. **Prepare datasets** (same as above):

   Define one or more `DistflowDataset` objects with the appropriate formatter.

2. **Provide accuracy values manually**:

   The `accuracy/` directory is not part of the repository, so you must supply your own accuracy mapping. The keys must exactly match the `dataset_name` field of each dataset:

   ```python
   accuracys = {
       "dataflow": 0.25,
       "infinity-instruct": 0.30,
       "openr1": 0.45,
   }
   ```

3. **Implement a metric class**:

   Your metric class must implement `score(dataset: DistflowDataset) -> list[MetricsResult]`:

   ```python
   class MyMetric:
       def score(self, dataset: DistflowDataset):
           # Compute metric for the dataset
           return [{"name": "my_metric", "value": 0.8, "meta": {}}]
   ```

4. **Run the benchmark**:

   ```bash
   uv run examples/run_benchmark.py
   ```

   The benchmark computes Pearson / Spearman correlation and a linear fit between your metric and the provided accuracies.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Embedding model name |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `MAX_CONCURRENT_REQUESTS` | `1024` | Max async concurrent embedding requests |
| `TRUNCATE_PROMPT_TOKENS` | `40960` | Token truncation length |
| `SIGMA_CONSTANT_VALUE` | `1.0` | RBF kernel bandwidth |
| `BIAS` | `True` | Use biased MMD estimator |

### Package Dependencies

Core dependencies (see `pyproject.toml`):

- Python ≥ 3.10
- torch, transformers
- openai (for async vLLM API client)
- datasets, sentence-transformers
- scikit-learn, pandas, pydantic

Optional:

- `vllm` (for local vLLM embedding via `VllmEmbed`)

## Experiment Settings

Please refer to [Experiment.md](./Experiment.md) for detailed experiment configurations.
