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
   vllm serve Qwen/Qwen3-Embedding-8B --task embed
   ```

2. **Configure the datasets** in `compute_mmd.py`:

   ```python
   DS1_CONFIG = {
       "name": "oda-math",
       "data_path": "OpenDataArena/ODA-Math-460k",
       "data_size": 5000,
       "split": "train",
       "shuffle_seed": 42,
   }
   formatter1 = AlpacaFormatter(
       user_key="question",
       assistant_key="response",
   )

   DS2_CONFIG = {
       "name": "infinity-instruct",
       "data_path": "BAAI/Infinity-Instruct",
       "data_size": 5000,
       "split": "train",
       "shuffle_seed": 42,
   }
   formatter2 = ShareGptFormatter(
       conversations_key="conversations",
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
