# Data-Preparation-Bench

A benchmark for evaluating the data preparation capabilities of large language models (LLMs). The benchmark is organized into two modules:

## Modules

### 1. Data Synthesis & Augmentation

Given raw metadata, the model is tasked with synthesizing or augmenting datasets to improve downstream model training.

### 2. Data Quality Assessment

Given raw metadata, the model is tasked with predicting the training data's impact on downstream task performance.

## Quick Start

### Usage

The package is published on PyPI and can be installed via pip:

```python
pip install distflow
```

For vLLM embedding support, install the optional dependency:

```python
pip install distflow[vllm]
```

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To get started:

```bash
git clone https://github.com/haolpku/Data-Preparation-Bench.git
cd Data-Preparation-Bench
uv sync
```

To use your own datasets, modify the configuration dictionaries and formatters in [compute_mmd.py](./examples/compute_mmd.py):

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

Typically, you only need to update `data_path` with your dataset and define a formatter that converts raw items to the required format. After making these changes, run the MMD computation with:

```bash
uv run examples/compute_mmd.py
```

### Development

To set up the development environment locally:

```bash
uv sync --extra dev
uv run pre-commit install
```

Before committing, format and lint the code:

```bash
uv run pre-commit run --all-files
```

## Experiment Settings

Please refer to [Experiment.md](./Experiment.md) for detailed experiment configurations.
