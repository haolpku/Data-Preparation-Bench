# Experiment Settings

## MMD Computation

For MMD computation, please refer to [compute_mmd.py](./examples/compute_mmd.py). Embeddings are obtained via a vLLM OpenAI-compatible embedding server using the async `OpenAIEmbed` client (`distflow.embed.openai_embed`). Key hyperparameters are configured as follows:

- Embedding model: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (4096-dimensional)
- Truncate length: 40960 tokens (right truncation)
- Kernel type: RBF
- Kernel sigma: 1.0 (constant)
- Estimator: Biased MMD estimator
- Dataset size: 5000 samples with seed 42
- Max concurrent requests: 1024

**Reference Datasets:**
- Math: [ODA-Math-460k](https://huggingface.co/datasets/OpenDataArena/ODA-Math-460k)
- General Text: [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)
- Medical: [ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed)
- Science: [Logics-STEM](https://arxiv.org/abs/2601.01562)
- Finance: [Fin-o1](https://arxiv.org/abs/2502.08127)
- Law: [DISC-Law-SFT](https://arxiv.org/abs/2309.11325)

The experiments were conducted with the following package versions:

- Python: 3.12.12
- vllm: 0.16.0
- torch: 2.9.1+cu128
- transformers: 4.57.6
- openai: 2.31.0
- sentence-transformers: 5.4.0
- uv: 0.10.0

## Training

Training is conducted using [LlamaFactory](https://github.com/hiyouga/LlamaFactory). Base models include:
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

**Training Configuration:**
```yaml
cutoff_len: 32768
packing: false
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

We use DeepSpeed ZeRO-3 for distributed training. Chat templates are set according to model families:
- `qwen` for Qwen2.5-7B
- `llama3` for Llama-3.1-8B
- `mistral` for Mistral-7B-v0.3

**Training Datasets:**

| Domain | Dataset | Samples |
|--------|---------|---------|
| Math | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 20,000 |
| Math | [ODA-Math-460k](https://huggingface.co/datasets/) | 20,000 |
| Math | [ScaleQuest](https://huggingface.co/datasets/dyyyyyyyy/ScaleQuest-Math) | 20,000 |
| Math | [Synthetic-1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1) | 20,000 |
| General | [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) | 20,000 |
| General | [dataflow-instruct-10k](https://huggingface.co/datasets/OpenDCAI/dataflow-instruct-10k) | 10,000 |
| General | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 20,000 |
| General | [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 20,000 |
| General | [WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | 20,000 |
| General | [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 20,000 |
| General | [smoltalk-chinese](https://huggingface.co/datasets/opencsg/smoltalk-chinese) | 20,000 |
| Science | [MegaScience](https://huggingface.co/datasets/) | 20,000 |
| Science | [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | 20,000 |
| Medical | [UltraMedical](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical) | 20,000 |
| Finance | [Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) | 20,000 |
| Law | [Lawyer-Llama](https://huggingface.co/datasets/) | 20,000 |

For data construction experiments, [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is additionally incorporated during fine-tuning.

## Evaluation

### General Text

General text evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with MMLU-Redux as the primary benchmark:

- max_model_len: 32768
- num_fewshot: 5
- apply_chat_template: true

### Math

Math evaluation is performed using [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) with the following generation parameters:

- temperature: 0.6
- max_tokens_per_call: 16384
- top_p: 1
- apply_chat_template: true
- repetition_penalty: 1.1

**Benchmarks:** GSM8K, AMC23, AIME24, Minerva Math, Gaokao2024-Mix, OlympiadBench, and MATH.

### Science

Science evaluation follows the MegaScience evaluation protocol. Benchmarks include MMLU-STEM, MMLU-Pro, GPQA, SuperGPQA, ChemBench, PIQA, and SciBench.

### Medical

Medical evaluation employs MedR-Bench, MedMCQA, and MedCaseReasoning.

### Finance

Finance evaluation uses XFinBench, FinEval-KR, and CPA-QKA.

### Law

Law evaluation uses LegalBench and LexGLUE.

> For benchmarks consisting of multiple subtasks, we report the average score across all subtasks. For tasks evaluated by exact answer matching, we additionally employ an LLM-as-a-judge protocol.

Models are served using vLLM for inference.

## Data Quality Evaluation

We evaluate whether data quality metrics can predict the downstream utility of candidate training datasets before fine-tuning, using Pearson correlation between metric scores and empirical downstream performance.

**DAS (Distributional Alignment Score):** Defined as the negative MMD distance between a candidate dataset and the domain proxy dataset:
$$
\text{DAS}(\mathcal{D}_i) = -\text{MMD}(\mathcal{D}_{\text{proxy}}, \mathcal{D}_i)
$$

**Proxy Datasets:**
- General: Infinity-Instruct
- Math: ODA-Math-460k
- Science: Logics-STEM
- Medical: ReasonMed
- Finance: Fin-o1
- Law: DISC-Law-SFT

**Implementation:**
- Base models: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3
- Embeddings: Qwen3-Embedding-8B (4096-dimensional)
- Sampling: 5,000 instances per dataset
- Kernel: Gaussian RBF with bandwidth $\sigma = 1.0$

**Key Results:** MMD shows strong negative correlations in Math (domain avg -0.86), Medical (-0.77), Science (-0.72), and General Text (-0.68), but weaker correlations in Finance (-0.18) and Law (0.36).

## Results

### General Text

#### Qwen2.5-7B

| Method | abstract_algebra | anatomy | astronomy | business_ethics | clinical_knowledge | college_biology | college_chemistry | college_computer_science | college_mathematics | college_medicine | college_physics | computer_security | conceptual_physics | econometrics | electrical_engineering | elementary_mathematics | formal_logic | global_facts | high_school_biology | high_school_chemistry | high_school_computer_science | high_school_european_history | high_school_geography | high_school_government_and_politics | high_school_macroeconomics | high_school_mathematics | high_school_microeconomics | high_school_physics | high_school_psychology | high_school_statistics | high_school_us_history | high_school_world_history | human_aging | human_sexuality | international_law | jurisprudence | logical_fallacies | machine_learning | management | marketing | medical_genetics | miscellaneous | moral_disputes | moral_scenarios | nutrition | philosophy | prehistory | professional_accounting | professional_law | professional_medicine | professional_psychology | public_relations | security_studies | sociology | us_foreign_policy | virology | world_religions | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 56.2 | 71.7 | 93.6 | 90.6 | 75.8 | 87.8 | 66.7 | 71.1 | 59.6 | 73.6 | 57.0 | 81.4 | 76.1 | 72.2 | 70.4 | 68.0 | 57.5 | 54.5 | 91.6 | 67.7 | 83.0 | 89.0 | 89.0 | 89.0 | 81.4 | 60.0 | 92.9 | 63.9 | 91.7 | 67.3 | 90.0 | 93.9 | 82.8 | 86.4 | 78.9 | 81.8 | 94.6 | 60.7 | 91.9 | 96.9 | 83.0 | 88.9 | 86.5 | 43.3 | 81.4 | 83.1 | 83.0 | 62.0 | 57.3 | 79.8 | 78.1 | 82.4 | 72.3 | 91.8 | 88.9 | 90.7 | 86.9 | 78.0 |
| finance-instruct | 62.9 | 67.7 | 91.5 | 82.4 | 72.7 | 88.8 | 62.7 | 72.2 | 58.6 | 74.7 | 52.0 | 79.4 | 76.1 | 72.2 | 73.5 | 69.1 | 55.2 | 61.4 | 89.5 | 63.6 | 79.0 | 86.8 | 90.0 | 91.0 | 81.4 | 58.0 | 91.8 | 61.9 | 91.7 | 69.4 | 85.0 | 94.9 | 77.0 | 84.0 | 84.2 | 85.9 | 94.6 | 68.5 | 90.9 | 95.8 | 83.0 | 87.8 | 82.3 | 48.5 | 81.4 | 82.0 | 83.0 | 62.0 | 50.0 | 79.8 | 80.2 | 82.4 | 75.5 | 89.8 | 87.9 | 93.0 | 86.9 | 77.6 |
| lawyer-llama | 52.8 | 71.7 | 88.3 | 89.4 | 74.7 | 84.7 | 61.3 | 69.1 | 55.6 | 71.3 | 54.0 | 81.4 | 76.1 | 70.1 | 69.4 | 69.1 | 56.3 | 63.6 | 89.5 | 62.6 | 82.0 | 87.9 | 88.0 | 91.0 | 82.5 | 56.0 | 91.8 | 66.0 | 90.6 | 73.5 | 87.0 | 93.9 | 82.8 | 85.2 | 80.0 | 81.8 | 93.2 | 65.2 | 91.9 | 95.8 | 84.0 | 87.8 | 85.4 | 50.5 | 82.5 | 80.9 | 85.0 | 57.6 | 56.1 | 78.8 | 82.3 | 79.1 | 75.5 | 90.8 | 86.9 | 90.7 | 85.9 | 77.5 |
| megascience | 59.6 | 73.7 | 89.4 | 90.6 | 76.8 | 85.7 | 61.3 | 71.1 | 54.5 | 72.4 | 59.0 | 81.4 | 76.1 | 68.0 | 68.4 | 71.1 | 64.4 | 56.8 | 90.5 | 68.7 | 81.0 | 87.9 | 87.0 | 89.0 | 83.5 | 59.0 | 91.8 | 63.9 | 89.6 | 70.4 | 85.0 | 91.9 | 81.6 | 88.9 | 80.0 | 80.8 | 93.2 | 65.2 | 87.9 | 94.8 | 79.0 | 86.7 | 81.2 | 42.3 | 81.4 | 79.8 | 85.0 | 64.1 | 58.5 | 80.8 | 81.2 | 84.6 | 73.4 | 88.8 | 88.9 | 88.4 | 87.9 | 77.6 |
| nemotron-science | 58.4 | 72.7 | 91.5 | 91.8 | 76.8 | 86.7 | 66.7 | 74.2 | 55.6 | 70.1 | 57.0 | 82.5 | 70.7 | 68.0 | 67.3 | 64.9 | 60.9 | 58.0 | 90.5 | 71.7 | 89.0 | 87.9 | 89.0 | 89.0 | 82.5 | 58.0 | 92.9 | 61.9 | 89.6 | 70.4 | 86.0 | 93.9 | 82.8 | 86.4 | 82.1 | 80.8 | 91.9 | 69.7 | 89.9 | 96.9 | 84.0 | 90.0 | 83.3 | 54.6 | 80.4 | 79.8 | 86.0 | 59.8 | 56.1 | 84.8 | 83.3 | 78.0 | 71.3 | 87.8 | 87.9 | 93.0 | 87.9 | 78.2 |
| openhermes | 56.2 | 69.7 | 87.2 | 88.2 | 77.8 | 86.7 | 62.7 | 70.1 | 62.6 | 71.3 | 54.0 | 79.4 | 78.3 | 68.0 | 70.4 | 73.2 | 57.5 | 59.1 | 92.6 | 64.6 | 82.0 | 86.8 | 90.0 | 92.0 | 82.5 | 56.0 | 94.9 | 63.9 | 91.7 | 69.4 | 83.0 | 93.9 | 82.8 | 87.7 | 82.1 | 83.8 | 91.9 | 64.0 | 91.9 | 96.9 | 84.0 | 86.7 | 85.4 | 41.2 | 84.5 | 80.9 | 83.0 | 62.0 | 57.3 | 77.8 | 81.2 | 80.2 | 75.5 | 87.8 | 87.9 | 90.7 | 88.9 | 77.8 |
| openr1 | 58.4 | 69.7 | 89.4 | 87.1 | 73.7 | 84.7 | 61.3 | 63.9 | 51.5 | 74.7 | 57.0 | 83.5 | 77.2 | 71.1 | 71.4 | 67.0 | 57.5 | 53.4 | 92.6 | 68.7 | 83.0 | 89.0 | 85.0 | 90.0 | 80.4 | 60.0 | 91.8 | 60.8 | 90.6 | 73.5 | 87.0 | 94.9 | 81.6 | 84.0 | 76.8 | 80.8 | 90.5 | 59.6 | 86.9 | 96.9 | 79.0 | 87.8 | 84.4 | 49.5 | 81.4 | 80.9 | 82.0 | 59.8 | 53.7 | 78.8 | 80.2 | 79.1 | 74.5 | 86.7 | 88.9 | 88.4 | 88.9 | 76.9 |
| scale | 57.3 | 70.7 | 86.2 | 82.4 | 71.7 | 83.7 | 60.0 | 68.0 | 56.6 | 72.4 | 56.0 | 82.5 | 76.1 | 69.1 | 69.4 | 63.9 | 50.6 | 65.9 | 88.4 | 58.6 | 80.0 | 90.1 | 88.0 | 89.0 | 80.4 | 51.0 | 93.9 | 62.9 | 90.6 | 67.3 | 84.0 | 91.9 | 80.5 | 85.2 | 78.9 | 79.8 | 93.2 | 66.3 | 88.9 | 97.9 | 84.0 | 87.8 | 84.4 | 42.3 | 83.5 | 79.8 | 83.0 | 57.6 | 56.1 | 76.8 | 77.1 | 79.1 | 74.5 | 88.8 | 88.9 | 90.7 | 87.9 | 76.3 |
| smoltalk | 60.7 | 69.7 | 90.4 | 89.4 | 74.7 | 87.8 | 62.7 | 70.1 | 60.6 | 73.6 | 58.0 | 82.5 | 75.0 | 69.1 | 71.4 | 69.1 | 57.5 | 61.4 | 91.6 | 70.7 | 85.0 | 87.9 | 88.0 | 91.0 | 81.4 | 55.0 | 92.9 | 58.8 | 90.6 | 69.4 | 84.0 | 93.9 | 82.8 | 85.2 | 80.0 | 82.8 | 91.9 | 68.5 | 91.9 | 94.8 | 83.0 | 87.8 | 88.5 | 49.5 | 83.5 | 80.9 | 83.0 | 62.0 | 57.3 | 77.8 | 81.2 | 76.9 | 76.6 | 91.8 | 86.9 | 90.7 | 87.9 | 78.0 |
| synthetic_1 | 58.4 | 70.7 | 92.6 | 85.9 | 72.7 | 83.7 | 64.0 | 64.9 | 56.6 | 73.6 | 55.0 | 82.5 | 80.4 | 72.2 | 68.4 | 60.8 | 46.0 | 59.1 | 91.6 | 65.7 | 79.0 | 89.0 | 87.0 | 90.0 | 78.4 | 59.0 | 88.8 | 57.7 | 88.5 | 71.4 | 87.0 | 90.9 | 77.0 | 84.0 | 81.1 | 80.8 | 89.2 | 60.7 | 89.9 | 95.8 | 79.0 | 87.8 | 81.2 | 45.4 | 81.4 | 79.8 | 79.0 | 59.8 | 53.7 | 77.8 | 82.3 | 78.0 | 73.4 | 86.7 | 88.9 | 93.0 | 90.9 | 76.3 |
| tulu | 57.3 | 71.7 | 90.4 | 87.1 | 72.7 | 85.7 | 62.7 | 70.1 | 61.6 | 74.7 | 55.0 | 82.5 | 76.1 | 66.0 | 71.4 | 69.1 | 59.8 | 63.6 | 93.7 | 71.7 | 80.0 | 87.9 | 88.0 | 89.0 | 82.5 | 57.0 | 93.9 | 60.8 | 90.6 | 67.3 | 86.0 | 92.9 | 83.9 | 88.9 | 83.2 | 81.8 | 90.5 | 64.0 | 89.9 | 93.8 | 82.0 | 87.8 | 85.4 | 54.6 | 83.5 | 80.9 | 80.0 | 62.0 | 56.1 | 80.8 | 80.2 | 80.2 | 75.5 | 89.8 | 89.9 | 88.4 | 87.9 | 77.9 |
| ultrachat | 56.2 | 72.7 | 90.4 | 89.4 | 76.8 | 86.7 | 66.7 | 72.2 | 53.5 | 71.3 | 56.0 | 78.4 | 76.1 | 66.0 | 70.4 | 67.0 | 59.8 | 53.4 | 91.6 | 67.7 | 78.0 | 87.9 | 89.0 | 89.0 | 82.5 | 59.0 | 91.8 | 60.8 | 90.6 | 69.4 | 85.0 | 94.9 | 79.3 | 86.4 | 82.1 | 84.8 | 94.6 | 64.0 | 88.9 | 96.9 | 83.0 | 87.8 | 86.5 | 40.2 | 83.5 | 80.9 | 85.0 | 58.7 | 51.2 | 79.8 | 81.2 | 80.2 | 77.7 | 89.8 | 87.9 | 93.0 | 87.9 | 77.4 |
| ultramedical | 59.6 | 72.7 | 91.5 | 92.9 | 75.8 | 84.7 | 65.3 | 71.1 | 52.5 | 74.7 | 51.0 | 82.5 | 78.3 | 68.0 | 68.4 | 68.0 | 56.3 | 55.7 | 87.4 | 67.7 | 84.0 | 90.1 | 85.0 | 90.0 | 81.4 | 58.0 | 92.9 | 63.9 | 92.7 | 68.4 | 90.0 | 92.9 | 79.3 | 85.2 | 83.2 | 81.8 | 89.2 | 64.0 | 86.9 | 96.9 | 84.0 | 87.8 | 83.3 | 45.4 | 80.4 | 79.8 | 83.0 | 56.5 | 56.1 | 76.8 | 82.3 | 74.7 | 72.3 | 91.8 | 89.9 | 95.3 | 86.9 | 77.3 |
| wizardlm | 50.6 | 70.7 | 89.4 | 88.2 | 74.7 | 89.8 | 66.7 | 71.1 | 64.6 | 74.7 | 56.0 | 82.5 | 78.3 | 72.2 | 72.4 | 70.1 | 58.6 | 60.2 | 91.6 | 68.7 | 81.0 | 87.9 | 88.0 | 90.0 | 83.5 | 54.0 | 92.9 | 62.9 | 92.7 | 70.4 | 86.0 | 91.9 | 81.6 | 85.2 | 84.2 | 82.8 | 91.9 | 64.0 | 88.9 | 94.8 | 83.0 | 87.8 | 83.3 | 42.3 | 82.5 | 79.8 | 82.0 | 60.9 | 54.9 | 79.8 | 81.2 | 84.6 | 77.7 | 92.9 | 85.9 | 90.7 | 87.9 | 77.9 |

#### Llama-3.1-8B

| Method | abstract_algebra | anatomy | astronomy | business_ethics | clinical_knowledge | college_biology | college_chemistry | college_computer_science | college_mathematics | college_medicine | college_physics | computer_security | conceptual_physics | econometrics | electrical_engineering | elementary_mathematics | formal_logic | global_facts | high_school_biology | high_school_chemistry | high_school_computer_science | high_school_european_history | high_school_geography | high_school_government_and_politics | high_school_macroeconomics | high_school_mathematics | high_school_microeconomics | high_school_physics | high_school_psychology | high_school_statistics | high_school_us_history | high_school_world_history | human_aging | human_sexuality | international_law | jurisprudence | logical_fallacies | machine_learning | management | marketing | medical_genetics | miscellaneous | moral_disputes | moral_scenarios | nutrition | philosophy | prehistory | professional_accounting | professional_law | professional_medicine | professional_psychology | public_relations | security_studies | sociology | us_foreign_policy | virology | world_religions | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 39.3 | 60.6 | 67.0 | 64.7 | 63.6 | 67.3 | 48.0 | 50.5 | 27.3 | 69.0 | 37.0 | 77.3 | 52.2 | 46.4 | 64.3 | 42.3 | 41.4 | 39.8 | 82.1 | 45.5 | 66.0 | 74.7 | 70.0 | 80.0 | 56.7 | 42.0 | 68.4 | 38.1 | 80.2 | 46.9 | 81.0 | 87.9 | 65.5 | 80.2 | 77.9 | 66.7 | 87.8 | 47.2 | 75.8 | 90.6 | 75.0 | 83.3 | 70.8 | 24.7 | 73.2 | 70.8 | 70.0 | 50.0 | 40.2 | 65.7 | 65.6 | 74.7 | 64.9 | 78.6 | 80.8 | 79.1 | 84.8 | 63.5 |
| finance-instruct | 41.6 | 57.6 | 68.1 | 69.4 | 72.7 | 66.3 | 45.3 | 46.4 | 33.3 | 70.1 | 37.0 | 79.4 | 51.1 | 51.5 | 56.1 | 41.2 | 41.4 | 44.3 | 77.9 | 48.5 | 59.0 | 80.2 | 76.0 | 84.0 | 54.6 | 37.0 | 74.5 | 38.1 | 75.0 | 46.9 | 85.0 | 87.9 | 65.5 | 80.2 | 80.0 | 74.7 | 87.8 | 51.7 | 79.8 | 92.7 | 77.0 | 84.4 | 74.0 | 30.9 | 71.1 | 71.9 | 68.0 | 47.8 | 45.1 | 68.7 | 71.9 | 73.6 | 69.1 | 87.8 | 82.8 | 90.7 | 82.8 | 65.1 |
| lawyer-llama | 37.1 | 62.6 | 67.0 | 63.5 | 68.7 | 69.4 | 46.7 | 46.4 | 37.4 | 67.8 | 39.0 | 78.4 | 58.7 | 50.5 | 56.1 | 45.4 | 40.2 | 45.5 | 73.7 | 45.5 | 65.0 | 75.8 | 67.0 | 76.0 | 48.5 | 33.0 | 72.4 | 33.0 | 74.0 | 38.8 | 86.0 | 85.9 | 64.4 | 76.5 | 78.9 | 66.7 | 79.7 | 53.9 | 71.7 | 92.7 | 75.0 | 84.4 | 74.0 | 30.9 | 77.3 | 65.2 | 62.0 | 44.6 | 45.1 | 68.7 | 72.9 | 75.8 | 67.0 | 86.7 | 84.8 | 93.0 | 83.8 | 63.7 |
| megascience | 36.0 | 65.7 | 73.4 | 64.7 | 68.7 | 69.4 | 46.7 | 49.5 | 31.3 | 72.4 | 34.0 | 79.4 | 45.7 | 49.5 | 58.2 | 42.3 | 37.9 | 45.5 | 78.9 | 44.4 | 67.0 | 81.3 | 76.0 | 80.0 | 52.6 | 34.0 | 75.5 | 41.2 | 75.0 | 42.9 | 86.0 | 86.9 | 72.4 | 80.2 | 76.8 | 68.7 | 86.5 | 50.6 | 79.8 | 92.7 | 84.0 | 87.8 | 68.8 | 32.0 | 74.2 | 70.8 | 65.0 | 42.4 | 46.3 | 66.7 | 71.9 | 67.0 | 59.6 | 85.7 | 88.9 | 83.7 | 88.9 | 64.6 |
| nemotron-science | 36.0 | 67.7 | 67.0 | 70.6 | 74.7 | 72.4 | 57.3 | 56.7 | 41.4 | 69.0 | 46.0 | 79.4 | 53.3 | 52.6 | 64.3 | 46.4 | 46.0 | 44.3 | 83.2 | 49.5 | 66.0 | 79.1 | 85.0 | 86.0 | 54.6 | 39.0 | 78.6 | 47.4 | 84.4 | 56.1 | 86.0 | 91.9 | 72.4 | 85.2 | 77.9 | 74.7 | 90.5 | 51.7 | 75.8 | 92.7 | 76.0 | 87.8 | 70.8 | 28.9 | 73.2 | 69.7 | 62.0 | 50.0 | 48.8 | 69.7 | 71.9 | 75.8 | 67.0 | 88.8 | 85.9 | 86.0 | 84.8 | 67.5 |
| openhermes | 43.8 | 66.7 | 66.0 | 69.4 | 70.7 | 73.5 | 52.0 | 52.6 | 38.4 | 69.0 | 49.0 | 81.4 | 54.3 | 54.6 | 65.3 | 41.2 | 48.3 | 39.8 | 77.9 | 47.5 | 67.0 | 83.5 | 82.0 | 85.0 | 62.9 | 41.0 | 76.5 | 47.4 | 82.3 | 49.0 | 86.0 | 90.9 | 74.7 | 80.2 | 77.9 | 74.7 | 91.9 | 53.9 | 80.8 | 94.8 | 84.0 | 86.7 | 70.8 | 46.4 | 69.1 | 75.3 | 63.0 | 52.2 | 51.2 | 68.7 | 76.0 | 73.6 | 74.5 | 86.7 | 87.9 | 83.7 | 86.9 | 68.1 |
| openr1 | 29.2 | 35.4 | 30.9 | 29.4 | 28.3 | 22.4 | 17.3 | 25.8 | 22.2 | 28.7 | 20.0 | 34.0 | 27.2 | 23.7 | 30.6 | 25.8 | 23.0 | 25.0 | 24.2 | 31.3 | 32.0 | 33.0 | 26.0 | 27.0 | 18.6 | 28.0 | 26.5 | 21.6 | 25.0 | 23.5 | 40.0 | 39.4 | 25.3 | 23.5 | 38.9 | 28.3 | 31.1 | 32.6 | 28.3 | 39.6 | 28.0 | 30.0 | 31.2 | 26.8 | 29.9 | 29.2 | 33.0 | 29.3 | 25.6 | 18.2 | 34.4 | 34.1 | 23.4 | 21.4 | 39.4 | 34.9 | 35.4 | 28.5 |
| scale | 43.8 | 62.6 | 63.8 | 69.4 | 69.7 | 71.4 | 50.7 | 47.4 | 32.3 | 65.5 | 43.0 | 79.4 | 51.1 | 52.6 | 62.2 | 40.2 | 44.8 | 37.5 | 78.9 | 41.4 | 59.0 | 75.8 | 72.0 | 82.0 | 51.5 | 38.0 | 71.4 | 38.1 | 79.2 | 43.9 | 84.0 | 86.9 | 71.3 | 79.0 | 77.9 | 66.7 | 83.8 | 52.8 | 77.8 | 89.6 | 74.0 | 75.6 | 71.9 | 32.0 | 73.2 | 73.0 | 67.0 | 47.8 | 41.5 | 69.7 | 67.7 | 70.3 | 67.0 | 80.6 | 83.8 | 93.0 | 82.8 | 64.2 |
| smoltalk | 39.3 | 73.7 | 76.6 | 75.3 | 74.7 | 74.5 | 52.0 | 60.8 | 34.3 | 73.6 | 44.0 | 78.4 | 53.3 | 59.8 | 63.3 | 41.2 | 51.7 | 40.9 | 82.1 | 59.6 | 67.0 | 82.4 | 82.0 | 86.0 | 69.1 | 46.0 | 73.5 | 51.5 | 83.3 | 55.1 | 88.0 | 87.9 | 72.4 | 84.0 | 80.0 | 69.7 | 90.5 | 51.7 | 82.8 | 95.8 | 81.0 | 88.9 | 74.0 | 42.3 | 77.3 | 77.5 | 64.0 | 53.3 | 51.2 | 67.7 | 70.8 | 78.0 | 70.2 | 86.7 | 89.9 | 88.4 | 87.9 | 69.4 |
| synthetic_1 | 27.0 | 33.3 | 34.0 | 25.9 | 33.3 | 22.4 | 17.3 | 30.9 | 21.2 | 29.9 | 21.0 | 43.3 | 27.2 | 28.9 | 30.6 | 26.8 | 26.4 | 31.8 | 27.4 | 28.3 | 41.0 | 51.6 | 32.0 | 33.0 | 22.7 | 26.0 | 21.4 | 21.6 | 24.0 | 25.5 | 53.0 | 51.5 | 25.3 | 38.3 | 37.9 | 25.3 | 32.4 | 31.5 | 30.3 | 38.5 | 22.0 | 37.8 | 32.3 | 18.6 | 29.9 | 33.7 | 33.0 | 28.3 | 31.7 | 22.2 | 37.5 | 35.2 | 23.4 | 27.6 | 41.4 | 32.6 | 32.3 | 30.7 |
| tulu | 36.0 | 63.6 | 67.0 | 76.5 | 68.7 | 68.4 | 50.7 | 48.5 | 37.4 | 64.4 | 50.0 | 80.4 | 53.3 | 47.4 | 61.2 | 41.2 | 47.1 | 40.9 | 76.8 | 51.5 | 63.0 | 79.1 | 79.0 | 84.0 | 60.8 | 42.0 | 78.6 | 46.4 | 85.4 | 48.0 | 84.0 | 86.9 | 73.6 | 75.3 | 78.9 | 72.7 | 91.9 | 58.4 | 80.8 | 92.7 | 78.0 | 87.8 | 77.1 | 39.2 | 74.2 | 77.5 | 63.0 | 48.9 | 52.4 | 65.7 | 74.0 | 70.3 | 69.1 | 84.7 | 84.8 | 86.0 | 84.8 | 66.8 |
| ultrachat | 40.4 | 63.6 | 72.3 | 72.9 | 73.7 | 75.5 | 45.3 | 55.7 | 35.4 | 71.3 | 47.0 | 77.3 | 54.3 | 54.6 | 57.1 | 40.2 | 49.4 | 42.0 | 81.1 | 51.5 | 63.0 | 78.0 | 80.0 | 84.0 | 57.7 | 44.0 | 79.6 | 49.5 | 85.4 | 45.9 | 86.0 | 89.9 | 69.0 | 81.5 | 80.0 | 72.7 | 90.5 | 50.6 | 72.7 | 95.8 | 82.0 | 87.8 | 80.2 | 37.1 | 76.3 | 76.4 | 64.0 | 46.7 | 50.0 | 65.7 | 71.9 | 74.7 | 71.3 | 88.8 | 86.9 | 86.0 | 87.9 | 67.6 |
| ultramedical | 36.0 | 47.5 | 54.3 | 45.9 | 43.4 | 42.9 | 36.0 | 40.2 | 20.2 | 51.7 | 26.0 | 63.9 | 34.8 | 37.1 | 43.9 | 28.9 | 28.7 | 33.0 | 55.8 | 27.3 | 49.0 | 61.5 | 55.0 | 63.0 | 36.1 | 26.0 | 49.0 | 20.6 | 53.1 | 26.5 | 72.0 | 77.8 | 55.2 | 59.3 | 48.4 | 42.4 | 59.5 | 37.1 | 54.5 | 60.4 | 54.0 | 74.4 | 40.6 | 20.6 | 50.5 | 41.6 | 50.0 | 34.8 | 35.4 | 34.3 | 44.8 | 58.2 | 41.5 | 44.9 | 68.7 | 62.8 | 66.7 | 46.1 |
| wizardlm | 40.4 | 65.7 | 72.3 | 70.6 | 72.7 | 69.4 | 54.7 | 51.5 | 38.4 | 66.7 | 47.0 | 77.3 | 56.5 | 56.7 | 63.3 | 43.3 | 48.3 | 39.8 | 82.1 | 52.5 | 64.0 | 78.0 | 80.0 | 81.0 | 58.8 | 37.0 | 73.5 | 50.5 | 82.3 | 43.9 | 88.0 | 88.9 | 73.6 | 77.8 | 76.8 | 74.7 | 91.9 | 52.8 | 75.8 | 92.7 | 76.0 | 86.7 | 74.0 | 37.1 | 74.2 | 77.5 | 63.0 | 54.3 | 47.6 | 65.7 | 72.9 | 72.5 | 70.2 | 88.8 | 86.9 | 90.7 | 86.9 | 67.3 |

#### Mistral-7B-v0.3

| Method | abstract_algebra | anatomy | astronomy | business_ethics | clinical_knowledge | college_biology | college_chemistry | college_computer_science | college_mathematics | college_medicine | college_physics | computer_security | conceptual_physics | econometrics | electrical_engineering | elementary_mathematics | formal_logic | global_facts | high_school_biology | high_school_chemistry | high_school_computer_science | high_school_european_history | high_school_geography | high_school_government_and_politics | high_school_macroeconomics | high_school_mathematics | high_school_microeconomics | high_school_physics | high_school_psychology | high_school_statistics | high_school_us_history | high_school_world_history | human_aging | human_sexuality | international_law | jurisprudence | logical_fallacies | machine_learning | management | marketing | medical_genetics | miscellaneous | moral_disputes | moral_scenarios | nutrition | philosophy | prehistory | professional_accounting | professional_law | professional_medicine | professional_psychology | public_relations | security_studies | sociology | us_foreign_policy | virology | world_religions | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 27.0 | 54.5 | 58.5 | 55.3 | 59.6 | 66.3 | 48.0 | 36.1 | 37.4 | 63.2 | 45.0 | 77.3 | 45.7 | 43.3 | 52.0 | 39.2 | 23.0 | 31.8 | 70.5 | 44.4 | 64.0 | 69.2 | 72.0 | 67.0 | 53.6 | 31.0 | 54.1 | 36.1 | 71.9 | 35.7 | 79.0 | 84.8 | 65.5 | 69.1 | 73.7 | 66.7 | 86.5 | 42.7 | 72.7 | 79.2 | 64.0 | 83.3 | 63.5 | 19.6 | 54.6 | 66.3 | 67.0 | 42.4 | 50.0 | 54.5 | 61.5 | 71.4 | 58.5 | 71.4 | 79.8 | 76.7 | 77.8 | 58.2 |
| finance-instruct | 31.5 | 53.5 | 60.6 | 56.5 | 59.6 | 58.2 | 41.3 | 39.2 | 34.3 | 55.2 | 34.0 | 69.1 | 45.7 | 41.2 | 53.1 | 38.1 | 28.7 | 35.2 | 65.3 | 40.4 | 48.0 | 72.5 | 60.0 | 71.0 | 50.5 | 25.0 | 48.0 | 22.7 | 68.8 | 30.6 | 75.0 | 77.8 | 67.8 | 65.4 | 78.9 | 64.6 | 81.1 | 43.8 | 71.7 | 74.0 | 54.0 | 83.3 | 66.7 | 29.9 | 58.8 | 67.4 | 58.0 | 42.4 | 34.1 | 41.4 | 56.2 | 72.5 | 63.8 | 75.5 | 78.8 | 79.1 | 78.8 | 55.8 |
| lawyer-llama | 25.8 | 58.6 | 59.6 | 38.8 | 53.5 | 60.2 | 50.7 | 42.3 | 22.2 | 60.9 | 27.0 | 59.8 | 30.4 | 39.2 | 38.8 | 10.3 | 34.5 | 23.9 | 54.7 | 41.4 | 36.0 | 68.1 | 45.0 | 66.0 | 53.6 | 17.0 | 42.9 | 29.9 | 69.8 | 37.8 | 72.0 | 74.7 | 46.0 | 64.2 | 65.3 | 66.7 | 75.7 | 32.6 | 56.6 | 67.7 | 45.0 | 73.3 | 37.5 | 29.9 | 56.7 | 46.1 | 51.0 | 42.4 | 36.6 | 40.4 | 39.6 | 59.3 | 46.8 | 61.2 | 71.7 | 69.8 | 75.8 | 49.2 |
| megascience | 34.8 | 55.6 | 58.5 | 58.8 | 64.6 | 61.2 | 48.0 | 49.5 | 34.3 | 57.5 | 42.0 | 75.3 | 46.7 | 39.2 | 52.0 | 43.3 | 27.6 | 44.3 | 70.5 | 39.4 | 60.0 | 78.0 | 64.0 | 70.0 | 50.5 | 22.0 | 61.2 | 38.1 | 72.9 | 33.7 | 76.0 | 78.8 | 65.5 | 75.3 | 77.9 | 69.7 | 79.7 | 46.1 | 74.7 | 82.3 | 73.0 | 82.2 | 65.6 | 22.7 | 68.0 | 68.5 | 63.0 | 48.9 | 45.1 | 53.5 | 58.3 | 74.7 | 66.0 | 80.6 | 76.8 | 79.1 | 78.8 | 59.4 |
| nemotron-science | 36.0 | 58.6 | 61.7 | 56.5 | 62.6 | 63.3 | 60.0 | 44.3 | 38.4 | 60.9 | 29.0 | 74.2 | 47.8 | 47.4 | 49.0 | 43.3 | 31.0 | 37.5 | 65.3 | 47.5 | 55.0 | 76.9 | 79.0 | 76.0 | 60.8 | 31.0 | 58.2 | 34.0 | 77.1 | 46.9 | 81.0 | 78.8 | 65.5 | 74.1 | 73.7 | 73.7 | 74.3 | 44.9 | 74.7 | 85.4 | 68.0 | 82.2 | 62.5 | 27.8 | 61.9 | 58.4 | 64.0 | 43.5 | 43.9 | 57.6 | 65.6 | 65.9 | 62.8 | 81.6 | 81.8 | 81.4 | 79.8 | 60.1 |
| openhermes | 36.0 | 55.6 | 56.4 | 63.5 | 63.6 | 60.2 | 46.7 | 43.3 | 37.4 | 55.2 | 45.0 | 68.0 | 42.4 | 38.1 | 59.2 | 39.2 | 34.5 | 35.2 | 63.2 | 40.4 | 59.0 | 80.2 | 71.0 | 73.0 | 55.7 | 32.0 | 61.2 | 36.1 | 70.8 | 36.7 | 82.0 | 80.8 | 70.1 | 71.6 | 71.6 | 68.7 | 83.8 | 46.1 | 67.7 | 87.5 | 61.0 | 81.1 | 61.5 | 30.9 | 59.8 | 65.2 | 68.0 | 46.7 | 48.8 | 53.5 | 54.2 | 62.6 | 66.0 | 75.5 | 78.8 | 69.8 | 76.8 | 58.7 |
| openr1 | 22.5 | 40.4 | 38.3 | 32.9 | 31.3 | 38.8 | 30.7 | 35.1 | 27.3 | 29.9 | 26.0 | 40.2 | 28.3 | 29.9 | 30.6 | 29.9 | 20.7 | 34.1 | 38.9 | 31.3 | 34.0 | 70.3 | 43.0 | 40.0 | 19.6 | 23.0 | 30.6 | 26.8 | 42.7 | 20.4 | 58.0 | 64.6 | 31.0 | 43.2 | 50.5 | 33.3 | 33.8 | 39.3 | 39.4 | 43.8 | 34.0 | 54.4 | 36.5 | 22.7 | 42.3 | 40.4 | 35.0 | 32.6 | 28.0 | 26.3 | 38.5 | 47.3 | 44.7 | 32.7 | 41.4 | 34.9 | 48.5 | 36.2 |
| scale | 30.3 | 57.6 | 57.4 | 47.1 | 56.6 | 59.2 | 40.0 | 56.7 | 36.4 | 52.9 | 29.0 | 66.0 | 48.9 | 35.1 | 55.1 | 43.3 | 31.0 | 46.6 | 58.9 | 32.3 | 53.0 | 69.2 | 64.0 | 64.0 | 46.4 | 33.0 | 49.0 | 33.0 | 71.9 | 23.5 | 68.0 | 77.8 | 65.5 | 74.1 | 72.6 | 61.6 | 67.6 | 44.9 | 65.7 | 72.9 | 61.0 | 77.8 | 61.5 | 24.7 | 55.7 | 64.0 | 53.0 | 44.6 | 39.0 | 48.5 | 62.5 | 68.1 | 53.2 | 76.5 | 81.8 | 76.7 | 74.7 | 55.1 |
| smoltalk | 33.7 | 55.6 | 57.4 | 60.0 | 55.6 | 62.2 | 41.3 | 37.1 | 31.3 | 58.6 | 40.0 | 68.0 | 48.9 | 45.4 | 54.1 | 40.2 | 34.5 | 43.2 | 68.4 | 30.3 | 55.0 | 74.7 | 62.0 | 67.0 | 52.6 | 34.0 | 54.1 | 39.2 | 75.0 | 32.7 | 77.0 | 83.8 | 65.5 | 65.4 | 71.6 | 69.7 | 79.7 | 42.7 | 68.7 | 78.1 | 70.0 | 77.8 | 62.5 | 36.1 | 62.9 | 62.9 | 70.0 | 50.0 | 43.9 | 54.5 | 61.5 | 67.0 | 64.9 | 78.6 | 75.8 | 79.1 | 78.8 | 58.1 |
| synthetic_1 | 32.6 | 52.5 | 55.3 | 44.7 | 43.4 | 49.0 | 32.0 | 30.9 | 33.3 | 46.0 | 31.0 | 58.8 | 40.2 | 40.2 | 46.9 | 38.1 | 20.7 | 36.4 | 47.4 | 38.4 | 50.0 | 73.6 | 51.0 | 56.0 | 43.3 | 36.0 | 43.9 | 32.0 | 56.2 | 25.5 | 68.0 | 82.8 | 49.4 | 59.3 | 68.4 | 53.5 | 63.5 | 36.0 | 59.6 | 62.5 | 49.0 | 76.7 | 59.4 | 26.8 | 49.5 | 62.9 | 53.0 | 44.6 | 30.5 | 37.4 | 50.0 | 61.5 | 55.3 | 53.1 | 62.6 | 72.1 | 72.7 | 49.2 |
| tulu | 38.2 | 56.6 | 59.6 | 63.5 | 62.6 | 64.3 | 56.0 | 49.5 | 38.4 | 70.1 | 45.0 | 74.2 | 50.0 | 40.2 | 58.2 | 40.2 | 35.6 | 43.2 | 68.4 | 45.5 | 56.0 | 75.8 | 75.0 | 74.0 | 58.8 | 32.0 | 61.2 | 40.2 | 75.0 | 38.8 | 79.0 | 78.8 | 64.4 | 74.1 | 73.7 | 73.7 | 82.4 | 46.1 | 75.8 | 83.3 | 63.0 | 81.1 | 69.8 | 43.3 | 62.9 | 69.7 | 69.0 | 47.8 | 39.0 | 58.6 | 66.7 | 72.5 | 69.1 | 76.5 | 78.8 | 83.7 | 77.8 | 61.5 |
| ultrachat | 36.0 | 58.6 | 59.6 | 70.6 | 64.6 | 72.4 | 56.0 | 44.3 | 38.4 | 63.2 | 49.0 | 71.1 | 44.6 | 43.3 | 54.1 | 38.1 | 37.9 | 42.0 | 72.6 | 42.4 | 59.0 | 75.8 | 77.0 | 76.0 | 56.7 | 38.0 | 61.2 | 32.0 | 78.1 | 46.9 | 76.0 | 80.8 | 70.1 | 76.5 | 75.8 | 76.8 | 90.5 | 48.3 | 74.7 | 88.5 | 70.0 | 83.3 | 68.8 | 39.2 | 61.9 | 74.2 | 70.0 | 51.1 | 52.4 | 59.6 | 74.0 | 70.3 | 71.3 | 85.7 | 83.8 | 74.4 | 79.8 | 62.9 |
| ultramedical | 28.1 | 32.3 | 47.9 | 31.8 | 40.4 | 43.9 | 34.7 | 37.1 | 29.3 | 37.9 | 29.0 | 34.0 | 25.0 | 36.1 | 32.7 | 21.6 | 28.7 | 28.4 | 44.2 | 39.4 | 39.0 | 68.1 | 38.0 | 54.0 | 26.8 | 23.0 | 27.6 | 20.6 | 44.8 | 33.7 | 69.0 | 60.6 | 42.5 | 45.7 | 51.6 | 34.3 | 29.7 | 25.8 | 41.4 | 39.6 | 27.0 | 47.8 | 41.7 | 26.8 | 45.4 | 31.5 | 39.0 | 26.1 | 28.0 | 38.4 | 37.5 | 44.0 | 44.7 | 46.9 | 60.6 | 53.5 | 39.4 | 38.2 |
| wizardlm | 34.8 | 57.6 | 60.6 | 58.8 | 64.6 | 65.3 | 61.3 | 47.4 | 36.4 | 56.3 | 42.0 | 70.1 | 46.7 | 44.3 | 59.2 | 43.3 | 28.7 | 38.6 | 72.6 | 45.5 | 52.0 | 79.1 | 72.0 | 75.0 | 59.8 | 31.0 | 62.2 | 39.2 | 77.1 | 41.8 | 74.0 | 77.8 | 63.2 | 67.9 | 73.7 | 75.8 | 81.1 | 43.8 | 73.7 | 82.3 | 61.0 | 82.2 | 68.8 | 35.1 | 56.7 | 69.7 | 63.0 | 48.9 | 48.8 | 56.6 | 65.6 | 68.1 | 64.9 | 79.6 | 82.8 | 74.4 | 83.8 | 60.5 |

### Math

#### Qwen2.5-7B

| Method | aime24 | amc23 | gaokao2024_mix | gsm8k | math | minerva_math | olympiadbench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 3.3 | 47.5 | 33.0 | 87.8 | 70.9 | 30.5 | 33.6 | 43.8 |
| infinity-instruct | 16.7 | 47.5 | 28.6 | 88.0 | 68.4 | 27.2 | 31.3 | 44.0 |
| openhermes | 0.0 | 27.5 | 33.0 | 77.9 | 37.6 | 15.8 | 13.8 | 29.4 |
| openr1 | 20.0 | 57.5 | 70.3 | 92.6 | 82.7 | 40.1 | 46.8 | 58.6 |
| scale | 6.7 | 45.0 | 29.7 | 89.2 | 72.2 | 32.7 | 34.5 | 44.3 |
| smoltalk | 6.7 | 55.0 | 47.3 | 81.5 | 68.9 | 28.3 | 33.3 | 45.9 |
| synthetic_1 | 16.7 | 50.0 | 68.1 | 92.6 | 82.4 | 38.2 | 45.9 | 56.3 |
| tulu | 3.3 | 35.0 | 30.8 | 82.3 | 48.6 | 17.3 | 18.8 | 33.7 |
| ultrachat | 0.0 | 15.0 | 27.5 | 80.0 | 47.2 | 15.8 | 16.0 | 28.8 |
| wizardlm | 3.3 | 22.5 | 22.0 | 79.7 | 46.0 | 17.3 | 15.1 | 29.4 |

#### Llama-3.1-8B

| Method | aime24 | amc23 | gaokao2024_mix | gsm8k | math | minerva_math | olympiadbench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 0.0 | 7.5 | 20.9 | 67.4 | 29.2 | 13.6 | 7.6 | 20.9 |
| infinity-instruct | 0.0 | 15.0 | 18.7 | 64.0 | 26.0 | 13.2 | 7.1 | 20.6 |
| openhermes | 0.0 | 7.5 | 20.9 | 58.4 | 15.4 | 6.2 | 4.0 | 16.1 |
| openr1 | 3.3 | 22.5 | 36.3 | 80.9 | 53.0 | 20.6 | 20.7 | 33.9 |
| scale | 3.3 | 10.0 | 15.4 | 77.1 | 36.1 | 14.7 | 10.8 | 23.9 |
| smoltalk | 0.0 | 2.5 | 20.9 | 30.6 | 19.4 | 13.2 | 5.9 | 13.2 |
| synthetic_1 | 0.0 | 20.0 | 37.4 | 81.7 | 47.7 | 16.2 | 17.6 | 31.5 |
| tulu | 0.0 | 12.5 | 12.1 | 64.4 | 20.8 | 16.2 | 5.2 | 18.7 |
| ultrachat | 0.0 | 7.5 | 17.6 | 42.3 | 15.9 | 7.4 | 3.1 | 13.4 |
| wizardlm | 0.0 | 0.0 | 17.6 | 33.4 | 12.1 | 8.8 | 4.0 | 10.8 |

#### Mistral-7B-v0.3

| Method | aime24 | amc23 | gaokao2024_mix | gsm8k | math | minerva_math | olympiadbench | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 0.0 | 15.0 | 14.3 | 57.2 | 17.2 | 9.6 | 4.4 | 16.8 |
| infinity-instruct | 0.0 | 7.5 | 12.1 | 50.9 | 14.3 | 7.4 | 3.6 | 13.7 |
| openhermes | 0.0 | 7.5 | 13.2 | 41.7 | 7.7 | 3.7 | 1.9 | 10.8 |
| openr1 | 3.3 | 32.5 | 36.3 | 80.5 | 51.8 | 17.6 | 22.1 | 34.9 |
| scale | 0.0 | 10.0 | 13.2 | 68.8 | 27.0 | 5.9 | 6.4 | 18.8 |
| smoltalk | 0.0 | 2.5 | 11.0 | 47.0 | 13.6 | 8.5 | 3.4 | 12.3 |
| synthetic_1 | 0.0 | 17.5 | 37.4 | 82.8 | 43.0 | 13.2 | 16.9 | 30.1 |
| tulu | 0.0 | 5.0 | 9.9 | 53.8 | 10.7 | 7.7 | 4.4 | 13.1 |
| ultrachat | 0.0 | 2.5 | 18.7 | 17.3 | 6.0 | 6.6 | 2.2 | 7.6 |
| wizardlm | 0.0 | 0.0 | 13.2 | 20.5 | 7.0 | 5.1 | 2.7 | 6.9 |

### Science

#### Qwen2.5-7B

| Method | ChemBench-multi-choise | ChemBench-str-match | gpqa_diamond | gpqa_main | mmlu | mmlu_pro | piqa | scibench-chemistry | scibench-math | scibench-physics | super_gpqa | Avg | Weighted Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 43.4 | 41.0 | 24.7 | 26.3 | 67.6 | 43.0 | 70.9 | 28.6 | 41.5 | 30.4 | 20.0 | 39.8 | 39.1 |
| infinity-instruct | 44.9 | 35.2 | 22.7 | 23.9 | 67.2 | 43.0 | 79.2 | 32.0 | 42.2 | 30.4 | 18.7 | 39.9 | 38.6 |
| megascience | 46.4 | 38.9 | 35.9 | 29.5 | 73.5 | 54.7 | 76.5 | 33.5 | 39.5 | 34.8 | 29.0 | 44.7 | 47.4 |
| nemotron-science | 49.4 | 28.7 | 25.8 | 25.7 | 70.2 | 47.1 | 82.5 | 14.3 | 29.3 | 13.7 | 25.8 | 37.5 | 43.6 |
| openhermes | 43.9 | 31.6 | 27.8 | 26.3 | 64.2 | 41.0 | 77.4 | 18.4 | 19.0 | 15.4 | 19.8 | 35.0 | 37.8 |
| smoltalk | 42.0 | 43.4 | 27.8 | 25.9 | 66.2 | 44.1 | 76.4 | 30.1 | 36.7 | 31.7 | 20.4 | 40.4 | 39.3 |
| tulu | 45.7 | 34.0 | 25.8 | 27.2 | 67.7 | 45.2 | 77.4 | 20.7 | 29.9 | 22.0 | 22.0 | 38.0 | 40.6 |
| ultrachat | 39.5 | 38.9 | 22.7 | 24.3 | 60.1 | 39.5 | 73.6 | 18.4 | 21.8 | 12.3 | 18.6 | 33.6 | 35.6 |
| wizardlm | 37.8 | 28.3 | 28.3 | 26.6 | 64.9 | 40.6 | 72.6 | 18.8 | 26.5 | 15.0 | 19.6 | 34.4 | 37.4 |

#### Llama-3.1-8B

| Method | ChemBench-multi-choise | ChemBench-str-match | gpqa_diamond | gpqa_main | mmlu | mmlu_pro | piqa | scibench-chemistry | scibench-math | scibench-physics | super_gpqa | Avg | Weighted Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 28.2 | 21.3 | 14.6 | 18.8 | 44.2 | 22.0 | 19.7 | 18.0 | 15.0 | 9.7 | 11.8 | 20.3 | 22.8 |
| infinity-instruct | 37.0 | 19.7 | 15.7 | 16.1 | 49.1 | 24.9 | 37.6 | 13.9 | 14.3 | 12.8 | 11.2 | 22.9 | 25.2 |
| megascience | 48.5 | 21.7 | 27.3 | 24.6 | 63.1 | 40.0 | 59.0 | 17.7 | 19.7 | 13.7 | 21.4 | 32.4 | 37.6 |
| nemotron-science | 45.6 | 15.2 | 19.2 | 24.3 | 59.2 | 33.2 | 70.1 | 3.8 | 12.2 | 1.3 | 19.0 | 27.6 | 34.2 |
| openhermes | 26.6 | 18.0 | 19.7 | 21.4 | 44.3 | 25.3 | 37.1 | 2.3 | 3.4 | 0.4 | 13.3 | 19.3 | 24.5 |
| smoltalk | 32.2 | 22.1 | 19.2 | 19.4 | 44.5 | 23.3 | 38.0 | 10.2 | 11.6 | 4.8 | 12.0 | 21.6 | 23.9 |
| tulu | 40.8 | 23.4 | 21.2 | 20.8 | 53.3 | 27.6 | 55.2 | 9.8 | 10.2 | 9.3 | 15.5 | 26.1 | 29.4 |
| ultrachat | 20.4 | 18.9 | 14.1 | 19.9 | 43.8 | 27.7 | 17.4 | 3.4 | 6.8 | 1.8 | 15.0 | 17.2 | 24.8 |
| wizardlm | 24.6 | 15.2 | 13.6 | 20.3 | 48.4 | 27.5 | 36.2 | 8.6 | 8.2 | 5.3 | 14.1 | 20.2 | 26.2 |

#### Mistral-7B-v0.3

| Method | ChemBench-multi-choise | ChemBench-str-match | gpqa_diamond | gpqa_main | mmlu | mmlu_pro | piqa | scibench-chemistry | scibench-math | scibench-physics | super_gpqa | Avg | Weighted Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataflow | 32.6 | 13.5 | 9.1 | 13.6 | 44.6 | 18.7 | 42.5 | 7.1 | 8.8 | 4.4 | 12.3 | 18.9 | 23.2 |
| infinity-instruct | 30.9 | 16.0 | 14.1 | 16.7 | 42.8 | 18.5 | 47.1 | 6.4 | 8.8 | 5.3 | 9.0 | 19.6 | 21.3 |
| megascience | 43.6 | 16.8 | 22.2 | 19.4 | 56.0 | 31.4 | 66.7 | 9.0 | 8.2 | 10.1 | 17.9 | 27.4 | 32.4 |
| nemotron-science | 44.7 | 10.2 | 16.2 | 16.7 | 53.9 | 26.7 | 75.2 | 3.4 | 2.7 | 3.1 | 16.2 | 24.5 | 30.3 |
| openhermes | 16.6 | 10.7 | 8.1 | 10.3 | 35.6 | 15.4 | 46.2 | 0.4 | 3.4 | 1.3 | 9.6 | 14.3 | 18.4 |
| smoltalk | 24.3 | 15.6 | 14.1 | 17.0 | 36.0 | 15.9 | 32.8 | 4.9 | 6.8 | 2.6 | 8.7 | 16.2 | 18.2 |
| tulu | 35.9 | 13.1 | 23.7 | 21.4 | 43.4 | 17.8 | 56.4 | 5.6 | 10.9 | 3.5 | 10.8 | 22.1 | 22.7 |
| ultrachat | 19.1 | 11.9 | 12.6 | 13.4 | 38.8 | 18.3 | 35.9 | 4.5 | 5.4 | 4.0 | 11.9 | 16.0 | 20.7 |
| wizardlm | 18.9 | 12.3 | 14.1 | 10.3 | 37.8 | 16.3 | 39.9 | 3.8 | 4.8 | 1.8 | 11.0 | 15.5 | 19.7 |

### Medicine

#### Llama-3.1-8B

| Method | MedCaseReasoning | MedMCQA | MedR-Bench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 21.0 | 47.1 | 79.8 | 49.3 |
| infinity-instruct | 19.0 | 46.6 | 78.0 | 47.8 |
| openhermes | 20.1 | 48.0 | 78.4 | 48.8 |
| smoltalk | 19.1 | 41.5 | 79.5 | 46.7 |
| tulu | 21.0 | 45.2 | 76.1 | 47.4 |
| ultrachat | 19.5 | 47.8 | 80.7 | 49.3 |
| ultramedical | 22.0 | 62.4 | 74.5 | 53.0 |
| wizardlm | 19.3 | 47.5 | 76.5 | 47.8 |

#### Mistral-7B-v0.3

| Method | MedCaseReasoning | MedMCQA | MedR-Bench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 14.5 | 47.1 | 70.3 | 44.0 |
| infinity-instruct | 14.8 | 40.4 | 72.1 | 42.5 |
| openhermes | 12.5 | 42.0 | 71.9 | 42.1 |
| smoltalk | 16.3 | 42.2 | 79.1 | 45.8 |
| tulu | 15.3 | 42.6 | 71.0 | 43.0 |
| ultrachat | 15.5 | 43.0 | 75.9 | 44.8 |
| ultramedical | 17.4 | 58.9 | 69.0 | 48.4 |
| wizardlm | 15.7 | 40.2 | 78.8 | 44.9 |

#### Qwen2.5-7B

| Method | MedCaseReasoning | MedMCQA | MedR-Bench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 18.5 | 55.2 | 77.2 | 50.3 |
| infinity-instruct | 19.1 | 54.9 | 74.4 | 49.5 |
| openhermes | 17.4 | 55.0 | 80.0 | 50.8 |
| smoltalk | 18.1 | 55.3 | 80.9 | 51.4 |
| tulu | 17.8 | 52.1 | 77.5 | 49.1 |
| ultrachat | 18.8 | 55.4 | 77.6 | 50.6 |
| ultramedical | 19.8 | 66.5 | 72.4 | 52.9 |
| wizardlm | 18.2 | 54.3 | 78.3 | 50.3 |

### Finance

#### Llama-3.1-8B

| Method | CPA-KQA | FinEval-KR | XFinBench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 27.1 | 30.7 | 59.3 | 39.0 |
| finance-instruct | 30.0 | 35.6 | 58.2 | 41.3 |
| infinity-instruct | 37.6 | 41.6 | 63.7 | 47.6 |
| openhermes | 38.6 | 36.6 | 57.2 | 44.1 |
| smoltalk | 29.0 | 33.7 | 54.9 | 39.2 |
| tulu | 38.1 | 35.6 | 62.3 | 45.3 |
| ultrachat | 32.9 | 33.7 | 55.9 | 40.8 |
| wizardlm | 32.4 | 30.7 | 57.0 | 40.0 |

#### Mistral-7B-v0.3

| Method | CPA-KQA | FinEval-KR | XFinBench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 26.2 | 25.7 | 56.1 | 36.0 |
| finance-instruct | 23.3 | 28.7 | 54.9 | 35.7 |
| infinity-instruct | 33.3 | 39.6 | 56.6 | 43.2 |
| openhermes | 25.2 | 30.7 | 54.3 | 36.7 |
| smoltalk | 20.5 | 22.8 | 50.3 | 31.2 |
| tulu | 20.5 | 23.8 | 53.3 | 32.5 |
| ultrachat | 20.0 | 21.8 | 48.0 | 29.9 |
| wizardlm | 25.7 | 31.7 | 54.0 | 37.1 |

#### Qwen2.5-7B

| Method | CPA-KQA | FinEval-KR | XFinBench | Avg |
| --- | --- | --- | --- | --- |
| dataflow | 56.7 | 64.4 | 62.3 | 61.1 |
| finance-instruct | 57.6 | 55.4 | 58.6 | 57.2 |
| infinity-instruct | 61.4 | 61.4 | 63.9 | 62.2 |
| openhermes | 58.6 | 57.4 | 61.1 | 59.0 |
| smoltalk | 59.0 | 63.4 | 60.9 | 61.1 |
| tulu | 59.5 | 64.4 | 62.5 | 62.1 |
| ultrachat | 58.1 | 61.4 | 59.1 | 59.5 |
| wizardlm | 60.0 | 64.4 | 63.7 | 62.7 |

### Law

#### Llama-3.1-8B

| Method | LegalBench | LexGLUE | Avg |
| --- | --- | --- | --- |
| dataflow | 83.5 | 51.2 | 67.4 |
| infinity-instruct | 91.2 | 68.9 | 80.1 |
| lawyer-llama | 84.2 | 56.3 | 70.2 |
| openhermes | 91.4 | 66.2 | 78.8 |
| smoltalk | 90.0 | 69.4 | 79.7 |
| tulu | 91.0 | 65.3 | 78.1 |
| ultrachat | 89.6 | 66.3 | 77.9 |
| wizardlm | 90.1 | 66.9 | 78.5 |

#### Mistral-7B-v0.3

| Method | LegalBench | LexGLUE | Avg |
| --- | --- | --- | --- |
| dataflow | 82.0 | 41.3 | 61.6 |
| infinity-instruct | 90.9 | 61.8 | 76.4 |
| lawyer-llama | 89.2 | 51.5 | 70.3 |
| openhermes | 91.1 | 64.4 | 77.7 |
| smoltalk | 88.9 | 48.6 | 68.8 |
| tulu | 89.9 | 56.5 | 73.2 |
| ultrachat | 90.7 | 60.3 | 75.5 |
| wizardlm | 92.3 | 29.0 | 60.7 |

#### Qwen2.5-7B

| Method | LegalBench | LexGLUE | Avg |
| --- | --- | --- | --- |
| dataflow | 85.6 | 64.6 | 75.1 |
| infinity-instruct | 91.0 | 64.7 | 77.8 |
| lawyer-llama | 80.4 | 62.2 | 71.3 |
| openhermes | 91.5 | 61.0 | 76.3 |
| smoltalk | 87.1 | 65.3 | 76.2 |
| tulu | 92.3 | 63.9 | 78.1 |
| ultrachat | 85.6 | 64.4 | 75.0 |
| wizardlm | 91.9 | 64.8 | 78.4 |
