# Experiment Settings

## MMD Computation

For MMD computation, please refer to [compute_mmd.py](./examples/compute_mmd.py). Key hyperparameters are configured as follows:

- Truncate length: 40960
- Kernel type: RBF
- Kernel sigma: 1.0 (constant)
- Estimator: Biased MMD estimator
- Dataset size: 5000 samples with seed 42

**Reference Datasets:**
- Math: [ODA-Math-460k](https://huggingface.co/datasets/OpenDataArena/ODA-Math-460k)
- General Text: [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)
- Medical: [ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed)

The experiments were conducted with the following package versions:

- vllm: 0.8.5.post1
- torch: 2.6.0
- transformers: 4.53.0

## Training

Training is conducted using [LlamaFactory](https://github.com/hiyouga/LlamaFactory). Base models include:
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

**Training Configuration:**
```yaml
cutoff_len: 32768
packing: false
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

We use DeepSpeed ZeRO-3 for distributed training. Chat templates are set according to model families:
- `qwen` for Qwen2.5-7B
- `llama3` for Llama-3.1-8B and Llama-3.2-3B
- `mistral` for Mistral-7B-v0.3

**Training Datasets:**

| Domain | Dataset | Samples |
|--------|---------|---------|
| Math | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 20,000 |
| Math | [ScaleQuest-Math](https://huggingface.co/datasets/dyyyyyyyy/ScaleQuest-Math) | 20,000 |
| Math | [Synthetic-1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1) | 20,000 |
| General | [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) | 20,000 |
| General | [dataflow-instruct-10k](https://huggingface.co/datasets/OpenDCAI/dataflow-instruct-10k) | 10,000 |
| General | [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 20,000 |
| General | [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 20,000 |
| General | [WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | 20,000 |
| General | [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 20,000 |
| General | [smoltalk-chinese](https://huggingface.co/datasets/opencsg/smoltalk-chinese) | 20,000 |
| Medical | [UltraMedical](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical) | 20,000 |

## Evaluation

### Math

Math evaluation is performed using [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) with the following generation parameters:

- temperature: 0.6
- max_tokens_per_call: 16384
- top_p: 1
- apply_chat_template: true
- repetition_penalty: 1.1

**Benchmarks:** GSM8K, AMC23, AIME24, Minerva Math, Gaokao2024-Mix, OlympiadBench, and MATH.

### General Text

General text evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with MMLU-Redux as the primary benchmark:

- max_model_len: 32768
- num_fewshot: 5
- apply_chat_template: true

### Medical

Medical evaluation employs [MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA). See the [official evaluation code](https://github.com/TsinghuaC3I/MedXpertQA) for implementation details.

**Evaluation Parameters:**
- method: zero_shot
- prompting-type: cot
- temperature: 0

Models are served using vLLM for inference.
