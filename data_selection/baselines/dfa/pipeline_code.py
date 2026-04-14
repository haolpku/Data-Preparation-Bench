import os
from dataflow.utils.registry import OPERATOR_REGISTRY
if hasattr(OPERATOR_REGISTRY, "_get_all"):
    OPERATOR_REGISTRY._get_all()

from dataflow.utils.storage import FileStorage
from dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter
from dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter

# Initialize LLM serving (if required by any operator)
os.environ["API_KEY"] = "sk-URhU1qxF5iVuW2ZZND5fI0pQcuESJyBt33dv4WGf3JY9dZbI"

# Initialize storage
storage = FileStorage(
    first_entry_file_name="/home/hxy/dcai/Data-Preparation-Bench/data_selection/dataset/teknium/OpenHermes-2.5/processed/openhermes2_5_alpaca_sample.jsonl",
    cache_path="/home/hxy/dcai/Data-Preparation-Bench/data_selection/output/data/processed_dfa_20260414_021910/",
    file_name_prefix="dataflow_cache_step",
    cache_type="jsonl"
)

# Initialize operators
deita_quality_filter = DeitaQualityFilter(
    min_score=2.5,
    max_score=10000.0,
    device='cuda',
    model_cache_dir='./dataflow_cache',
    max_length=512
)

superfiltering_filter = SuperfilteringFilter(
    min_score=0.0,
    max_score=1.0,
    device='cuda',
    model_cache_dir='./dataflow_cache',
    max_length=512
)

# Run pipeline
# Step 1: Run DeitaQualityFilter
deita_output_key = deita_quality_filter.run(
    storage.step(),
    input_instruction_key='instruction',
    input_output_key='output',
    output_key='DeitaQualityScore'
)

# Step 2: Run SuperfilteringFilter
superfiltering_output_key = superfiltering_filter.run(
    storage.step(),
    input_instruction_key='instruction',
    input_input_key='input',
    input_output_key='output',
    output_key='SuperfilteringScore'
)

# Final output
output = {
    "DeitaQualityFilterOutput": deita_output_key,
    "SuperfilteringFilterOutput": superfiltering_output_key
}

print(output)