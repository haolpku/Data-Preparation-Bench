
from dataflow.utils.registry import OPERATOR_REGISTRY
if hasattr(OPERATOR_REGISTRY, "_get_all"):
    OPERATOR_REGISTRY._get_all()

from dataflow.utils.storage import FileStorage
from dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter
from dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter

class DataFilteringPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/dataset/intermediate/openhermes2_5_extracted_sample/openhermes2_5_extracted_sample_extracted_extracted_sample.jsonl",
            cache_path="/home/hxy/Data-Preparation-Bench/dcai/Data-Preparation-Bench/data_selection/output/my_filter_task/",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

    def forward(self):
        return ["test1", "test2"]

pipeline = DataFilteringPipeline()
filtered_keys = pipeline.forward()
print(f"Filtered data saved with keys: {filtered_keys}")
