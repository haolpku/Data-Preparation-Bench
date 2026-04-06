import os
from dataflow.utils.registry import OPERATOR_REGISTRY
if hasattr(OPERATOR_REGISTRY, "_get_all"):
    OPERATOR_REGISTRY._get_all()

from dataflow.utils.storage import FileStorage
from dataflow.operators.text_sft.filter.deita_quality_filter import DeitaQualityFilter
from dataflow.operators.text_sft.filter.superfiltering_filter import SuperfilteringFilter

class DataFilteringPipeline:
    def __init__(self):
        # Initialize operators
        self.deita_quality_filter = DeitaQualityFilter(min_score=2.5, max_score=10000.0, device='cuda', model_cache_dir='./dataflow_cache', max_length=512)
        self.superfiltering_filter = SuperfilteringFilter(min_score=0.0, max_score=1.0, device='cuda', model_cache_dir='./dataflow_cache', max_length=512)

    def run(self, storage):
        # Sequential calls to operators
        self.deita_quality_filter.run(storage.step(), input_instruction_key='instruction', input_output_key='output', output_key='DeitaQualityScore')
        self.superfiltering_filter.run(storage.step(), input_instruction_key='instruction', input_input_key='input', input_output_key='output', output_key='SuperfilteringScore')

if __name__ == "__main__":
    storage = FileStorage(first_entry_file_name="/home/hxy/dcai/data_selection/dataset/dfa_databricks-dolly_extracted_sample.jsonl", cache_path="/home/hxy/dcai/data_selection/output/data_selection/dataset_dfa_20260404_071522/", file_name_prefix="dataflow_cache_step", cache_type="jsonl")
    pipeline = DataFilteringPipeline()
    pipeline.run(storage)