import builtins
import random
from typing import Any, Literal, cast

from data_preparation_bench.data.data_formatter import FormatterProtocol
from data_preparation_bench.data.types import DatasetProcessOutputItem
from data_preparation_bench.utils import logger


def load_dataset(
    dataset_name: str,
    data_path: str,
    load_type: Literal["datasets", "modelscope", "pandas"],
    formatter: FormatterProtocol,
    data_size: int = -1,
    split: str = "train",
    sep: str = "\t",
    dtype: str = "str",
    shuffle_seed: int = 42,
    use_json: bool = False,
) -> tuple[str, list[DatasetProcessOutputItem]]:
    logger.info(f"开始加载数据集: {dataset_name}, 路径: {data_path}, 类型: {load_type}")

    # 数据大小
    logger.debug(f"数据大小限制: {data_size if data_size > 0 else '全部'}")

    match load_type:
        case "datasets":
            from datasets import load_dataset

            logger.debug(f"使用 datasets 加载, split={split}, use_json={use_json}")
            if use_json:
                dataset = load_dataset("json", data_files=data_path, split=split)
            else:
                dataset = load_dataset(path=data_path, split=split)
        case "modelscope":
            from modelscope.msdatasets import MsDataset

            logger.debug(f"使用 modelscope 加载, split={split}")
            dataset = MsDataset.load(data_path, split=split)
        case "pandas":
            from datasets import Dataset, load_dataset
            from pandas import read_csv

            logger.debug("使用 pandas 加载")
            dtype_actual = getattr(builtins, dtype)
            df = read_csv(data_path, sep=sep, dtype=dtype_actual)
            dataset = Dataset.from_pandas(df)

    logger.info(f"数据集加载完成，总样本数: {len(dataset)}")

    random.seed(shuffle_seed)
    logger.debug(f"使用随机种子: {shuffle_seed}")
    random_indices = list(range(len(dataset)))
    if data_size > 0 and data_size < len(dataset):
        logger.info(f"随机采样 {data_size} 条数据")
        random_indices = random.sample(random_indices, data_size)
    else:
        logger.info("使用全部数据")
        random.shuffle(random_indices)
    sampled_data = cast(list[dict[str, Any]], [dataset[i] for i in random_indices])
    logger.debug(f"采样完成，开始格式化数据")
    formatted_data = [formatter.format(data_item) for data_item in sampled_data]
    return dataset_name, formatted_data
