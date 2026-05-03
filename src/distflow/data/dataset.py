import builtins
import random
from functools import cached_property
from typing import Any, Literal, cast

from distflow.data.data_formatter import FormatterProtocol
from distflow.data.types import DatasetProcessOutputItem
from distflow.utils import logger


class DistflowDataset:
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        load_type: Literal["datasets", "modelscope", "pandas"],
        formatter: FormatterProtocol,
        data_size: int = -1,
        name: str = "default",
        split: str = "train",
        sep: str = "\t",
        dtype: str = "str",
        shuffle_seed: int = 42,
        use_json: bool = False,
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.load_type = load_type
        self.formatter = formatter
        self.data_size = data_size
        self._name = name
        self.split = split
        self.sep = sep
        self.dtype = dtype
        self.shuffle_seed = shuffle_seed
        self.use_json = use_json

    @property
    def name(self):
        return self.dataset_name

    @cached_property
    def _data_list(self) -> list[DatasetProcessOutputItem]:
        logger.info(
            f"开始加载数据集: {self.dataset_name}, 路径: {self.data_path}, 类型: {self.load_type}"
        )

        # 数据大小
        logger.debug(
            f"数据大小限制: {self.data_size if self.data_size > 0 else '全部'}"
        )

        match self.load_type:
            case "datasets":
                from datasets import load_dataset

                logger.debug(
                    f"使用 datasets 加载, split={self.split}, use_json={self.use_json}"
                )
                if self.use_json:
                    dataset = load_dataset(
                        "json", data_files=self.data_path, split=self.split
                    )
                else:
                    dataset = load_dataset(
                        path=self.data_path, name=self._name, split=self.split
                    )
            case "modelscope":
                from modelscope.msdatasets import MsDataset

                logger.debug(f"使用 modelscope 加载, split={self.split}")
                dataset = MsDataset.load(
                    self.data_path, subset_name=self._name, split=self.split
                )
            case "pandas":
                from datasets import Dataset, load_dataset
                from pandas import read_csv

                logger.debug("使用 pandas 加载")
                dtype_actual = getattr(builtins, self.dtype)
                df = read_csv(self.data_path, sep=self.sep, dtype=dtype_actual)
                dataset = Dataset.from_pandas(df)
            case _:
                raise ValueError(f"不支持的 load_type: {self.load_type}")

        logger.info(f"数据集加载完成，总样本数: {len(dataset)}")

        random.seed(self.shuffle_seed)
        logger.debug(f"使用随机种子: {self.shuffle_seed}")
        random_indices = list(range(len(dataset)))
        if self.data_size > 0 and self.data_size < len(dataset):
            logger.info(f"随机采样 {self.data_size} 条数据")
            random_indices = random.sample(random_indices, self.data_size)
        else:
            logger.info("使用全部数据")
            random.shuffle(random_indices)
        sampled_data = cast(list[dict[str, Any]], [dataset[i] for i in random_indices])
        logger.debug(f"采样完成，开始格式化数据")
        formatted_data = [
            self.formatter.format(data_item) for data_item in sampled_data
        ]
        return formatted_data

    def load(self) -> list[DatasetProcessOutputItem]:
        return self._data_list
