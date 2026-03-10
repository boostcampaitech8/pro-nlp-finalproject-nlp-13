import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Union


def load_raw_dataset(
    data_path: str,
    test_size: float = 0.1,
    seed: int = 42
) -> Union[Dataset, DatasetDict]:
    print(f"Loading raw data from {data_path}...")
    
    df = pd.read_csv(data_path)

    dataset = Dataset.from_pandas(df)

    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns("__index_level_0__")

    if "test" not in dataset:
        dataset = dataset.train_test_split(
            test_size=test_size,
            seed=seed
        )
    
    return dataset