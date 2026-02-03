from datasets import Dataset, DatasetDict
import pandas as pd
from typing import List, Optional
from utils import set_seed

class CSVToDataset:
    def __init__(self, 
                 csv_path:str,
                 seed: int = 42,
                 deterministic: bool = False,
                usecols: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.seed = seed
        self.deterministic = deterministic
        self.usecols = usecols
        self.dataset = None

        set_seed(self.seed, deterministic=self.deterministic)

    def load_csv(self) ->pd.DataFrame:
        print(f"CSV 로딩: {self.csv_path}")
        if self.usecols is not None:
            print(f"  - 선택된 컬럼: {self.usecols}")
            df = pd.read_csv(self.csv_path, usecols=self.usecols)
        else:
            df = pd.read_csv(self.csv_path)

        return df
    
    def convert_to_dataset(
            self,
            df: Optional[pd.DataFrame] = None,
            split_ratio : Optional[float] = None
    ) -> Dataset:
        
        if df is None:
            df = self.load_csv()

        print("\nDataset으로 변환 중...")
        dataset = Dataset.from_pandas(df)

        if split_ratio is not None:
            dataset = dataset.train_test_split(test_size=split_ratio, seed=self.seed)
            print(f"  - Train 샘플: {len(dataset['train'])}")
            print(f"  - Test 샘플: {len(dataset['test'])}")
        else:
            print(f"  - 전체 샘플: {len(dataset)}")

        self.dataset = dataset
        return dataset
    
