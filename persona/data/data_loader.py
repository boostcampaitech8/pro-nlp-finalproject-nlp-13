from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from data.preprocessor import clean_dataframe
from persona.utils.set_seed import set_seed


@dataclass(frozen=True)
class DataConfig:
    """데이터 설정"""
    train_path: Union[str, Path]
    usecols: Optional[List[str]] = None
    test_size: float = 0.1
    seed: int = 42
    do_split: bool = True


def load_csv(path: Union[str, Path], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """CSV 로드"""
    print(f"CSV 로딩: {path}")
    if usecols:
        print(f"  - 선택된 컬럼: {usecols}")
        return pd.read_csv(path, usecols=usecols)
    return pd.read_csv(path)


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    """DataFrame → Dataset 변환"""
    return Dataset.from_pandas(df, preserve_index=False)


def make_train_test_dataset(data_cfg: DataConfig) -> DatasetDict:
    """
    Train/Test Dataset 생성
    
    Args:
        data_cfg: 데이터 설정
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    # Seed 설정
    set_seed(data_cfg.seed)
    
    # CSV 로드
    df = load_csv(data_cfg.train_path, data_cfg.usecols)
    
    # 전처리
    df = clean_dataframe(df)
    
    # Split
    if data_cfg.do_split:
        train_df, test_df = train_test_split(
            df,
            test_size=data_cfg.test_size,
            random_state=data_cfg.seed,
        )
        
        print(f"  - Train: {len(train_df)} 샘플")
        print(f"  - Test: {len(test_df)} 샘플")
        
        train_ds = df_to_dataset(train_df)
        test_ds = df_to_dataset(test_df)
        
        return DatasetDict({
            "train": train_ds,
            "test": test_ds
        })
    else:
        print(f"\n데이터 분할 없음: {len(df)} 샘플")
        dataset = df_to_dataset(df)
        return DatasetDict({"train": dataset})


def create_tokenized_dataset(
    dataset: DatasetDict,
    tokenize_wrapper,
) -> DatasetDict:
    print("Dataset 변환 (messages → text → tokens)")
    
    tokenized = DatasetDict()
    
    for split in dataset.keys():
        print(f"\n[{split}]")
        
        # Step 1: example → messages
        print("  [1/3] Building messages...")
        msg_ds = dataset[split].map(
            tokenize_wrapper.build_messages,
            batched=False,
            remove_columns=dataset[split].column_names,
            desc=f"Build messages ({split})"
        )
        
        # Step 2: messages → text
        print("  [2/3] Converting to text...")
        text_ds = msg_ds.map(
            tokenize_wrapper.to_text,
            batched=False,
            remove_columns=["messages"],
            desc=f"To text ({split})"
        )
        
        # Step 3: text → tokens
        print("  [3/3] Tokenizing...")
        tokenized[split] = text_ds.map(
            tokenize_wrapper.tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenize ({split})"
        )
        
        print(f"  ✅ {split}: {len(tokenized[split])} 샘플")
    
    print("\n✅ 변환 완료!")
    
    return tokenized