import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datasets import load_dataset


@dataclass(frozen=True)
class DPODataConfig:
    train_path: str
    eval_path: Optional[str] = None
    seed: int = 42
    csv_dir: Optional[str] = None
    margin_threshold: float = 0.995
    eval_ratio: float = 0.1


from datasets import load_dataset

def load_dpo_dataset(json_path: str):
    train_ds = load_dataset("json", data_files=json_path, field="train", split="train")
    eval_ds = load_dataset("json", data_files=json_path, field="eval", split="train")

    def process(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    train_ds = train_ds.map(process, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(process, remove_columns=eval_ds.column_names)

    print(f"!!!!! 데이터셋 로드 완료 !!!!!")
    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Eval samples: {len(eval_ds)}")
    
    return train_ds, eval_ds