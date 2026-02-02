import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from datasets import Dataset

@dataclass(frozen=True)
class DPODataConfig:
    train_path: str
    eval_path: Optional[str] = None
    seed: int = 42
    csv_dir: Optional[str] = None
    margin_threshold: float = 0.995
    eval_ratio: float = 0.1


def load_dpo_dataset(json_path: str, eval_split_ratio: float = 0.9):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dpo_data = []
    for item in data:
        dpo_data.append({
            "prompt": item["instruction"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })

    dataset = Dataset.from_list(dpo_data)

    split_dataset = dataset.train_test_split(
        test_size=eval_split_ratio,
        seed=42
    )

    print(f"--- 데이터셋 로드 완료 ---")
    print(f"  - Train samples: {len(split_dataset['train'])}")
    print(f"  - Eval samples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']