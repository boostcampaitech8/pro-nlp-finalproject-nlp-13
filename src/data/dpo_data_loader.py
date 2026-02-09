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

def load_dpo_dataset(json_path: str, tokenizer):
    # train_ds = load_dataset("json", data_files=json_path, field="train", split="train")
    train_ds = load_dataset("json", data_files=json_path, split="train")

    # eval_ds = load_dataset("json", data_files=json_path, field="eval", split="train")

    def process(example):
        full_prompt = tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        )

        return {
            "prompt": full_prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    train_ds = train_ds.map(process, remove_columns=train_ds.column_names)
    # eval_ds = eval_ds.map(process, remove_columns=eval_ds.column_names)
    split_ds = train_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split_ds["train"], split_ds["test"]

    print(f"!!!!! 데이터셋 로드 완료 !!!!!")
    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Eval samples: {len(eval_ds)}")
    
    return train_ds, eval_ds