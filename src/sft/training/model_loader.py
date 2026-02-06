from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass(frozen=True)
class ModelConfig:
    name: str

@dataclass(frozen=True)
class TrainConfig:
    fp16: bool = True
    bf16: bool = False


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = model_cfg.name
    print(f"[Loading Model]: {model_name}")

    if train_cfg.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return model, tokenizer