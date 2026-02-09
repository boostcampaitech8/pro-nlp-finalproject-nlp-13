import torch
from typing import Optional
from dataclasses import dataclass
from transformers import BitsAndBytesConfig


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = True


def _get_bnb_config(model_cfg: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if not model_cfg.use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_cfg.compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant
    )