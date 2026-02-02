from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    compute_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = True