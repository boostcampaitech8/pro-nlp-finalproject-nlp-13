from dataclasses import dataclass
from typing import Optional, List, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    use_8bit: bool = False
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    use_gradient_checkpointing: bool = True
    trust_remote_code: bool = True


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Union[str, List[str]] = None
    adapter_path: Optional[str] = None
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    use_dora: bool = False
    
    def __post_init__(self):
        # default target_modules 설정
        if self.target_modules is None:
            object.__setattr__(self, 'target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "up_proj", "down_proj"
            ])


def _get_bnb_config(model_cfg: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """BitsAndBytes 양자화 설정 생성"""
    if model_cfg.use_8bit and model_cfg.use_4bit:
        raise ValueError("Cannot use both 8bit and 4bit quantization!")
    
    if model_cfg.use_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    elif model_cfg.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_cfg.compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant
        )
    return None


def load_model(model_cfg: ModelConfig, lora_cfg: Optional[LoRAConfig] = None):
    """
    모델 로드 및 LoRA 적용 (학습용)
    
    Args:
        model_cfg: 모델 설정
        lora_cfg: LoRA 설정 (None이면 LoRA 적용 안 함)
    
    Returns:
        LoRA가 적용된 모델
    """
    bnb_config = _get_bnb_config(model_cfg)
    
    print(f"Loading Base Model: {model_cfg.model_name_or_path}")
    
    # 양자화 여부 출력
    if model_cfg.use_8bit:
        print("✅ Using 8-bit quantization")
    elif model_cfg.use_4bit:
        print("✅ Using 4-bit quantization")
    else:
        print("✅ Using FP16 (no quantization)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        use_cache=False if model_cfg.use_gradient_checkpointing else True,
    )
    
    print(f"✅ Model loaded - Parameters: {model.num_parameters():,}")
    
    # 양자화 모델 준비
    if model_cfg.use_8bit or model_cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)
        print("✅ Model prepared for k-bit training")
    
    # Gradient checkpointing
    if model_cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # LoRA 적용
    if lora_cfg is not None:
        print("\n" + "-"*70)
        print("Applying LoRA...")
        print("-"*70)
        
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
            use_dora=lora_cfg.use_dora
        )
        model = get_peft_model(model, peft_config)
        
        print(f"✅ LoRA applied")
        print(f"   - r: {lora_cfg.r}")
        print(f"   - alpha: {lora_cfg.lora_alpha}")
        print(f"   - dropout: {lora_cfg.lora_dropout}")
        print(f"   - target_modules: {lora_cfg.target_modules}")
        
        print("\nTrainable Parameters:")
        model.print_trainable_parameters()
    
    return model


def load_model_inference(
    model_cfg: ModelConfig,
    adapter_path: str,
):
    """
    모델 로드 및 LoRA 어댑터 적용 (추론용)
    
    Args:
        model_cfg: 모델 설정
        adapter_path: LoRA 어댑터 경로
    
    Returns:
        추론용 모델
    """
    bnb_config = _get_bnb_config(model_cfg)
    
    print(f"Loading Base Model for Inference: {model_cfg.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,        
        use_cache=True, 
    )
    
    print(f"Loading LoRA Adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False
    )
    
    model.eval()
    return model


def load_tokenizer(model_name_or_path: str):
    """토크나이저 로드"""
    print(f"Loading Tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # padding token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_model_config_from_yaml(config: dict) -> ModelConfig:
    """YAML config에서 ModelConfig 생성"""
    quant_config = config['model']['quantization']
    
    return ModelConfig(
        model_name_or_path=config['model']['name'],
        use_8bit=quant_config.get('load_in_8bit', False) if quant_config['enabled'] else False,
        use_4bit=quant_config.get('load_in_4bit', False) if quant_config['enabled'] else False,
        use_gradient_checkpointing=config['training']['gradient_checkpointing'],
    )


def create_lora_config_from_yaml(config: dict) -> LoRAConfig:
    """YAML config에서 LoRAConfig 생성"""
    lora_config = config['lora']
    
    return LoRAConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config['bias'],
    )