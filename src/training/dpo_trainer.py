from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from transformers import AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from src.training.model_loader import ModelConfig, _get_bnb_config

@dataclass(frozen=True)
class DPOTrainerConfig:
    output_dir: str
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"

    fp16: bool = True
    bf16: bool = False

    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    logging_steps: int = 5
    eval_steps: int = 20
    save_steps: int = 20

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: int = 2

    max_length: int = 2048
    max_prompt_length: int = 1536
    max_completion_length: int = 512

    remove_unused_columns: bool = False
    report_to: Optional[str] = None
    seed: int = 42


def load_model_with_two_adapters(
    model_cfg: ModelConfig,
    sft_adapter_path: str,
    train_adapter_name: str = "policy",
    ref_adapter_name: str = "ref"
):
    if not Path(sft_adapter_path).exists():
            raise FileNotFoundError(
                f"SFT adapter가 {sft_adapter_path}에 존재하지 않습니다.\n"
                "SFT 학습을 먼저하거나, config YAML 파일 경로를 확인하세요."
            )
    
    bnb_config = _get_bnb_config(model_cfg)

    print(f"\n ===== DPF Models 로딩 =====") 
    print(f"\n  1. Base Model 로딩")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        use_cache=False,
    )
    base_model = prepare_model_for_kbit_training(base_model)


    print(f"\n   2. SFT 모델 로딩 : {sft_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=True,
        adapter_name=train_adapter_name,
    )

    print(f"\n   3. Reference 모델 로딩 : {sft_adapter_path}")
    model.load_adapter(
        sft_adapter_path,
        is_trainable=False,
        adapter_name=ref_adapter_name,
    )

    model.set_adapter(train_adapter_name)
    print("\n !!!!! 모델 로드 성공 !!!!!")
    print(f"  - Train adapter: '{train_adapter_name}' (trainable)")
    print(f"  - Reference adapter: '{ref_adapter_name}' (frozen)")
    print("-" * 60 + "\n")
    
    return model

def build_dpo_trainer(
    dpo_cfg: DPOTrainerConfig,
    model_cfg: ModelConfig,
    sft_adapter_path: str,
    tokenizer,
    train_dataset,
    eval_dataset=None,
) -> DPOTrainer:

    model = load_model_with_two_adapters(
        model_cfg=model_cfg,
        sft_adapter_path=sft_adapter_path,
        train_adapter_name="policy",
        ref_adapter_name="ref"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    training_args = DPOConfig(
        output_dir=dpo_cfg.output_dir,

        num_train_epochs=dpo_cfg.num_train_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        gradient_checkpointing=model_cfg.use_gradient_checkpointing,

        learning_rate=dpo_cfg.learning_rate,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        warmup_ratio=dpo_cfg.warmup_ratio,

        beta=dpo_cfg.beta,
        label_smoothing=dpo_cfg.label_smoothing,
        loss_type=dpo_cfg.loss_type,

        fp16=dpo_cfg.fp16,
        bf16=dpo_cfg.bf16,

        eval_strategy=dpo_cfg.eval_strategy,
        save_strategy=dpo_cfg.save_strategy,
        logging_steps=dpo_cfg.logging_steps,
        eval_steps=dpo_cfg.eval_steps,
        save_steps=dpo_cfg.save_steps,

        load_best_model_at_end=dpo_cfg.load_best_model_at_end,
        metric_for_best_model=dpo_cfg.metric_for_best_model,
        greater_is_better=dpo_cfg.greater_is_better,
        save_total_limit=dpo_cfg.save_total_limit,

        remove_unused_columns=dpo_cfg.remove_unused_columns,
        report_to=dpo_cfg.report_to,
        seed=dpo_cfg.seed,

        max_length=dpo_cfg.max_length,
        max_prompt_length=dpo_cfg.max_prompt_length,
        max_completion_length=dpo_cfg.max_completion_length,

        model_adapter_name="policy",
        ref_adapter_name="ref",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    return trainer
