# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Any, Dict, Optional

# import torch
# from peft import LoraConfig as PeftLoraConfig, get_peft_model
# from transformers import EarlyStoppingCallback, TrainingArguments
# from trl import SFTTrainer


# @dataclass(frozen=True)
# class CustomLoraConfig: 
#     enabled: bool = True
#     r: int = 16 
#     alpha: int = 32
#     dropout: float = 0.1
#     target_modules: str | list[str] = "all-linear"
#     bias: str = "none"
#     task_type: str = "CAUSAL_LM"
    
#     modules_to_save: Optional[list[str]] = None
#     lora_alpha: Optional[int] = None


# @dataclass(frozen=True)
# class SftTrainConfig:
#     output_dir: str = "./outputs/sft"

#     run_name: Optional[str] = None
#     logging_dir: Optional[str] = None
#     logging_steps: int = 10
#     logging_first_step: bool = True
#     report_to: str | list[str] = "none"

#     num_train_epochs: float = 2.0
#     learning_rate: float = 1e-4
#     weight_decay: float = 0.01
#     warmup_ratio: float = 0.01
#     warmup_steps: int = 0

#     per_device_train_batch_size: int = 2
#     per_device_eval_batch_size: int = 2
#     gradient_accumulation_steps: int = 8
#     max_grad_norm: float = 1.0

#     optim: str = "adamw_torch" 
#     adam_beta1: float = 0.9
#     adam_beta2: float = 0.999
#     adam_epsilon: float = 1e-8
#     lr_scheduler_type: str = "cosine"

#     eval_strategy: str = "steps"
#     eval_steps: int = 300
#     eval_delay: int = 0
#     eval_on_start: bool = False

#     save_strategy: str = "steps"
#     save_steps: int = 200
#     save_total_limit: int = 2
#     save_only_model: bool = False
#     save_safetensors: bool = True

#     load_best_model_at_end: bool = True
#     metric_for_best_model: str = "eval_loss"
#     greater_is_better: bool = False

#     early_stopping_patience: int = 0
#     early_stopping_threshold: float = 0.0

#     max_seq_length: int = 2048
#     fp16: bool = True 
#     bf16: bool = False

#     gradient_checkpointing: bool = True
#     gradient_checkpointing_kwargs: Optional[dict] = None

#     seed: int = 42
#     data_seed: Optional[int] = None
#     dataloader_num_workers: int = 0
#     dataloader_pin_memory: bool = True
#     remove_unused_columns: bool = False

#     resume_from_checkpoint: Optional[str] = None
#     ignore_data_skip: bool = False

#     local_rank: int = -1
#     ddp_find_unused_parameters: Optional[bool] = None

#     packing: bool = False


# def apply_lora(model, lora_cfg: CustomLoraConfig):
#     if not lora_cfg.enabled:
#         return model

#     peft_config = PeftLoraConfig(
#         r=lora_cfg.r,
#         lora_alpha=lora_cfg.alpha,
#         lora_dropout=lora_cfg.dropout,
#         bias=lora_cfg.bias,
#         task_type=lora_cfg.task_type,
#         target_modules=lora_cfg.target_modules,
#         modules_to_save=lora_cfg.modules_to_save,
#     )

#     model = get_peft_model(model, peft_config)

#     try:
#         model.print_trainable_parameters()
#     except Exception:
#         pass

#     return model


# def build_training_args(cfg: SftTrainConfig) -> TrainingArguments:
#     logging_dir = cfg.logging_dir or f"{cfg.output_dir}/logs"
#     gradient_ckpt_kwargs = cfg.gradient_checkpointing_kwargs
#     if cfg.gradient_checkpointing and gradient_ckpt_kwargs is None:
#         gradient_ckpt_kwargs = {"use_reentrant": False}

#     return TrainingArguments(
#         output_dir=cfg.output_dir,
#         run_name=cfg.run_name,
#         logging_dir=logging_dir,
#         logging_steps=cfg.logging_steps,
#         logging_first_step=cfg.logging_first_step,
#         report_to=cfg.report_to,
#         num_train_epochs=cfg.num_train_epochs,
#         learning_rate=cfg.learning_rate,
#         weight_decay=cfg.weight_decay,
#         warmup_ratio=cfg.warmup_ratio,
#         warmup_steps=cfg.warmup_steps,
#         per_device_train_batch_size=cfg.per_device_train_batch_size,
#         per_device_eval_batch_size=cfg.per_device_eval_batch_size,
#         gradient_accumulation_steps=cfg.gradient_accumulation_steps,
#         max_grad_norm=cfg.max_grad_norm,
#         optim=cfg.optim,
#         adam_beta1=cfg.adam_beta1,
#         adam_beta2=cfg.adam_beta2,
#         adam_epsilon=cfg.adam_epsilon,
#         lr_scheduler_type=cfg.lr_scheduler_type,
#         eval_strategy=cfg.eval_strategy,
#         eval_steps=cfg.eval_steps,
#         eval_delay=cfg.eval_delay,
#         eval_on_start=cfg.eval_on_start,
#         save_strategy=cfg.save_strategy,
#         save_steps=cfg.save_steps,
#         save_total_limit=cfg.save_total_limit,
#         save_only_model=cfg.save_only_model,
#         save_safetensors=cfg.save_safetensors,
#         load_best_model_at_end=cfg.load_best_model_at_end,
#         metric_for_best_model=cfg.metric_for_best_model,
#         greater_is_better=cfg.greater_is_better,
#         fp16=cfg.fp16,
#         bf16=cfg.bf16,
#         gradient_checkpointing=cfg.gradient_checkpointing,
#         gradient_checkpointing_kwargs=gradient_ckpt_kwargs,
#         seed=cfg.seed,
#         data_seed=cfg.data_seed,
#         dataloader_num_workers=cfg.dataloader_num_workers,
#         dataloader_pin_memory=cfg.dataloader_pin_memory,
#         remove_unused_columns=cfg.remove_unused_columns,
#         resume_from_checkpoint=cfg.resume_from_checkpoint,
#         ignore_data_skip=cfg.ignore_data_skip,
#         local_rank=cfg.local_rank,
#         ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
#     )



# def build_sft_trainer(
#     model,
#     tokenizer,
#     train_dataset,
#     eval_dataset,
#     sft_cfg: SftTrainConfig,
#     lora_cfg: CustomLoraConfig,
#     data_collator=None,  # ✅ 외부에서 주입
# ):
#     model = apply_lora(model, lora_cfg)
#     args = build_training_args(sft_cfg)

#     def formatting_func(example: Dict[str, Any]) -> str:
#         messages = example["messages"]
#         return tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False,
#         )

#     callbacks = []
#     if sft_cfg.early_stopping_patience and sft_cfg.early_stopping_patience > 0:
#         callbacks.append(
#             EarlyStoppingCallback(
#                 early_stopping_patience=sft_cfg.early_stopping_patience,
#                 early_stopping_threshold=sft_cfg.early_stopping_threshold,
#             )
#         )

#     return SFTTrainer(
#         model=model,
#         args=args,
#         tokenizer=tokenizer,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         formatting_func=formatting_func,
#         data_collator=data_collator,
#         max_seq_length=sft_cfg.max_seq_length,
#         packing=sft_cfg.packing,
#         callbacks=callbacks,
#     )


from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Union

import torch
from peft import LoraConfig as PeftLoraConfig
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. Config Definitions (설정 클래스)
# ==========================================

@dataclass(frozen=True)
class CustomLoraConfig: 
    enabled: bool = True
    r: int = 16 
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, List[str]] = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None

@dataclass(frozen=True)
class SftTrainConfig:
    # 기본 경로
    output_dir: str = "./outputs/sft"
    
    # 학습 하이퍼파라미터
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4 # VRAM 절약 핵심
    
    # 최적화
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # 로깅 & 평가
    logging_steps: int = 10
    eval_strategy: str = "epoch"   # steps보다 epoch 단위 추천 (데이터 클 경우)
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    
    # 하드웨어 가속 (V100 설정)
    fp16: bool = True       # V100 필수
    bf16: bool = False      # Ampere 이상만 True
    
    # 기타
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    packing: bool = False
    report_to: str = "none" # wandb 사용시 변경
    seed: int = 42

# ==========================================
# 2. Builder Function
# ==========================================

def build_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    sft_cfg: SftTrainConfig,
    lora_cfg: CustomLoraConfig,
    collator=None,         # 외부에서 주입
    formatting_func=None,  # 외부에서 주입
) -> SFTTrainer:
    """
    SFTConfig와 LoraConfig를 조립하여 SFTTrainer를 반환합니다.
    """
    
    print("🛠️  Building SFT Trainer...")

    # 1. SFTConfig 생성 (TRL 전용 설정 클래스 사용)
    # TrainingArguments를 상속받으므로 모든 인자 호환됨
    args = SFTConfig(
        output_dir=sft_cfg.output_dir,
        num_train_epochs=sft_cfg.num_train_epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sft_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        weight_decay=sft_cfg.weight_decay,
        warmup_ratio=sft_cfg.warmup_ratio,
        optim=sft_cfg.optim,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        
        logging_steps=sft_cfg.logging_steps,
        eval_strategy=sft_cfg.eval_strategy,
        save_strategy=sft_cfg.save_strategy,
        save_total_limit=sft_cfg.save_total_limit,
        load_best_model_at_end=sft_cfg.load_best_model_at_end,
        
        fp16=sft_cfg.fp16,
        bf16=sft_cfg.bf16,
        
        gradient_checkpointing=sft_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if sft_cfg.gradient_checkpointing else None,
        
        max_seq_length=sft_cfg.max_seq_length,
        packing=sft_cfg.packing,
        dataset_text_field="text", # formatting_func 사용 시 무시되지만 필수 인자
        report_to=sft_cfg.report_to,
        seed=sft_cfg.seed,
        remove_unused_columns=False, # DataCollator 사용 시 필수 (중요!)
    )

    # 2. LoRA Config 생성
    peft_config = None
    if lora_cfg.enabled:
        peft_config = PeftLoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
            target_modules=lora_cfg.target_modules,
            modules_to_save=lora_cfg.modules_to_save,
        )

    # 3. Callbacks (Early Stopping 등)
    callbacks = []
    # 필요하면 여기에 추가

    # 4. Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,       # 여기서 넘겨주면 내부에서 자동 적용
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=callbacks,
    )

    return trainer