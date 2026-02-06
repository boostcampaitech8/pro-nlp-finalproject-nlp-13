from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from peft import LoraConfig as PeftLoraConfig
from trl import SFTConfig, SFTTrainer


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
    output_dir: str = "./outputs/sft"

    # training
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    fp16: bool = True
    bf16: bool = False

    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    packing: bool = False
    report_to: str = "none"
    seed: int = 42

    remove_unused_columns: bool = False


def _build_sft_args(cfg: SftTrainConfig) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optim,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        dataset_text_field="text",
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=cfg.remove_unused_columns,
    )


def _build_peft_config(cfg: CustomLoraConfig) -> Optional[PeftLoraConfig]:
    if not cfg.enabled:
        return None
    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=cfg.target_modules,
        modules_to_save=cfg.modules_to_save,
    )


def build_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    sft_cfg: SftTrainConfig,
    lora_cfg: CustomLoraConfig,
    collator=None,
    formatting_func=None,
) -> SFTTrainer:
    args = _build_sft_args(sft_cfg)
    peft_config = _build_peft_config(lora_cfg)

    return SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=collator,
        formatting_func=formatting_func,
    )