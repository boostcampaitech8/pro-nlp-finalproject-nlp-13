"""
EXAONE LoRA 페르소나 학습
기존 모듈의 함수를 import해서 사용
"""

import os
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict

from csv_to_dataset import CSVToDataset
from template_wrapper import format_chat_template 

from utils import set_seed, get_model_template_type, print_gpu_utilization

from data_collator import get_data_collator
from callbacks import GenerationCallback


def load_config(config_path):
    """Config 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_wandb(config):
    """Wandb 초기화"""
    if not config['wandb']['enabled']:
        return None
    
    try:
        import wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity'),
            name=config.get('experiment_name'),
            tags=config['wandb'].get('tags', []),
            config=config
        )
        print("✅ Wandb initialized!")
        return wandb
    except ImportError:
        print("⚠️ Wandb not installed")
        return None


def prepare_data(config, tokenizer):      
    print("\n[1/2] CSV → Dataset")
    
    csv_loader = CSVToDataset(
        csv_path=config['data']['train_file'],
        seed=config['data']['seed'],
        usecols=config['data']['usecols']
    )
    
    dataset = csv_loader.convert_to_dataset(
        split_ratio=config['data']['test_size']
    )
    
    print("\n[2/2] Chat template + 토큰화")
    
    def _format(example):
        return format_chat_template(example, tokenizer, config['model']['max_length'])
    
    if isinstance(dataset, DatasetDict):
        tokenized_dataset = DatasetDict()
        for split in dataset.keys():
            tokenized_dataset[split] = dataset[split].map(
                _format,
                num_proc=4,
                remove_columns=['situation', 'context', 'response'],
                desc=f"Formatting {split}"
            )
    else:
        tokenized_dataset = dataset.map(
            _format,
            num_proc=4,
            remove_columns=['situation', 'context', 'response'],
            desc="Formatting"
        )
    
    print("데이터 전처리 완료")
    
    return tokenized_dataset


def setup_tokenizer(config):
    """토크나이저 설정"""
    print("\n" + "="*70)
    print("Tokenizer")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Loaded: {config['model']['name']}")
    print(f"   - Vocab: {len(tokenizer):,}")
    
    return tokenizer


def setup_model(config):
    load_in_8bit = config['model']['quantization'].get('load_in_8bit', False) if config['model']['quantization']['enabled'] else False
    load_in_4bit = config['model']['quantization'].get('load_in_4bit', False) if config['model']['quantization']['enabled'] else False
    
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot use both 8bit and 4bit!")
    
    if load_in_8bit:
        print("✅ 8bit quantization")
    elif load_in_4bit:
        print("✅ 4bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch.float16 if not (load_in_8bit or load_in_4bit) else None,
        device_map="auto",
        use_cache=False
    )
    
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    
    print(f"✅ Loaded")
    print(f"   - Params: {model.num_parameters():,}")
    
    if torch.cuda.is_available():
        print_gpu_utilization()
    
    print("\nApplying LoRA...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    print(f"✅ LoRA applied (r={config['lora']['r']}, alpha={config['lora']['alpha']})")
    model.print_trainable_parameters()
    
    return model


def main(config_path="train_config.yaml"):
    """메인 학습 함수"""
    print("EXAONE LoRA Training")
    
    # Config 로드
    config = load_config(config_path)
    set_seed(config['training']['seed'])
    
    # Wandb
    wandb_run = setup_wandb(config)
    
    # 토크나이저
    tokenizer = setup_tokenizer(config)
    
    # 데이터 전처리 (기존 모듈 활용!)
    dataset = prepare_data(config, tokenizer)
    
    # 모델
    model = setup_model(config)
    
    # DataCollator    
    template_type = config['model'].get('template_type') or get_model_template_type(config['model']['name'])
    
    data_collator = get_data_collator(
        tokenizer=tokenizer,
        template_type=template_type,
        learning_mode=config['data_collator']['learning_mode']
    )
    
    # Training Args
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['logging_dir'], exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        logging_dir=config['paths']['logging_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        optim=config['training']['optim'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        fp16=config['training']['fp16'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        seed=config['training']['seed'],
        report_to="wandb" if config['wandb']['enabled'] else "none",
        run_name=config.get('experiment_name') if config['wandb']['enabled'] else None,
    )
    
    # Callbacks
    callbacks = []
    if config['callbacks']['generation']['enabled']:
        eval_dataset = dataset.get('test') if hasattr(dataset, 'keys') else dataset
        
        callbacks.append(GenerationCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=config['callbacks']['generation']['num_samples'],
            max_new_tokens=config['callbacks']['generation']['max_new_tokens']
        ))
    
    # Trainer
    print("\n" + "="*70)
    print("Trainer")
    print("="*70)
    
    train_dataset = dataset.get('train') if hasattr(dataset, 'keys') else dataset
    eval_dataset = dataset.get('test') if hasattr(dataset, 'keys') else None
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    print(f"✅ Ready!")
    print(f"   - Train: {len(train_dataset):,}")
    print(f"   - Eval: {len(eval_dataset):,}" if eval_dataset else "   - Eval: None")
    print(f"   - Effective batch: {effective_batch}")
    
    print("\n" + "="*70)
    print("🔥 Training Start!")
    print("="*70 + "\n")
    
    trainer.train()
    
    print("\n✅ Training Complete!")
    
    # 모델 저장
    trainer.save_model(config['paths']['output_dir'])
    tokenizer.save_pretrained(config['paths']['output_dir'])
    
    print(f"✅ Saved: {config['paths']['output_dir']}")
    
    if wandb_run:
        wandb_run.finish()
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EXAONE LoRA Training")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    main(args.config)