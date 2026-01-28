import os
from transformers import AutoTokenizer, TrainingArguments, Trainer

from persona.data.data_loader import DataConfig, make_train_test_dataset, create_tokenized_dataset
from persona.data.tokenize_wrapper import TokenizeWrapper, TokenizeConfig
from persona.training.load_model import (
    ModelConfig, 
    LoRAConfig, 
    load_model,
    create_model_config_from_yaml,
    create_lora_config_from_yaml
)
from persona.training.data_collator import get_data_collator


def get_model_template_type(model_name: str) -> str:
    """모델명으로 template type 자동 결정"""
    MODEL_MAPPING = {
        "exaone": "exaone",
        "llama": "llama3",
        "mistral": "mistral",
    }
    
    model_name_lower = model_name.lower()
    
    for key, template in MODEL_MAPPING.items():
        if key in model_name_lower:
            return template
    
    print(f"⚠️ Unknown model '{model_name}', using default 'exaone'")
    return "exaone"


class PersonaTrainer:
    """EXAONE LoRA 페르소나 학습 Trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.data_collator = None
        self.trainer = None
    
    def setup_tokenizer(self):
        """토크나이저 설정"""
        print("\n" + "="*70)
        print("Tokenizer Setup")
        print("="*70)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded: {self.config['model']['name']}")
        print(f"   - Vocab size: {len(self.tokenizer):,}")
    
    def prepare_data(self):
        print("Data Preparation")
        
        # DataConfig
        data_cfg = DataConfig(
            train_path=self.config['data']['train_file'],
            usecols=self.config['data']['usecols'],
            test_size=self.config['data']['test_size'],
            seed=self.config['data']['seed'],
            do_split=True
        )
        
        # Train/Test Dataset
        print("\n[Step 1/2] CSV → Dataset")
        dataset = make_train_test_dataset(data_cfg)
        
        # TokenizeWrapper
        tokenize_cfg = TokenizeConfig(
            max_length=self.config['model']['max_length'],
            padding=False,
            truncation=True,
            prompt_path=self.config['model']['prompt_path']
        )
        
        tokenize_wrapper = TokenizeWrapper(self.tokenizer, tokenize_cfg)
        
        # Tokenize
        print("\n[Step 2/2] messages → text → tokens")
        self.dataset = create_tokenized_dataset(
            dataset=dataset,
            tokenize_wrapper=tokenize_wrapper
        )
        
        print("Data preparation complete!")
    
    def setup_model(self):
        model_cfg = create_model_config_from_yaml(self.config)
        lora_cfg = create_lora_config_from_yaml(self.config)
        self.model = load_model(model_cfg, lora_cfg)
    
    def setup_data_collator(self):
        """DataCollator 설정"""
        print("\n" + "="*70)
        print("DataCollator Setup")
        print("="*70)
        
        template_type = self.config['model'].get('template_type') or get_model_template_type(
            self.config['model']['name']
        )
        
        self.data_collator = get_data_collator(
            tokenizer=self.tokenizer,
            template_type=template_type,
            learning_mode=self.config['data_collator']['learning_mode']
        )
    
    def setup_trainer(self):
        """HuggingFace Trainer 설정"""
        print("Trainer Setup")
        
        # 경로 생성
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logging_dir'], exist_ok=True)
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.config['paths']['output_dir'],
            logging_dir=self.config['paths']['logging_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            optim=self.config['training']['optim'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            fp16=self.config['training']['fp16'],
            logging_steps=self.config['training']['logging_steps'],
            logging_first_step=self.config['training']['logging_first_step'],
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            remove_unused_columns=self.config['training']['remove_unused_columns'],
            seed=self.config['training']['seed'],
            report_to="wandb" if self.config['wandb']['enabled'] else "none",
            run_name=self.config.get('experiment_name') if self.config['wandb']['enabled'] else None,
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            data_collator=self.data_collator,
        )
        
        effective_batch = (
            training_args.per_device_train_batch_size * 
            training_args.gradient_accumulation_steps
        )
        
        print(f"   - Train samples: {len(self.dataset['train']):,}")
        if 'test' in self.dataset:
            print(f"   - Test samples: {len(self.dataset['test']):,}")
        print(f"   - Effective batch size: {effective_batch}")
    
    def train(self):
        self.trainer.train()
    
    def save(self):
        """모델 저장"""
        print("\nSaving model...")
        self.trainer.save_model(self.config['paths']['output_dir'])
        self.tokenizer.save_pretrained(self.config['paths']['output_dir'])
        print(f"✅ Model saved: {self.config['paths']['output_dir']}")