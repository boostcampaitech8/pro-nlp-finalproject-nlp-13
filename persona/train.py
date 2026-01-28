import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from persona.utils.config import load_config
from persona.utils.set_seed import set_seed
from persona.utils.setup_wandb import setup_wandb
from persona.training.trainer import PersonaTrainer


def main(config_path: str = "/data/ephemeral/pro-nlp-finalproject-nlp-13/persona/configs/train_configs.yaml"):
    
    # Config 로드
    config = load_config(config_path)
    
    # Seed 설정
    set_seed(config['training']['seed'])
    
    # Wandb 초기화
    wandb_run = setup_wandb(config)
    
    # Trainer 초기화 및 실행
    trainer = PersonaTrainer(config)
    
    trainer.setup_tokenizer()
    trainer.prepare_data()
    trainer.setup_model()
    trainer.setup_data_collator()
    trainer.setup_trainer()
    trainer.train()
    trainer.save()
    
    # Wandb 종료
    if wandb_run is not None:
        wandb_run.finish()
        print("✅ Wandb run finished!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EXAONE LoRA Persona Training")
    parser.add_argument(
        "--config",
        type=str,
        default="persona/configs/train_configs.yaml",
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    main(args.config)