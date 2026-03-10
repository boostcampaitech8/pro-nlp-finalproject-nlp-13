import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import wandb

import torch
from transformers import AutoTokenizer

from src.dpo.training.model_loader import ModelConfig
from src.dpo.training.dpo_trainer import DPOTrainerConfig, build_dpo_trainer
from src.dpo.data.dpo_data_loader import DPODataConfig, load_dpo_dataset
from src.common.set_seed import set_seed
from src.common.wandb import setup_wandb


def main(
    model_cfg: ModelConfig,
    dpo_cfg: DPOTrainerConfig,
    dpo_data_cfg: DPODataConfig,
    sft_adapter_path: str,
    wandb_cfg: Dict[str, Any] = None,
):
    set_seed(dpo_cfg.seed)


    print("===== Tokenizer 로딩 =====")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token not found, set to eos_token: {tokenizer.eos_token}")


    print("===== DPO 데이터셋 로딩 =====")
    train_dataset, eval_dataset = load_dpo_dataset(
        dpo_data_cfg.train_path,
        tokenizer
    )


    print("===== DPO Trainer (SFT adapter) 빌드 =====")
    trainer = build_dpo_trainer(
        dpo_cfg=dpo_cfg,
        model_cfg=model_cfg,
        sft_adapter_path=sft_adapter_path,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


    print("-" * 60)
    print("===== DPO 학습 시작 =====")
    print("-" * 60)
    trainer.train()


    final_model_path = Path(dpo_cfg.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))


    print("-" * 60)
    print("===== DPO 학습 완료 =====")
    print(f"===== 모델 저장 경로: {final_model_path} =====")
    print("-" * 60)
    

def create_configs(cfg_dict: Dict[str, Any]):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model_dict = cfg_dict["model"].copy()
    if "compute_dtype" in model_dict and isinstance(model_dict["compute_dtype"], str):
        model_dict["compute_dtype"] = dtype_map.get(
            model_dict["compute_dtype"],
            torch.float16
        )

    model_cfg = ModelConfig(**model_dict)
    dpo_cfg = DPOTrainerConfig(**cfg_dict["dpo"]["trainer"])
    dpo_data_cfg = DPODataConfig(**cfg_dict["dpo"]["data"])
    sft_adapter_path = cfg_dict["dpo"]["sft_adapter_path"]

    return model_cfg, dpo_cfg, dpo_data_cfg, sft_adapter_path
    

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config_dpo.yaml",
        help="Path to DPO config YAML file"
    )
    args = parser.parse_args()

    raw_cfg = load_config(args.config)

    wandb_cfg = raw_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        setup_wandb(raw_cfg)

    model_cfg, dpo_cfg, dpo_data_cfg, sft_adapter_path = create_configs(raw_cfg)

    main(
        model_cfg=model_cfg,
        dpo_cfg=dpo_cfg,
        dpo_data_cfg=dpo_data_cfg,
        sft_adapter_path=sft_adapter_path,
    )

    if wandb_cfg.get("enabled", False):
        wandb.finish()
