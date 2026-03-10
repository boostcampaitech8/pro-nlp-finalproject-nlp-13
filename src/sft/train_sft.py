from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.common.set_seed import set_seed
from src.common.wandb import setup_wandb

from src.sft.data.data_loader import load_raw_dataset
from src.sft.data.preprocessor import parse_and_add_system_prompt
from src.sft.data.collator import get_collator_and_format_func

from src.sft.training.model_loader import ModelConfig, TrainConfig, load_model_and_tokenizer
from src.sft.training.trainer import CustomLoraConfig, SftTrainConfig, build_sft_trainer


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="e.g. config/sft_train.yaml (config 폴더는 src와 같은 레벨)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    wandb = setup_wandb(cfg)

    data_cfg = cfg.get("data", {})
    data_path = data_cfg["path"]
    test_size = float(data_cfg.get("test_size", 0.1))
    system_content = data_cfg.get("system_content", "")

    dataset = load_raw_dataset(data_path=data_path, test_size=test_size, seed=seed)
    if system_content:
        dataset = parse_and_add_system_prompt(dataset, system_content=system_content)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    model_cfg = ModelConfig(**cfg["model"])  
    train_load_cfg = TrainConfig(**cfg.get("model_load", {})) 

    model, tokenizer = load_model_and_tokenizer(
        model_cfg=model_cfg,
        train_cfg=train_load_cfg,
    )

    collator, formatting_func = get_collator_and_format_func(tokenizer)

    sft_cfg = SftTrainConfig(**cfg.get("train", {}))
    lora_cfg = CustomLoraConfig(**cfg.get("lora", {}))

    out_dir = ensure_dir(sft_cfg.output_dir)
    (out_dir / "used_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        sft_cfg=sft_cfg,
        lora_cfg=lora_cfg,
        collator=collator,
        formatting_func=formatting_func,
    )

    resume_ckpt = cfg.get("train", {}).get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if wandb is not None:
        wandb.finish()

    print(f"[Training done] Saved to: {out_dir}")


if __name__ == "__main__":
    main()