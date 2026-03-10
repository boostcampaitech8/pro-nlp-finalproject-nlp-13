import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def is_already_merged(save_path: str) -> bool:
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin"
    ]

    return not any(
        os.path.exists(os.path.join(save_path, f))
        for f in adapter_files
    )


def main(args):
    # ===== 이미 merge 되었는지 확인 =====
    if os.path.exists(args.save_to) and is_already_merged(args.save_to):
        print(f"⚠️ 이미 병합된 모델이 존재합니다 -> {args.save_to}")
        print("Merge 과정을 생략합니다.")
        return

    if 'sft' in args.lora_adapter_path:
        print("===== SFT LoRA Weights 병함 시작 =====")
    elif 'dpo' in args.lora_adapter_path:
        print("===== DPO LoRA Weights 병함 시작 =====")

    print("===== 베이스 모델 로드 중... =====")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    # ===== LoRA Adapter 존재 여부 확인 =====
    if not os.path.exists(args.lora_adapter_path):
        raise ValueError(f"LoRA adapter 경로가 존재하지 않습니다: {args.lora_adapter_path}")

    print("===== LoRA Adapter 로드 중... =====")
    model = PeftModel.from_pretrained(model, args.lora_adapter_path)

    print("===== LoRA 병합 진행 중... =====")
    merged_model = model.merge_and_unload()

    os.makedirs(args.save_to, exist_ok=True)

    print(f"===== 병합된 모델 저장 중 -> {args.save_to} =====")
    merged_model.save_pretrained(args.save_to, safe_serialization=True)

    # ===== Tokenizer 저장 =====
    print("===== 토크나이저 저장 중... =====")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.save_to)

    print("===== 모든 작업이 완료되었습니다! =====\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Base or SFT merged model path"
    )

    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="LoRA adapter path"
    )

    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="Path to save merged model"
    )

    args = parser.parse_args()
    main(args)