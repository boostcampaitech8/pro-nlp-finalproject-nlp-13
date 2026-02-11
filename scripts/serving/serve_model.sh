#!/bin/bash

# 모델 경로 및 이름 설정
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
SFT_ADAPTER_PATH="/data/ephemeral/pro-nlp-finalproject-nlp-13/outputs/sft/qwen3_4b_sft_final"
SFT_MERGED_PATH="/data/ephemeral/pro-nlp-finalproject-nlp-13/outputs/merged_model/qwen_sft_merged_model_final"

DPO_ADAPTER_PATH="/data/ephemeral/pro-nlp-finalproject-nlp-13/outputs/dpo/qwen3_4b_dpo_final/policy"
DPO_MERGED_PATH="/data/ephemeral/pro-nlp-finalproject-nlp-13/outputs/merged_model/qwen_dpo_merged_model_final"

MODEL_NAME="dpo-final-policy"
TEMPLATE_PATH="${DPO_MERGED_PATH}/chat_template.jinja"

# 모델 병합
python -m backend.merge_lora_weights --base_model_path $BASE_MODEL --lora_adapter_path $SFT_ADAPTER_PATH --save_to $SFT_MERGED_PATH
python -m backend.merge_lora_weights --base_model_path $SFT_MERGED_PATH --lora_adapter_path $DPO_ADAPTER_PATH --save_to $DPO_MERGED_PATH

# vLLM 서버 실행
vllm serve "$DPO_MERGED_PATH" \
    --port 8080 \
    --served-model-name "$DPO_MERGED_PATH" \
    --chat-template "$TEMPLATE_PATH" \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 10000 \
    --enforce-eager