#!/bin/bash

# 모델 경로 및 이름 설정
MODEL_PATH="/data/ephemeral/pro-nlp-finalproject-nlp-13/outputs/merged_model/qwen_dpo_merged_model_final_jang_v1"
MODEL_NAME="dpo-final-policy"
TEMPLATE_PATH="${MODEL_PATH}/chat_template.jinja"

# vLLM 서버 실행
vllm serve "$MODEL_PATH" \
    --port 8080 \
    --served-model-name "$MODEL_NAME" \
    --chat-template "$TEMPLATE_PATH" \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 10000 \
    --enforce-eager