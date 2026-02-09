#!/bin/bash

python -m src.RAG.metric.generate_test_data \
    --data_path "data/persona_data/AiHub_SNS.csv" \
    --train_output_path "data/persona_data/dpo_train_dataset.json" \
    --train_output_path "data/persona_data/dpo_test_dataset.json" \
    --sample_size 100 \
    --batch_size 5