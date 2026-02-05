from trl import DataCollatorForCompletionOnlyLM


# ============================================================
# Response Template Registry
# ============================================================

RESPONSE_TEMPLATES = {
    "exaone": {
        "assistant_only": "[|assistant|]",
        "skip_think": "</think>",
    },
}


# ============================================================
# Factory Function
# ============================================================

def get_data_collator(tokenizer, template_type="exaone", learning_mode="skip_think"):
    """
    DataCollator Factory
    
    Args:
        tokenizer: Tokenizer
        template_type: 모델 템플릿 (exaone, llama3, mistral)
        learning_mode: 학습 범위
            - assistant_only: assistant 전체 학습 (think 포함)
            - skip_think: think 이후만 학습 (think 제외)
    
    Returns:
        DataCollatorForCompletionOnlyLM
    """
    # Validation
    if template_type not in RESPONSE_TEMPLATES:
        raise ValueError(
            f"Unknown template_type: {template_type}. "
            f"Available: {list(RESPONSE_TEMPLATES.keys())}"
        )
    
    if learning_mode not in ["assistant_only", "skip_think"]:
        raise ValueError(
            f"Unknown learning_mode: {learning_mode}. "
            f"Available: ['assistant_only', 'skip_think']"
        )
    
    # Response template 선택
    response_template = RESPONSE_TEMPLATES[template_type][learning_mode]
    
    # DataCollator 생성 (HuggingFace)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    print(f"✅ DataCollator created")
    print(f"   - Template type: {template_type}")
    print(f"   - Learning mode: {learning_mode}")
    print(f"   - Response template: '{response_template}'")
    
    return data_collator