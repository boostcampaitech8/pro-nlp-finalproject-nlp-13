from trl import DataCollatorForCompletionOnlyLM


# Response Template Registry
RESPONSE_TEMPLATES = {
    "exaone": {
        "assistant_only": "[|assistant|]",
        "skip_think": "</think>",
    },
    "llama3": {
        "assistant_only": "<|start_header_id|>assistant<|end_header_id|>",
        "skip_think": "</think>",
    },
    "mistral": {
        "assistant_only": "[/INST]",
        "skip_think": "</think>",
    },
}


def get_data_collator(tokenizer, template_type="exaone", learning_mode="skip_think"):
    """
    DataCollator Factory
    
    Args:
        tokenizer: Tokenizer
        template_type: exaone, llama3, mistral
        learning_mode: assistant_only, skip_think
    
    Returns:
        DataCollatorForCompletionOnlyLM
    """
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
    
    response_template = RESPONSE_TEMPLATES[template_type][learning_mode]
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    print(f"✅ DataCollator created")
    print(f"   - Template: {template_type}")
    print(f"   - Mode: {learning_mode}")
    print(f"   - Response template: '{response_template}'")
    
    return data_collator