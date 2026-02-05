import re


def format_chat_template(example, tokenizer, max_length=512):
    cleaned_response = re.sub(
        r'<think>.*?</think>',
        '',
        example['response'],
        flags=re.DOTALL
    ).strip()
    
    # 연속 개행 정리
    cleaned_response = re.sub(r'\n+', '\n', cleaned_response).strip()
    
    # Messages 구성
    messages = [
        {
            "role": "system",
            "content": f"""당신은 'B'입니다. A가 다음 상황에 처해있을 때, 최대한 공감하며 일관적인 페르소나를 유지하세요.

[상황]
{example['situation']}

[대화 규칙]
- A: 대화 상대방 (User)
- B: 당신 (페르소나를 가진 사람)
- 페르소나를 항상 유지하며 응답하세요."""
        },
        {
            "role": "user",
            "content": example['context']
        },
        {
            "role": "assistant",
            "content": cleaned_response
        }
    ]
    
    # Chat template 적용 (모델 기본 template 사용)
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 토큰화
    tokenized = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    }