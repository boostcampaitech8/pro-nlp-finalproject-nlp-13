import re
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TokenizeConfig:
    """토큰화 설정"""
    max_length: int = 1024
    padding: bool = False
    truncation: bool = True
    prompt_path: str = "persona/prompt/prompt.txt"


class TokenizeWrapper:
    """Chat Template 적용 및 토큰화"""
    
    def __init__(self, tokenizer: Any, config: TokenizeConfig):
        self.tokenizer = tokenizer
        self.cfg = config
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Prompt 템플릿 로드"""
        if not os.path.exists(self.cfg.prompt_path):
            # 기본 템플릿
            return """당신은 'B'입니다. A가 다음 상황에 처해있을 때, 최대한 공감하며 일관적인 페르소나를 유지하세요.

[상황]
{situation}

[대화 규칙]
- A: 대화 상대방 (User)
- B: 당신 (페르소나를 가진 사람)
- 페르소나를 항상 유지하며 응답하세요."""
        
        with open(self.cfg.prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _clean_response(self, response: str) -> str:
        """<think> 태그 제거"""
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        cleaned = re.sub(r'\n+', '\n', cleaned).strip()
        return cleaned
    
    def build_messages(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dataset example → messages 변환
        
        Args:
            example: {'situation', 'context', 'response'}
        
        Returns:
            {'messages': [...]}
        """
        # Response 정제
        cleaned_response = self._clean_response(example['response'])
        
        # System content
        system_content = self.prompt_template.format(situation=example['situation'])
        
        # Messages 구성
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": example['context']},
            {"role": "assistant", "content": cleaned_response}
        ]
        
        return {"messages": messages}
    
    def to_text(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        messages → text 변환
        
        Args:
            example: {'messages': [...]}
        
        Returns:
            {'text': str}
        """
        text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}
    
    def tokenize_fn(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        text → token_ids 변환
        
        Args:
            example: {'text': str} or {'text': List[str]}
        
        Returns:
            {'input_ids', 'attention_mask'}
        """
        texts = example["text"]
        if isinstance(texts, str):
            texts = [texts]
        
        tok_kwargs = {
            "truncation": self.cfg.truncation,
            "padding": self.cfg.padding,
        }
        
        if self.cfg.truncation:
            tok_kwargs["max_length"] = self.cfg.max_length
        
        out = self.tokenizer(texts, **tok_kwargs)
        
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        }