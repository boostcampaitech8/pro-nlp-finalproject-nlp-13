from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.freshqa.utils import safe_parse_json
from src.freshqa.messages import build_messages

@dataclass(frozen=True)
class RouteResult:
    answer_type: str
    reasoning: str

def route_question(
        question: str,
        today: str,
        system_prompt: str,
        user_prompt: str,
        llm_call: Callable[[List[Dict[str, str]]], str],
) -> Tuple[str, RouteResult]:
    """
    question을 입력으로 받아서 질문을 분류
    A: 확실한 지식 / 계산 가능
    B: 확인 필요: 최신/변동/정밀 데이터
    C: 전제 오류
    D: 미래 예측
    """
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        system_kwargs={"TODAY": today},
        user_kwargs={"QUESTION": question},
    )

    response_text = llm_call(messages).strip()
    response_json = safe_parse_json(response_text)

    answer_type = response_json.get("answer_type")
    reasoning = response_json.get("reasoning", "")

    return response_text, RouteResult(answer_type=answer_type, reasoning=reasoning)