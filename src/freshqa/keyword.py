from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from src.freshqa.messages import build_messages
from src.freshqa.utils import safe_parse_json

Message = Dict[str, str]


@dataclass(frozen=True)
class KeywordResult:
    query: str


def extract_keyword(
    question: str,
    today: str,
    system_prompt: str,
    user_prompt: str,
    llm_call: Callable[[list[Message]], str],
) -> KeywordResult:
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        system_kwargs={"TODAY": today},
        user_kwargs={"QUESTION": question},
    )
    raw = ""
    try:
        raw = llm_call(messages).strip()
        parsed = safe_parse_json(raw)
    except Exception:
        return KeywordResult(query=question)
    query = ""
    try:
        if isinstance(parsed, dict):
            query = (parsed.get("query") or "").strip()
    except Exception:
        query = ""

    if not query:
        query = question

    return KeywordResult(query=query)