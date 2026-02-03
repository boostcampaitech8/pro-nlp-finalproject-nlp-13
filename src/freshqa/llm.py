from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

Message = Dict[str, str]


@dataclass(frozen=True)
class LLMConfig:
    model: str = "solar-pro2"
    temperature: float = 0.0
    top_p: float = 1.0
    reasoning_effort: str = "high"


def call_solar_pro2(
    client: Any,
    messages: List[Message],
    cfg: LLMConfig = LLMConfig(),
) -> str:
    try:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            stream=False,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            reasoning_effort=cfg.reasoning_effort,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n[LLM API Error] {type(e).__name__}: {e}")
        raise