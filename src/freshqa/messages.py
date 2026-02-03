from typing import Any, Dict, List, Mapping


Message = Dict[str, str]

def build_messages(
        system_prompt: str,
        user_prompt: str,
        system_kwargs: Dict[str, Any] | None = None,
        user_kwargs: Dict[str, Any] | None = None,
    ) -> List[Message]:
    system_kwargs = system_kwargs or {}
    user_kwargs = user_kwargs or {}

    system = system_prompt.format(**system_kwargs)
    user = user_prompt.format(**user_kwargs)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]