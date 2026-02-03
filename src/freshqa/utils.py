import json
from datetime import datetime
from typing import Any, Dict


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start_candidates = [p for p in (text.find("{"), text.find("[")) if p != -1]
    if not start_candidates:
        raise json.JSONDecodeError("No JSON found", text, 0)

    s = text[min(start_candidates):].lstrip()

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(s)
    return obj

def get_today():
    return datetime.now().strftime("%Y-%m-%d")
