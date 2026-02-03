from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from src.freshqa.llm import LLMConfig, call_solar_pro2
from src.freshqa.router import route_question
from src.freshqa.keyword import extract_keyword
from src.freshqa.retrieve import WikipediaRetriever, WikiConfig
from src.freshqa.messages import build_messages
from src.freshqa.utils import safe_parse_json, get_today
from src.freshqa.prompts import (
    SYSTEM_PROMPT_ROUTER, USER_PROMPT_ROUTER,
    SYSTEM_PROMPT_KEYWORD, USER_PROMPT_KEYWORD,
    SYSTEM_PROMPT_RETRIEVE, USER_PROMPT_RETRIEVE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    today = get_today()
    load_dotenv()

    llm_cfg, wiki_cfg, input_cfg = load_config(args.config)

    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )

    llm_call = build_llm_call(client, llm_cfg)
    wiki = WikipediaRetriever(cfg=wiki_cfg)

    input_csv = input_cfg["input_csv"]
    output_csv = input_cfg["output_csv"]
    question_col = input_cfg.get("question_col", "question")
    output_col = input_cfg.get("output_col", "model_response")

    df = pd.read_csv(input_csv, index_col=False)
    
    if output_col not in df.columns:
        df[output_col] = ""

    print(f"Start Processing {len(df)} questions...")

    for row_idx in tqdm(df.index, desc="FreshQA"):
        question_text = str(df.at[row_idx, question_col])

        _, route_obj = route_question(
            question=question_text,
            today=today,
            system_prompt=SYSTEM_PROMPT_ROUTER,
            user_prompt=USER_PROMPT_ROUTER,
            llm_call=llm_call,
        )

        answer_type = route_obj.answer_type
        
        # C,D
        if answer_type in ("C", "D"):
            answer_text = route_obj.answer or "정보 없음"
            df.at[row_idx, output_col] = answer_text
            continue
        
        # A,B
        keyword_obj = extract_keyword(
            question=question_text,
            today=today,
            system_prompt=SYSTEM_PROMPT_KEYWORD,
            user_prompt=USER_PROMPT_KEYWORD,
            llm_call=llm_call,
        )
        
        keyword_text = getattr(keyword_obj, "query", None) or question_text
        
        wiki_result = wiki.get_wiki_text(keyword_text)
        if not wiki_result:
            df.at[row_idx, output_col] = "정보 없음"
            continue

        answer_text = retrieve_answer(
            question=question_text,
            today=today,
            wiki_text=wiki_result.text,
            llm_call=llm_call,
        )

        df.at[row_idx, output_col] = answer_text

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def load_config(path: str) -> Tuple[LLMConfig, WikiConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    llm_cfg = LLMConfig(**(data.get("llm", {}) or {}))
    wiki_cfg = WikiConfig(**(data.get("wiki", {}) or {}))
    io_cfg = data.get("io", {}) or {}

    return llm_cfg, wiki_cfg, io_cfg


def build_llm_call(client: OpenAI, cfg: LLMConfig):
    def llm_call(messages: List[Dict[str, str]]) -> str:
        return call_solar_pro2(client=client, messages=messages, cfg=cfg)
    return llm_call


def retrieve_answer(
        question: str,
        today: str,
        wiki_text: str,
        llm_call,
) -> str:
    messages = build_messages(
        system_prompt=SYSTEM_PROMPT_RETRIEVE,
        user_prompt=USER_PROMPT_RETRIEVE,
        system_kwargs={"TODAY": today},
        user_kwargs={"QUESTION": question, "CONTENT": wiki_text},
    )
    response_text = llm_call(messages).strip()

    try:
        response_json = safe_parse_json(response_text)
        if isinstance(response_json, dict):
            return str(response_json.get("answer", "정보 없음")).strip() or "정보 없음"
        return "정보 없음"
    except Exception:
        return "정보 없음"


if __name__ == "__main__":
    main()