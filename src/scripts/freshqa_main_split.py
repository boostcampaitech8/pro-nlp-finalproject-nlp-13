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
    parser.add_argument("--limit", type=int, default=0) 
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
    df[output_col] = df[output_col].astype(object)

    if os.path.exists(output_csv):
        print(f"기존 결과 파일 발견! ({output_csv}) -> 진행 상황을 복원합니다.")
        df_prev = pd.read_csv(output_csv)
        
        if output_col in df_prev.columns:
            df.loc[df_prev.index, output_col] = df_prev[output_col]

    mask_todo = df[output_col].isna() | (df[output_col] == "")
    target_indices = df[mask_todo].index
    
    total_todo = len(target_indices)
    print(f"전체: {len(df)}개 | 완료: {len(df) - total_todo}개 | 👉 남은 작업: {total_todo}개")

    if total_todo == 0:
        print("모든 작업이 완료되었습니다!")
        return

    processed_count = 0
    
    for i, row_idx in enumerate(tqdm(target_indices, desc="FreshQA Resume")):
        
        if args.limit > 0 and processed_count >= args.limit:
            print(f"\n설정된 제한({args.limit}개)에 도달하여 멈춥니다.")
            break

        question_text = str(df.at[row_idx, question_col])

        try:
            _, route_obj = route_question(
                question=question_text,
                today=today,
                system_prompt=SYSTEM_PROMPT_ROUTER,
                user_prompt=USER_PROMPT_ROUTER,
                llm_call=llm_call,
            )

            answer_type = route_obj.answer_type
            
            if answer_type in ("C", "D"):
                answer_text = route_obj.get("answer", "정보 없음")
                df.at[row_idx, output_col] = answer_text
            else:
                keyword_res = extract_keyword(
                    question=question_text,
                    today=today,
                    system_prompt=SYSTEM_PROMPT_KEYWORD,
                    user_prompt=USER_PROMPT_KEYWORD,
                    llm_call=llm_call,
                )
                
                if isinstance(keyword_res, tuple):
                    keyword_obj = keyword_res[1]
                else:
                    keyword_obj = keyword_res

                keyword_text = getattr(keyword_obj, "query", None) or question_text
                print(f"\n[DEBUG] 검색 키워드: {keyword_text}")
                wiki_result = wiki.get_wiki_text(keyword_text)
                wiki_text = wiki_result.text if hasattr(wiki_result, 'text') else ""
                
                answer_text = retrieve_answer(
                    question=question_text,
                    today=today,
                    wiki_text=wiki_text,
                    llm_call=llm_call,
                )
                df.at[row_idx, output_col] = answer_text

        except Exception as e:
            print(f"\n[Error at row {row_idx}] {e}")
            df.at[row_idx, output_col] = "Error"

        processed_count += 1

        if processed_count % 10 == 0:
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"\n저장 완료: {output_csv}")


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
    
    wiki_preview = wiki_text[:100] if wiki_text else "(Wiki 검색 실패)"
    print(f"\n[DEBUG] Wiki Content: {wiki_preview}...")
    print(f"[DEBUG] Raw LLM Response: {response_text[:200]}")

    try:
        response_json = safe_parse_json(response_text)
        print(f"[DEBUG] Parsed JSON: {response_json}")
        
        if isinstance(response_json, dict):
            answer = str(response_json.get("answer", "정보 없음")).strip() or "정보 없음"
            print(f"[DEBUG] Extracted answer: {answer}")
            return answer
        
        print(f"[DEBUG] JSON is not dict, type: {type(response_json)}")
        return "정보 없음"
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}")
        print(f"[DEBUG] Response text: {response_text}")
        return "정보 없음"


if __name__ == "__main__":
    main()