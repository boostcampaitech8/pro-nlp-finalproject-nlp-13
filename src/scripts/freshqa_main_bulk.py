from __future__ import annotations

import os
import argparse
import asyncio
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
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    today = get_today()
    load_dotenv()

    llm_cfg, wiki_cfg, input_cfg = load_config(args.config)

    llm_call = build_llm_call(llm_cfg)
    wiki = WikipediaRetriever(cfg=wiki_cfg)

    input_csv = input_cfg["input_csv"]
    output_csv = input_cfg["output_csv"]
    question_col = input_cfg.get("question_col", "question")
    output_col = input_cfg.get("output_col", "model_response")

    df = pd.read_csv(input_csv, index_col=False)
    
    if output_col not in df.columns:
        df[output_col] = ""

    # 이미 처리된 행 확인 (재시작 시 건너뛰기)
    processed_rows = set()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv, index_col=False)
        processed_rows = set(existing_df[existing_df[output_col].notna() & (existing_df[output_col] != "")].index)
        print(f"Found {len(processed_rows)} already processed rows. Skipping them.")
        df.update(existing_df)
    
    # 처리할 행 필터링
    rows_to_process = [i for i in df.index if i not in processed_rows]
    print(f"Start Processing {len(rows_to_process)} remaining questions with {args.workers} workers...")

    if not rows_to_process:
        print("All questions already processed!")
        return

    # asyncio로 병렬 실행 (50개씩 배치 저장)
    asyncio.run(process_all_questions(
        df=df,
        rows_to_process=rows_to_process,
        question_col=question_col,
        output_col=output_col,
        output_csv=output_csv,
        today=today,
        llm_call=llm_call,
        wiki=wiki,
        max_workers=args.workers,
        batch_size=50,
    ))

    print(f"\nCompleted! Final output: {output_csv}")


def process_single_question(
    row_idx: int,
    question_text: str,
    today: str,
    llm_call,
    wiki: WikipediaRetriever,
) -> Tuple[int, str]:
    """
    단일 질문 처리 (병렬 실행용)
    
    Returns:
        Tuple[int, str]: (row_idx, 답변 텍스트)
    """
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
            return row_idx, route_obj.answer or "정보 없음"
        
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
            return row_idx, "정보 없음"

        answer_text = retrieve_answer(
            question=question_text,
            today=today,
            wiki_text=wiki_result.text,
            llm_call=llm_call,
        )

        return row_idx, answer_text
        
    except Exception as e:
        print(f"\nError in row {row_idx}: {e}")
        return row_idx, "에러 발생"


async def process_all_questions(
    df: pd.DataFrame,
    rows_to_process: List[int],
    question_col: str,
    output_col: str,
    output_csv: str,
    today: str,
    llm_call,
    wiki: WikipediaRetriever,
    max_workers: int = 4,
    batch_size: int = 50,
):
    """
    비동기로 모든 질문 처리 (배치 단위로 저장)
    batch_size개씩 모았을 때 CSV에 한 번에 저장
    """
    semaphore = asyncio.Semaphore(max_workers)
    save_lock = asyncio.Lock()
    batch_queue = []
    
    async def process_with_semaphore(row_idx: int, question_text: str):
        async with semaphore:
            return await asyncio.to_thread(
                process_single_question,
                row_idx=row_idx,
                question_text=question_text,
                today=today,
                llm_call=llm_call,
                wiki=wiki,
            )
    
    async def save_batch():
        """배치에 모인 결과를 CSV에 저장"""
        nonlocal batch_queue
        if batch_queue:
            for row_idx, answer_text in batch_queue:
                df.at[row_idx, output_col] = answer_text
            df.to_csv(output_csv, index=False)
            print(f"  ✓ Saved {len(batch_queue)} results")
            batch_queue = []
    
    tasks = [
        process_with_semaphore(row_idx, str(df.at[row_idx, question_col]))
        for row_idx in rows_to_process
    ]
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="FreshQA"):
        try:
            row_idx, answer_text = await task
            async with save_lock:
                batch_queue.append((row_idx, answer_text))
                if len(batch_queue) >= batch_size:
                    await save_batch()
        except Exception as e:
            print(f"\nError: {e}")
    
    # 남은 결과 저장
    async with save_lock:
        await save_batch()


def load_config(path: str) -> Tuple[LLMConfig, WikiConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    llm_cfg = LLMConfig(**(data.get("llm", {}) or {}))
    wiki_cfg = WikiConfig(**(data.get("wiki", {}) or {}))
    io_cfg = data.get("io", {}) or {}

    return llm_cfg, wiki_cfg, io_cfg


def build_llm_call(cfg: LLMConfig):
    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )
    
    def llm_call(messages: List[Dict[str, str]]) -> str:
        try:
            return call_solar_pro2(client=client, messages=messages, cfg=cfg)
        except Exception as e:
            print(f"\n[API Request Error] {type(e).__name__}: {e}")
            raise
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