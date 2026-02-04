from __future__ import annotations

import os
import threading
import argparse
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def build_wiki_getter(cfg: WikiConfig):
    _local = threading.local()

    def get_wiki() -> WikipediaRetriever:
        if not hasattr(_local, "wiki"):
            _local.wiki = WikipediaRetriever(cfg=cfg)
        return _local.wiki

    return get_wiki


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug_time", action="store_true")  # ✅ 시간 로그 on/off
    parser.add_argument("--log_every", type=int, default=10)  # ✅ N개마다 요약 로그
    args = parser.parse_args()

    today = get_today()
    load_dotenv()

    llm_cfg, wiki_cfg, input_cfg = load_config(args.config)
    llm_call = build_llm_call(llm_cfg)
    get_wiki = build_wiki_getter(wiki_cfg)

    input_csv = input_cfg["input_csv"]
    output_csv = input_cfg["output_csv"]
    question_col = input_cfg.get("question_col", "question")
    output_col = input_cfg.get("output_col", "model_response")

    df = pd.read_csv(input_csv, index_col=False)
    if output_col not in df.columns:
        df[output_col] = ""

    # 이미 처리된 행 확인
    processed_rows = set()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv, index_col=False)
        processed_rows = set(existing_df[existing_df[output_col].notna() & (existing_df[output_col] != "")].index)
        print(f"Found {len(processed_rows)} already processed rows. Skipping them.")
        df.update(existing_df)

    rows_to_process = [i for i in df.index if i not in processed_rows]
    print(f"Start Processing {len(rows_to_process)} remaining questions with {args.workers} workers...")

    if not rows_to_process:
        print("All questions already processed!")
        return

    t0_all = time.perf_counter()

    # ✅ 에러 로그 파일 경로
    error_log_path = output_csv.replace(".csv", "_errors.log")
    
    stats = process_questions_parallel(
        df=df,
        rows_to_process=rows_to_process,
        question_col=question_col,
        output_col=output_col,
        output_csv=output_csv,
        error_log_path=error_log_path,
        today=today,
        llm_call=llm_call,
        get_wiki=get_wiki,
        max_workers=args.workers,
        batch_size=50,
        debug_time=args.debug_time,
        log_every=args.log_every,
        t0_all=t0_all,
    )

    total_sec = time.perf_counter() - t0_all
    print(f"\n{'='*60}")
    print(f"✅ Completed! Final output: {output_csv}")
    print(f"📊 Summary:")
    print(f"   - Total: {stats['total']} questions")
    print(f"   - Success: {stats['success']}")
    if stats['errors'] > 0:
        print(f"   - Errors: {stats['errors']} ⚠️ (see {error_log_path})")
    else:
        print(f"   - Errors: {stats['errors']}")
    print(f"⏱️  Total elapsed: {total_sec:.2f}s ({total_sec/60:.2f} min)")
    print(f"{'='*60}")


def process_single_question(
    row_idx: int,
    question_text: str,
    today: str,
    llm_call,
    get_wiki,
    debug_time: bool = False,
) -> Tuple[int, str]:
    """
    단일 질문 처리 (병렬 실행용)
    Returns:
        Tuple[int, str]: (row_idx, 답변 텍스트)
    """
    thread_name = threading.current_thread().name
    t0 = time.perf_counter()

    try:
        # 1) route
        t_route0 = time.perf_counter()
        _, route_obj = route_question(
            question=question_text,
            today=today,
            system_prompt=SYSTEM_PROMPT_ROUTER,
            user_prompt=USER_PROMPT_ROUTER,
            llm_call=llm_call,
        )
        t_route = time.perf_counter() - t_route0

        answer_type = route_obj.answer_type
        if answer_type in ("C", "D"):
            if debug_time:
                total = time.perf_counter() - t0
                print(f"[{thread_name}] row={row_idx} route={t_route:.2f}s (C/D early return) total={total:.2f}s")
            return row_idx, route_obj.answer or "정보 없음"

        # 2) keyword
        t_kw0 = time.perf_counter()
        keyword_obj = extract_keyword(
            question=question_text,
            today=today,
            system_prompt=SYSTEM_PROMPT_KEYWORD,
            user_prompt=USER_PROMPT_KEYWORD,
            llm_call=llm_call,
        )
        t_kw = time.perf_counter() - t_kw0

        keyword_text = getattr(keyword_obj, "query", None) or question_text

        # 3) wiki
        t_wiki0 = time.perf_counter()
        wiki = get_wiki()
        wiki_result = wiki.get_wiki_text(keyword_text)
        t_wiki = time.perf_counter() - t_wiki0

        if not wiki_result:
            if debug_time:
                total = time.perf_counter() - t0
                print(f"[{thread_name}] row={row_idx} route={t_route:.2f}s kw={t_kw:.2f}s wiki={t_wiki:.2f}s (no wiki) total={total:.2f}s")
            return row_idx, "정보 없음"

        # 4) retrieve
        t_ret0 = time.perf_counter()
        answer_text = retrieve_answer(
            question=question_text,
            today=today,
            wiki_text=wiki_result.text,
            llm_call=llm_call,
        )
        t_ret = time.perf_counter() - t_ret0

        if debug_time:
            total = time.perf_counter() - t0
            print(
                f"[{thread_name}] row={row_idx} "
                f"route={t_route:.2f}s kw={t_kw:.2f}s wiki={t_wiki:.2f}s ret={t_ret:.2f}s total={total:.2f}s"
            )

        return row_idx, answer_text

    except Exception as e:
        # ✅ 어떤 스레드/어느 row에서 터졌는지 출력
        total = time.perf_counter() - t0
        err_msg = f"\n[ERROR] row={row_idx} thread={thread_name} elapsed={total:.2f}s err={type(e).__name__}: {e}"
        print(err_msg)
        import sys
        sys.stderr.flush()  # ✅ 강제로 에러 메시지 출력
        return row_idx, "에러 발생"


def process_questions_parallel(
    df: pd.DataFrame,
    rows_to_process: List[int],
    question_col: str,
    output_col: str,
    output_csv: str,
    error_log_path: str,
    today: str,
    llm_call,
    get_wiki,
    max_workers: int = 4,
    batch_size: int = 50,
    debug_time: bool = False,
    log_every: int = 20,
    t0_all: float | None = None,
):
    """✅ 에러 로깅 및 통계 추가"""
    batch_queue = []
    done = 0
    total = len(rows_to_process)
    error_count = 0
    error_log = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_question,
                row_idx=row_idx,
                question_text=str(df.at[row_idx, question_col]),
                today=today,
                llm_call=llm_call,
                get_wiki=get_wiki,
                debug_time=debug_time,
            ): row_idx
            for row_idx in rows_to_process
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="FreshQA"):
            try:
                row_idx, answer_text = future.result()
                batch_queue.append((row_idx, answer_text))
                done += 1
                
                # ✅ 에러 여부 판단 (에러는 "에러 발생" 텍스트)
                if answer_text == "에러 발생":
                    error_count += 1
                    error_log.append(f"row {row_idx}: Process failed")

                # ✅ N개마다 평균 처리시간 로그
                if t0_all is not None and (done % log_every == 0):
                    elapsed = time.perf_counter() - t0_all
                    avg = elapsed / done
                    remain = total - done
                    eta = avg * remain
                    print(f"  [PROGRESS] done={done}/{total} (errors={error_count}) avg={avg:.2f}s ETA={eta/60:.1f}min")

                if len(batch_queue) >= batch_size:
                    for idx, text in batch_queue:
                        df.at[idx, output_col] = text
                    df.to_csv(output_csv, index=False)
                    print(f"  ✓ Saved {len(batch_queue)} results")
                    batch_queue = []

            except Exception as e:
                # ✅ future.result 자체가 터질 경우
                done += 1
                error_count += 1
                err_msg = f"\n[ERROR] future err={type(e).__name__}: {e}"
                print(err_msg)
                error_log.append(err_msg)

    if batch_queue:
        for idx, text in batch_queue:
            df.at[idx, output_col] = text
        df.to_csv(output_csv, index=False)
        print(f"  ✓ Saved {len(batch_queue)} results")
    
    # ✅ 에러 로그 파일에 저장
    if error_log:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Error Report - {total} questions processed\n")
            f.write(f"{'='*60}\n\n")
            for log_entry in error_log:
                f.write(log_entry + "\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"Total Errors: {error_count}\n")
            f.write(f"Success Rate: {((total-error_count)/total*100):.1f}%\n")
    
    return {
        "total": total,
        "success": total - error_count,
        "errors": error_count,
    }


def load_config(path: str) -> Tuple[LLMConfig, WikiConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    llm_cfg = LLMConfig(**(data.get("llm", {}) or {}))
    wiki_cfg = WikiConfig(**(data.get("wiki", {}) or {}))
    io_cfg = data.get("io", {}) or {}

    return llm_cfg, wiki_cfg, io_cfg


def build_llm_call(cfg):
    _local = threading.local()

    def _get_client() -> OpenAI:
        if not hasattr(_local, "client"):
            _local.client = OpenAI(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                base_url="https://api.upstage.ai/v1"
            )
        return _local.client

    def llm_call(messages: List[Dict[str, str]]) -> str:
        client = _get_client()
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