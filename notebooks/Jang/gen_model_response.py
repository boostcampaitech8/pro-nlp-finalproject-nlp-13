import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# 사용할 모델 목록(수정 가능)
AVAILABLE_MODELS = {
    # OpenAI
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
    },

    # Upstage – Solar Pro 2
    "solar-pro2": {
        "provider": "upstage",
        "model": "solar-pro2",
        "base_url": "https://api.upstage.ai/v1",
    },

    # Anthropic – Claude 계열
    "claude-3-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-sonnet",
        "base_url": "https://api.anthropic.com/v1",
    },

    # Google Gemini
    "gemini-pro": {
        "provider": "gemini",
        "model": "gemini-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },

    # Mistral
    "mistral-large": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "base_url": "https://api.mistral.ai/v1",
    },

    # Meta Llama
    "llama3-70b": {
        "provider": "llama",
        "model": "llama3-70b",
        "base_url": "https://api.llama-api.com/v1",
    },
}

# 사용할 모델 선정
SELECTED_MODEL = ""

# 모델별 API Key 설정
OPENAI_API_KEY = ""
UPSTAGE_API_KEY = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""
MISTRAL_API_KEY = ""
LLAMA_API_KEY = ""

# 대상 파일 및 경로 설정
INPUT_CSV = "ko-freshqa_2025_test.csv"
OUTPUT_CSV = "ko-freshqa_2025_test_with_model_respense.csv"

QUESTION_COL = "question"
MODEL_RESPONSE_COL = "model_response"

# 병렬 요청 스레드
MAX_WORKERS = 10

# 날짜 지정
TODAY = datetime.now().strftime("%Y-%m-%d")

# 클라이언트 생성
def get_client(model_key: str):
    cfg = AVAILABLE_MODELS[model_key]
    provider = cfg["provider"]

    if provider == "openai":
        return OpenAI(api_key=OPENAI_API_KEY, base_url=cfg["base_url"])

    elif provider == "upstage":
        return OpenAI(api_key=UPSTAGE_API_KEY, base_url=cfg["base_url"])

    elif provider == "anthropic":
        return OpenAI(api_key=ANTHROPIC_API_KEY, base_url=cfg["base_url"])

    elif provider == "gemini":
        return OpenAI(api_key=GEMINI_API_KEY, base_url=cfg["base_url"])

    elif provider == "mistral":
        return OpenAI(api_key=MISTRAL_API_KEY, base_url=cfg["base_url"])

    elif provider == "llama":
        return OpenAI(api_key=LLAMA_API_KEY, base_url=cfg["base_url"])

    else:
        raise ValueError(f"Unknown provider: {provider}")


client = get_client(SELECTED_MODEL)

# 모델별 포맷 차이 자동 처리
def build_messages(question: str, provider: str):
    """
    provider에 따라 메시지 구조를 자동 생성
    """
    system_prompt = (
        "너는 질의에 대해 한국어로만 답변하는 어시스턴트다."
        "가능한 한 간결한 문장으로 답하라."
        f"오늘 날짜는 {TODAY}이다. 오늘 날짜 기준으로 질문에 대해 답변하라."
    )

    if provider in ("openai", "upstage", "mistral", "llama"):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    if provider == "anthropic":
        # Claude는 messages 대신 "system"+"messages" 조합 필요
        return {
            "system": system_prompt,
            "messages": [{"role": "user", "content": question}],
        }

    if provider == "gemini":
        # Gemini는 Google 포맷: contents = [{ "role": "user", "parts": ["text"] }]
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": system_prompt + "\n" + question}]
                }
            ]
        }

    raise ValueError("Unknown provider formatting")


# 모델별 호출 로직
def ask_model_short_answer(question: str) -> str:
    cfg = AVAILABLE_MODELS[SELECTED_MODEL]
    provider = cfg["provider"]
    model_name = cfg["model"]

    try:
        msg = build_messages(question, provider)

        # ============ OpenAI / Upstage / Mistral / Llama ============
        if provider in ("openai", "upstage", "mistral", "llama"):
            res = client.chat.completions.create(
                model=model_name,
                messages=msg,
                stream=False,
            )
            return res.choices[0].message.content.strip()

        # ===================== Anthropic =====================
        elif provider == "anthropic":
            res = client.messages.create(
                model=model_name,
                system=msg["system"],
                messages=msg["messages"],
                max_tokens=256,
            )
            return res.content[0].text.strip()

        # ===================== Gemini ========================
        elif provider == "gemini":
            res = client.models.generateContent(
                model=model_name,
                **msg,
            )
            return res.candidates[0].content.parts[0].text.strip()

        else:
            return "[ERROR] Unknown provider"

    except Exception as e:
        return f"[ERROR] {e}"

# 병렬 처리 + CSV 저장
def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"총 {total}개 질문 처리 시작... (모델: {SELECTED_MODEL})\n")

    results = [None] * total

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(ask_model_short_answer, rows[i][QUESTION_COL]): i
            for i in range(total)
        }

        with tqdm(total=total, desc="Model answering", ncols=90) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR] {e}"
                pbar.update(1)

    fieldnames = list(rows[0].keys())
    if MODEL_RESPONSE_COL not in fieldnames:
        fieldnames.append(MODEL_RESPONSE_COL)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row, answer in zip(rows, results):
            row[MODEL_RESPONSE_COL] = answer
            writer.writerow(row)

    print(f"\n✅ 완료! 결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()