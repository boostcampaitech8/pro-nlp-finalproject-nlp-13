import os
import re
import json
import torch
from pathlib import Path
from dotenv import load_dotenv 
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()
API_KEY = os.getenv('UPSTAGE_API_KEY')

SNS_CONVERSATION_JUDGE_PROMPT = """당신에게 '사용자 발화(user_message)'와 'AI 응답(ai_response)'가 주어집니다.  
당신의 임무는 AI 응답이 SNS 대화 상황에서 얼마나 적절한지를 평가하는 것입니다.

AI 응답은 실제 사람이 SNS에서 대화하는 것처럼 자연스럽고, 맥락에 맞으며, 대화를 이어가도록 유도하고, 과도하게 길지 않아야 합니다.

평가는 아래 4가지 평가 기준을 기반으로 각각 수행해야 합니다.

[평가 기준]
1. 자연스러움 (naturalness) : AI 응답이 실제 SNS 사용자처럼 자연스러운 대화체를 사용하는지 평가합니다.
    - 기계적이고 형식적인 문체가 아닌 구어체 표현 사용 여부
    - SNS 대화에서 흔히 사용되는 말투, 어휘, 줄임말, 감정 표현 사용 여부 (예: "ㅋㅋ", "ㅠㅠ", "ㄹㅇ", "헐", "맞아", "진짜?", "그치", "렬루" 등)
    - SNS 환경에서는 비표준 표현이나 신조어가 사용되더라도 실제 사용자들이 사용하는 표현이라면 자연스럽다고 판단할 수 있습니다.

2. 맥락 적합성 (contextual_relevance) : AI 응답이 사용자 발화를 정확히 이해하고 자연스럽게 이어지는지 평가합니다.
    - 사용자 감정, 상황, 의도를 적절히 반영했는지
    - 이전 발화와 논리적으로 연결되는지
    - 맥락과 무관한 일반적인 조언이나 정보 제공이 포함되지 않았는지

3. 참여 유도 (engagement) : AI 응답이 대화를 지속하려는 의도를 보이는지 평가합니다.
    - 공감 표현 또는 감정 반응 포함 여부
    - 자연스러운 질문 또는 반응을 통해 대화를 이어가려는 시도
    - 상대방이 추가로 말하고 싶게 만드는 요소 존재 여부
    - 일방적 정보 전달 형태인지 여부

4. 간결성 (conciseness) : AI 응답이 SNS 대화 특성에 맞게 짧고 핵심적인 형태로 작성되었는지 평가합니다.
    - 응답 길이가 불필요하게 길지 않은지
    - 한 번의 응답에 과도한 정보, 조언, 설명이 포함되지 않았는지
    - SNS 대화에서 일반적으로 사용되는 짧은 문장 구조를 따르는지

핵심 메시지를 간단하고 직관적으로 전달하는지
주의: 설명조, 보고서체, 비즈니스 이메일 말투는 낮은 점수를 부여하십시오.



[점수 척도]  
각 항목은 1점에서 5점 사이의 정수로 평가하십시오.

1점: 매우 부족함 – 기준을 거의 충족하지 못함  
2점: 부족함 – 일부 요소만 충족함  
3점: 보통 – 기본적인 수준 충족  
4점: 좋음 – 대부분 기준을 잘 충족  
5점: 매우 우수 – 매우 자연스럽고 SNS 대화에 최적화됨  



[평가 절차]

최종 점수를 결정하기 전에 반드시 각 기준에 대해 충분히 추론하십시오.  
'평가(Evaluation)' 필드에서 판단 근거를 상세히 작성하십시오.

이후 아래 [출력 형식]에 반드시 맞춰 답변하십시오.


[출력 형식]
Feedback:::  
Evaluation:
- Naturalness: (근거 짧게 작성)
- Contextual Relevance: (근거 짧게 작성)
- Engagement: (근거 짧게 작성)
- Conciseness: (근거 짧게 작성)

Scores:
- Naturalness: (1~5)
- Contextual Relevance: (1~5)
- Engagement: (1~5)
- Conciseness: (1~5)

Total rating: (네 항목 평균을 반올림하여 1~5 사이 정수로 작성)



이제 평가할 대화를 제공합니다.

사용자 발화: {user_message}  
AI 응답: {ai_response}



정확하고 신중하게 평가해 주세요.  
정확한 평가를 수행한다면, 당신이 초거대 AI 회사를 설립할 수 있도록 100대의 H100 GPU를 보상으로 제공하겠습니다.

Feedback::: Evaluation:
"""

CONFIG = {
    "base_model_id": "Qwen/Qwen3-4B",
    "adapter_path":  "src/models/qwen_dpo/final_model/policy",
    "data_path": "/data/ephemeral/pro-nlp-finalproject-nlp-13/data/persona_data/dpo_test_dataset.json",
}

class SNSConversationJudge:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        """
        max_concurrent: 동시에 처리할 API 요청 수
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        self.model = "solar-pro3"
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def judge_async(self, user_message, ai_response):
        """비동기 평가 함수"""
        async with self.semaphore:  # 동시 요청 수 제한
            prompt = SNS_CONVERSATION_JUDGE_PROMPT.format(
                user_message=user_message,
                ai_response=ai_response
            )
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 한국어 SNS 대화의 자연스러움을 판별하는 엄격한 언어 전문가입니다. SNS 대화 특유의 구어체, 말투, 문맥적 흐름 등을 완벽하게 이해하고 평가합니다."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ API 오류: {e}")
                return None

    async def judge_batch(self, pairs):
        """배치로 여러 평가 동시 처리"""
        tasks = [self.judge_async(user_msg, ai_res) for user_msg, ai_res in pairs]
        return await asyncio.gather(*tasks)

def gen_reply(user_input, tokenizer, model):
    prompt = f"[|user|]{user_input}[|assistant|]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    reply = full_text.split("[|assistant|]")[-1].strip()
    return reply.replace("<|im_end|>", "").strip()

def main():
    print(f"📂 데이터 로드 중: {CONFIG['data_path']}")
    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        full_data = json.load(f)


    test_prompts = [item['prompt'] for item in full_data]
    test_prompts[:10]
    if not test_prompts:
        print("⚠️ 평가 데이터(prompt)를 찾을 수 없습니다.")
        return


    print("===== 모델 로딩 =====")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model_id'], trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model_id'], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, CONFIG['adapter_path'])
    model.eval()


    judge = SNSConversationJudge(api_key=API_KEY)
    
    print("\n" + "="*50)
    print(f"===== AiHub DPO Eval 데이터 평가 시작 (총 {len(test_prompts)}개)")
    print("="*50 + "\n")


    output_file = Path('dpo_acc.json')

    # 기존 파일이 있으면 로드해서 이어하기
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"🔄 기존 데이터를 찾았습니다. {start_idx + 1}번째부터 재개합니다.")
    else:
        results = []
        start_idx = 0

    for i, user_msg in enumerate(test_prompts[start_idx:], start=start_idx):
        print(f"[{i+1}/{len(test_prompts)}] 평가 진행 중...")
        
        # 모델 답변 생성
        generated_res = gen_reply(user_msg, tokenizer, model)
        
        # Solar 평가
    #     evaluation_result = judge.judge(user_msg, generated_res)
        
    #     # 결과 출력
    #     # print(f"\n💬 User: {user_msg}")
    #     # print(f"🤖 DPO Model: {generated_res}")
    #     # print(f"📝 {evaluation_result}")
    #     # print("-" * 40)

    #     patterns = {
    #         "naturalness": r"Naturalness.*:\s*(\d)",
    #         "contextual_relevance": r"Contextual Relevance.*:\s*(\d)",
    #         "engagement": r"Engagement.*:\s*(\d)",
    #         "conciseness": r"Conciseness.*:\s*(\d)",
    #         "total_rating": r"Total rating.*:\s*(\d)"
    #     }
    
    #     current_result = {
    #         "id": i + 1,
    #         "prompt": user_msg,
    #         "response": generated_res,
    #         "raw_evaluation": evaluation_result, 
    #         "scores": {}
    #     }
        
    #     for key, pattern in patterns.items():
    #         match = re.search(pattern, evaluation_result)
    #         if match:
    #             current_result["scores"][key] = int(match.group(1))

    #     results.append(current_result)

    #     with open(output_file, "w", encoding="utf-8") as f:
    #         json.dump(results, f, indent=4, ensure_ascii=False)

    # dpo_metrics = calculate_metrics_100(results) # 현재 for문으로 모은 데이터

    # print("=== 모델 성능 평가 결과 (100점 만점) ===")
    # print(f"{'Metric':<25} | {'Score':<10}")
    # print("-" * 40)
    # for metric, score in dpo_metrics.items():
    #     print(f"{metric:<25} | {score:>10.2f}")
    

def calculate_metrics_100(results_list):
    # 계산할 지표 설정
    metrics = ["naturalness", "contextual_relevance", "engagement", "conciseness", "total_rating"]
    
    # 누적 점수와 개수를 담을 딕셔너리
    sum_scores = {m: 0 for m in metrics}
    count_scores = {m: 0 for m in metrics}
    
    for res in results_list:
        scores = res.get("scores", {})
        for m in metrics:
            if m in scores:
                # 5점 만점 데이터를 20을 곱해 100점 만점으로 변환
                sum_scores[m] += scores[m] * 20
                count_scores[m] += 1
                
    # 평균 계산
    avg_reports = {}
    for m in metrics:
        avg_reports[m] = sum_scores[m] / count_scores[m] if count_scores[m] > 0 else 0
        
    return avg_reports


if __name__ == "__main__":
    main()
