import os
import json
import torch
from dotenv import load_dotenv 
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()
API_KEY = os.getenv('UPSTAGE_API_KEY')

SNS_CONVERSATION_JUDGE_PROMPT = """당신에게 '사용자 발화(user_message)'와 'AI 응답(ai_response)'가 주어집니다.  
당신의 임무는 AI 응답이 SNS 대화 상황에서 얼마나 적절한지를 평가하는 것입니다.

AI 응답은 실제 사람이 SNS에서 대화하는 것처럼 자연스럽고, 맥락에 맞으며, 대화를 이어가도록 유도하고, 과도하게 길지 않아야 합니다.

평가는 아래 4가지 기준을 기반으로 각각 수행해야 합니다.

[평가 기준]

1. 자연스러움 (naturalness)  
- AI 특유의 기계적이고 형식적인 느낌이 제거되었는지 평가합니다.

2. 맥락 적합성 (contextual_relevance)  
- 이전 사용자 발화를 이해하고 그 흐름을 자연스럽게 이어가는지 평가합니다.

3. 참여 유도 (engagement)  
- 공감 표현, 반응, 질문 등으로 대화를 지속하려는 요소가 있는지 평가합니다.

4. 간결성 (conciseness)  
- SNS 대화 특성에 맞는 적절한 길이인지 평가합니다.

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

이후 아래 형식에 맞춰 답변하십시오.



Feedback:::  
Evaluation:
- Naturalness: (근거 작성)
- Contextual Relevance: (근거 작성)
- Engagement: (근거 작성)
- Conciseness: (근거 작성)

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


class SNSConversationJudge:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        self.model = "solar-pro"

    def judge(self, user_message, ai_response):
        prompt = SNS_CONVERSATION_JUDGE_PROMPT.format(
            user_message=user_message,
            ai_response=ai_response
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict SNS conversation evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

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
    return reply.replace("[|endofturn|]", "").strip()

def main():
    # 1. 경로 설정
    base_model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    adapter_path = "/data/ephemeral/pro-nlp-finalproject-nlp-13/src/models/exaone_dpo/final_model/policy"
    data_path = "/data/ephemeral/pro-nlp-finalproject-nlp-13/data/persona_data/AiHub_dpo_dataset.json"

    # 2. 데이터 로드 (eval 내의 prompt 추출)
    print(f"📂 데이터 로드 중: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 'eval' 키 안의 리스트에서 'prompt'만 가져오기
    test_prompts = [item['prompt'] for item in full_data.get('eval', [])]
    test_prompts = test_prompts[:10]
    if not test_prompts:
        print("⚠️ 평가 데이터(prompt)를 찾을 수 없습니다.")
        return

    # 3. 모델 로드
    print("🚀 EXAONE DPO 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 4. 평가 실행
    judge = SNSConversationJudge(api_key=API_KEY)
    
    print("\n" + "="*50)
    print(f"🎯 AiHub DPO Eval 데이터 평가 시작 (총 {len(test_prompts)}개)")
    print("="*50 + "\n")

    for i, user_msg in enumerate(test_prompts):
        print(f"[{i+1}/{len(test_prompts)}] 평가 진행 중...")
        
        # 모델 답변 생성
        generated_res = gen_reply(user_msg, tokenizer, model)
        
        # Solar 평가
        evaluation_result = judge.judge(user_msg, generated_res)
        
        # 결과 출력
        print(f"\n💬 User: {user_msg}")
        print(f"🤖 DPO Model: {generated_res}")
        print(f"📝 {evaluation_result}")
        print("-" * 40)

if __name__ == "__main__":
    main()
