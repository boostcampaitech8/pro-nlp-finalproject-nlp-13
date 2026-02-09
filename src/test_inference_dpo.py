import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 설정
def main():
    base_model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B" # 혹은 사용하신 베이스 모델 경로
    adapter_path = "./src/models/exaone_dpo/final_model/policy"

    # 2. 토크나이저 및 모델 로드 (메모리 절약을 위해 4비트 양자화 권장)
    print("--- 모델 로딩 중... ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 학습된 어댑터 병합
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # 3. 테스트 샘플 데이터
    test_samples = [
        {"user": "웅웅 함께 열심히 노력해보자!", "gt": "ㅋㅋ ㅇㅋㅇㅋ 힘내자!"},
        {"user": "NVIDIA GeForce GTX 970가 최소 사양임.", "gt": "그 정도면 다행히 집에서 돌아가겠다. 그런데 이 게임 지금 인기 많나?"},
        {"user": "풍수에 대해 알아?", "gt": "응, 조선 왕실에서 풍수가 얼마나 중요했는지 알쥐?"},
        {"user": "서울 시내의 한 호텔에서 2박 3일을 보내고 집에서 남은 휴가를 보낼 거야", "gt": "호텔에서도 휴가를 즐길 수 있게 되었구나"}
    ]               

    print("\n" + "="*50)
    print("AL-Hub 데이터 test 결과")
    print("="*50 + "\n")

    for i, sample in enumerate(test_samples):
        print(f"=== 테스트 샘플 {i+1} ===")
        print(f"User: {sample['user']}")
        print(f"Chosen: {sample['gt']}")
        
        # 여기서는 학습된 모델(SFT/DPO)의 응답만 출력합니다.
        # Base 모델 응답을 보려면 어댑터를 무효화(disable_adapters)하고 다시 뽑아야 합니다.
        print(f"--- SFT/DPO 모델 응답 ---")
        response = gen_reply(sample['user'], tokenizer, model)
        print(f"gen: {response}\n")


def gen_reply(user_input, tokenizer, model):
    # 1. EXAONE 공식 프롬프트 포맷 적용 (매우 중요)
    # 모델 학습 시 사용했던 템플릿과 동일해야 합니다.
    prompt = f"[|user|]{user_input}[|assistant|]"
    
    # 2. 입력을 텐서로 변환
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. 생성 옵션 최적화
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,      # 답변 길이를 적절히 제한
            do_sample=True,
            temperature=0.5,        # 수치를 낮춰서 더 일관된 답변 유도
            top_p=0.9,
            repetition_penalty=1.2, # "ㅎㅎㅎ" 반복 방지
            eos_token_id=tokenizer.eos_token_id, # 끝나는 지점 명시
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 4. 후처리: 입력 프롬프트는 제외하고 생성된 답변만 추출
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # [|assistant|] 이후의 텍스트만 가져오기
    if "[|assistant|]" in full_text:
        reply = full_text.split("[|assistant|]")[-1].strip()
    else:
        reply = full_text.strip()

    reply = reply.replace("[|endofturn|]", "").replace("[|filenames|]", "").strip()
    
    return reply

if __name__ == "__main__":
    main()
    