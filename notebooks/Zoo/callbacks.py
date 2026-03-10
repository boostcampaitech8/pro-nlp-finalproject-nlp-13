from transformers import TrainerCallback
import random
import torch


class GenerationCallback(TrainerCallback):
    """
    Evaluation 시 샘플 생성 결과 확인
    """
    
    def __init__(self, tokenizer, eval_dataset, num_samples=5, max_new_tokens=100):
        """
        Args:
            tokenizer: Tokenizer
            eval_dataset: Evaluation dataset
            num_samples: 생성할 샘플 수
            max_new_tokens: 생성할 최대 토큰 수
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        
        # 고정 샘플 인덱스 (매번 같은 샘플 확인)
        self.sample_indices = random.sample(
            range(len(eval_dataset)),
            min(num_samples, len(eval_dataset))
        )
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Evaluation 후 샘플 생성"""
        print("\n" + "="*70)
        print(f"📝 샘플 생성 결과 (Step {state.global_step})")
        print("="*70)
        
        model.eval()
        
        for idx, sample_idx in enumerate(self.sample_indices):
            sample = self.eval_dataset[sample_idx]
            
            # Assistant 전까지만 prompt
            input_ids = sample['input_ids']
            assistant_start = self._find_assistant_start(input_ids)
            
            if assistant_start is None:
                continue
            
            prompt_ids = input_ids[:assistant_start]
            gt_response = self._extract_gt_response(input_ids, assistant_start)
            
            # 생성
            prompt_tensor = torch.tensor([prompt_ids]).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    prompt_tensor,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][len(prompt_ids):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Prompt 텍스트
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            
            # 출력
            print(f"\n[Sample {idx + 1}]")
            print(f"Context: ...{prompt_text}")
            print(f"\n✓ GT:  {gt_response}")
            print(f"→ Gen: {generated_text}")
            print("-" * 70)
        
        print("="*70 + "\n")
    
    def _find_assistant_start(self, input_ids):
        """[|assistant|] 위치 찾기"""
        for i in range(len(input_ids)):
            token_str = self.tokenizer.decode([input_ids[i]])
            if "[|assistant|]" in token_str:
                return i + 1
        return None
    
    def _extract_gt_response(self, input_ids, start_idx):
        """Ground Truth 응답 추출"""
        response_ids = []
        for token_id in input_ids[start_idx:]:
            if token_id == self.tokenizer.pad_token_id:
                break
            
            token_str = self.tokenizer.decode([token_id])
            if "[|endofturn|]" in token_str:
                break
            
            response_ids.append(token_id)
        
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)