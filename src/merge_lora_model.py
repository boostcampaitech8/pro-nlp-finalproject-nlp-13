from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def main():
    base_path = "Qwen/Qwen3-4B" 
    adapter_path = "src/models/qwen_dpo/final_model/policy" 
    merged_path = "src/models/merged_model/dpo_model"

    if not os.path.exists(merged_path):
        os.makedirs(merged_path, exist_ok=True)
        print(f"Created directory: {merged_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

if __name__=="__main__":
    main()