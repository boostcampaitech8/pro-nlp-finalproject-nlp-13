import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("Seed 고정 완료")


def get_model_template_type(model_name):
    """
    모델명으로 template type 자동 결정
    
    Args:
        model_name: 모델 이름
    
    Returns:
        template_type: exaone, llama3, mistral
    """
    MODEL_MAPPING = {
        "exaone": "exaone",
        "llama": "llama3",
        "mistral": "mistral",
    }
    
    model_name_lower = model_name.lower()
    
    for key, template in MODEL_MAPPING.items():
        if key in model_name_lower:
            return template
    
    # 기본값
    print(f"⚠️ Unknown model '{model_name}', using default template 'exaone'")
    return "exaone"


def print_gpu_utilization():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("CUDA not available")