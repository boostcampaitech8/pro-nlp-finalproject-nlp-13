import random
import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """
    재현성을 위한 Seed 고정
    
    Args:
        seed: Random seed
        deterministic: CUDNN deterministic 설정 여부
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Seed set to {seed}")