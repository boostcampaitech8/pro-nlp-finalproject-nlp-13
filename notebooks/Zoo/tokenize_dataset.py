from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from typing import Optional, Dict, Any
from utils import set_seed

class DatasetTokenizer:
    def __init__(
            self,
            tokenizer_name:str,
            slow_tokenizer:Optional[bool]=None,
            seed:Optional[int]=42,
            deterministic:bool = False,
            batch_size: Optional[int] = 1000
        ):

        self.tokenizer_name = tokenizer_name
        self.slow_tokenizer = slow_tokenizer
        self.seed = seed
        self.deterministic = deterministic
        self.tokenizer = None
        self.batch_size = batch_size

        set_seed(self.seed, deterministic= self.deterministic)

    def load_tokenizer(self):
        print(f'토크나이저 로딩: {self.tokenizer_name}')
        if self.slow_tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, slow_tokenizer=self.slow_tokenizer)

        print(f"토크나이저 로드 완료!")
        print(f"  - Type: {type(self.tokenizer).__name__}")
        print(f"  - Vocab size: {self.tokenizer.vocab_size}")

        return self.tokenizer
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        
        if self.tokenizer is None:
            raise ValueError("토크나이저를 먼저 로드하세요")
        
        situation_tokens = self.tokenizer(
            examples['situation'],
            add_special_tokens=False
        )['input_ids']
        
        context_tokens = self.tokenizer(
            examples['context'],
            add_special_tokens=False
        )['input_ids']
        
        response_tokens = self.tokenizer(
            examples['response'],
            add_special_tokens=False
        )['input_ids']
        
        return {
            'situation_tokens': situation_tokens,
            'context_tokens': context_tokens,
            'response_tokens': response_tokens
        }
    
    def tokenize_dataset(
        self, 
        dataset: Dataset,
        num_proc: int = 4
    ) -> Dataset:
        print(f"\nDataset 토큰화 중...")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Num processes: {num_proc}")
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.batch_size,
            num_proc=num_proc,
            desc="토큰화"
        )
        
        print(f"토큰화 완료!")
        return tokenized_dataset
    