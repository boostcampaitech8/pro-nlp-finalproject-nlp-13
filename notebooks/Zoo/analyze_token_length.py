"""
토큰화된 Dataset의 길이 분석
"""
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np
from typing import Dict, List
from tqdm import tqdm


class TokenLengthAnalyzer:
    """토큰 길이 분석 클래스"""
    
    def __init__(self, tokenized_dataset_path: str):
        """
        Args:
            tokenized_dataset_path: 토큰화된 Dataset 경로
        """
        self.dataset_path = tokenized_dataset_path
        self.dataset = None
        self.statistics = None

    def load_dataset(self):
        """토큰화된 Dataset 로드"""
        print(f'데이터셋 불러오기: {self.dataset_path}')
        self.dataset = load_from_disk(self.dataset_path)
        
        if isinstance(self.dataset, DatasetDict):
            for split, ds in self.dataset.items():
                print(f'  - {split} 샘플 수: {len(ds)}')
        else:
            print(f'  - 전체 샘플 수: {len(self.dataset)}')

        return self.dataset
        
    def calculate_lengths(self, dataset: Dataset) -> Dict[str, List[int]]:
        """토큰 길이 계산"""
        print("토큰 길이 계산 중...")
        lengths = {
            'situation_lengths': [],
            'context_lengths': [],
            'response_lengths': [],
            'total_lengths': []
        }

        for sample in tqdm(dataset, desc="길이 계산"):
            sit_len = len(sample['situation_tokens'])
            ctx_len = len(sample['context_tokens'])
            resp_len = len(sample['response_tokens'])
            
            lengths['situation_lengths'].append(sit_len)
            lengths['context_lengths'].append(ctx_len)
            lengths['response_lengths'].append(resp_len)
            lengths['total_lengths'].append(sit_len + ctx_len + resp_len)
        
        print('토큰 길이 계산 완료!')
        return lengths

    def analyze_statistics(self, lengths: Dict[str, List[int]]) -> Dict[str, Dict]:
        """통계 분석"""
        print("\n통계 분석 중...")
        statistics = {}

        for key, value in lengths.items():
            arr = np.array(value)
            stats = {
                'count': len(value),
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'min': int(np.min(arr)),
                'max': int(np.max(arr)),
                'std': float(np.std(arr)),
                'percentile_25': float(np.percentile(arr, 25)),
                'percentile_75': float(np.percentile(arr, 75)),
                'percentile_90': float(np.percentile(arr, 90)),
                'percentile_95': float(np.percentile(arr, 95)),
                'percentile_99': float(np.percentile(arr, 99))
            }
            statistics[key] = stats

            print(f"\n[{key}]")
            print(f"  샘플 수: {stats['count']}")
            print(f"  평균: {stats['mean']:.1f}")
            print(f"  중앙값: {stats['median']:.1f}")
            print(f"  최소: {stats['min']}")
            print(f"  최대: {stats['max']}")
            print(f"  표준편차: {stats['std']:.1f}")
            print(f"  95% percentile: {stats['percentile_95']:.1f}")
            print(f"  99% percentile: {stats['percentile_99']:.1f}")

        return statistics
    
    def run_analysis(self):
        """전체 분석 실행"""
        self.load_dataset()
        
        if isinstance(self.dataset, DatasetDict):
            all_statistics = {}
            for split_name, split_dataset in self.dataset.items():
                print(f"\n[{split_name} split 분석]")
                
                lengths = self.calculate_lengths(split_dataset)
                stats = self.analyze_statistics(lengths)
                all_statistics[split_name] = stats
            
            self.statistics = all_statistics
            return all_statistics
        
        else:  
            lengths = self.calculate_lengths(self.dataset)
            statistics = self.analyze_statistics(lengths)
            
            self.statistics = statistics
            return statistics