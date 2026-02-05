"""
전체 데이터 전처리 파이프라인
CSV → Dataset → 토큰화 → 길이 분석
"""
import argparse
from csv_to_dataset import CSVToDataset
from tokenize_dataset import DatasetTokenizer
from analyze_token_length import TokenLengthAnalyzer
from datasets import DatasetDict
import tempfile
import os 


def main():
    parser = argparse.ArgumentParser(description="데이터 전처리 파이프라인")
    
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 파일 경로")
    parser.add_argument("--tokenizer_name", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B", help="토크나이저 이름")
    parser.add_argument("--split_ratio", type=float, default=None, help="Train/Test 분할 비율")
    parser.add_argument("--slow_tokenizer", type=str, default="None", choices=["True", "False", "None"])
    parser.add_argument("--batch_size", type=int, default=1000, help="토큰화 배치 크기")
    parser.add_argument("--num_proc", type=int, default=4, help="병렬 처리 프로세스 수")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic 모드")
    
    args = parser.parse_args()
    
    # slow_tokenizer 파싱
    if args.slow_tokenizer == "None":
        slow_tokenizer = None
    elif args.slow_tokenizer == "True":
        slow_tokenizer = True
    else:
        slow_tokenizer = False
    
    print("데이터 전처리 파이프라인 시작")
    print(f"CSV: {args.csv_path}")
    print(f"토크나이저: {args.tokenizer_name}\n")
    
    # Step 1: CSV → Dataset
    print("Step 1: CSV → Dataset 변환")
    converter = CSVToDataset(
        csv_path=args.csv_path,
        seed=args.seed,
        deterministic=args.deterministic,
        usecols=['situation', 'context', 'response']
    )
    dataset = converter.convert_to_dataset(split_ratio=args.split_ratio)
    
    # Step 2: Dataset → 토큰화
    print("\nStep 2: Dataset 토큰화")
    tokenizer_obj = DatasetTokenizer(
        tokenizer_name=args.tokenizer_name,
        slow_tokenizer=slow_tokenizer,
        seed=args.seed,
        deterministic=args.deterministic,
        batch_size=args.batch_size
    )
    tokenizer_obj.load_tokenizer()
    
    if isinstance(dataset, DatasetDict):
        tokenized_dataset = DatasetDict()
        for split in dataset.keys():
            print(f"\n[{split} split 토큰화]")
            tokenized_dataset[split] = tokenizer_obj.tokenize_dataset(
                dataset[split],
                num_proc=args.num_proc
            )
    else:
        tokenized_dataset = tokenizer_obj.tokenize_dataset(
            dataset,
            num_proc=args.num_proc
        )
    
    # Step 3: 토큰 길이 분석
    print("\nStep 3: 토큰 길이 분석")
    
    # 임시 디렉토리에 저장 후 분석
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "tokenized_temp")
        tokenized_dataset.save_to_disk(temp_path)
        
        analyzer = TokenLengthAnalyzer(tokenized_dataset_path=temp_path)
        analyzer.run_analysis()
    
    print("\n파이프라인 완료!")


if __name__ == "__main__":
    main()