"""
데이터 처리 설정 파일
"""
from pathlib import Path

# ======================
# 경로 설정
# ======================
BASE_PATH = Path(".")
TRAINING_PATH = BASE_PATH / "Training" / "02.라벨링데이터"
VALIDATION_PATH = BASE_PATH / "Validation" / "02.라벨링데이터"

# ======================
# 데이터 로드 설정
# ======================
LOAD_CONFIG = {
    "relations": ["친구", "지인"],  # None이면 전체
    "max_workers": 8,               # 병렬 처리 스레드 수
}

# ======================
# QA 데이터셋 설정
# ======================
QA_CONFIG = {
    "window_size": 4,             # context 턴(ABA) 수 + 1(B)
    "stride": 1,
    "response_role": "listener",  # 'listener', 'speaker', None
}

# ======================
# 출력 설정
# ======================
OUTPUT_CONFIG = {
    "save_dir": BASE_PATH / "processed",
    "single_qa_filename": "single_qa.csv",
    "multi_qa_filename": "multi_qa.csv",
}
