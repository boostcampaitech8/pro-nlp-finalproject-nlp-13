import pandas as pd
import os
import re
from langchain_core.documents import Document

TARGET_COLUMNS = [
        '장소명', '구분', '설명', '메뉴', '카테고리',
        '한줄설명', '리뷰', '리뷰 개수', '평점', '태그',
        '가격', '영업시간', '주소', '연락처', '출처'
        ]
    
def __load__(file_paths: list[str]) -> pd.DataFrame:
    df_list = []
    print(f"[DATA-LOADER] 총 {len(file_paths)} 개의 파일 로드를 시작합니다.")
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[DATA-LOADER][WARNING] No path existed: {path}")
            continue
        _, ext = os.path.splitext(path)
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext == '.json':
            df = pd.read_json(path)
        else:
            print(f"[DATA-LOADER][WARNING] The extension is wrong: {path}")
            continue
        
        df_list.append(df)
        print(f"[DATA-LOADER] Path: {path} HAS {len(df)} rows")
        
    if not df_list:
        raise ValueError("로드된 데이터가 없습니다. 경로를 확인해주세요.")
    
    raw_data = pd.concat(df_list, ignore_index=True)
    print(f"[DATA-LOADER] total rows: {len(raw_data)}")
    return raw_data
    
def __preprocess__(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'(\\n|\r|\t|\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()
    
def make_docs(file_paths: list[str]) -> list[Document]:
    raw_data = __load__(file_paths)
    documents = []
    for _, row in raw_data.iterrows():
        
        valid_lines = []
        sort = ""
    
        for col_name in TARGET_COLUMNS:
            val = row.get(col_name)
            if pd.isna(val):
                continue
            if isinstance(val, (list, dict)) and len(val) == 0:
                continue
            # 전처리
            clean_val = __preprocess__(val)
            if clean_val == "":
                continue

            valid_lines.append(f"{col_name}: {clean_val}")
            if col_name == '구분':
                sort = clean_val
        
        content = "\n".join(valid_lines)
        if not content:
            continue
        doc = Document(page_content=content, metadata={"sort":sort})
        documents.append(doc)
        
    return documents
    