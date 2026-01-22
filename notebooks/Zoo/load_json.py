"""
JSON 파일을 읽어 DataFrame으로 변환하는 모듈
"""
from __future__ import annotations

import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _clean_text(text: str, role: str) -> str:
    """텍스트 전처리: 공백 제거, 지칭 변환"""
    text = text.strip()
    text = re.sub(r'감정화자(님)?', 'A', text)
    text = re.sub(r'공감화자(님)?', 'B', text)
    text = re.sub(r'화자(님)?', 'A', text)
    text = re.sub(r'청자(님)?', 'B', text)
    return text


def _process_single_json(json_file: Path, data_type: str, empathy_classes: list) -> list:
    """단일 JSON 파일 처리 (내부 함수)"""
    try:
        file_name = json_file.stem
        parts = file_name.split('_')

        if len(parts) >= 3:
            emotion = parts[1]
            conv_id = parts[-1]
            relation = '_'.join(parts[2:-1])
        else:
            emotion = relation = conv_id = None

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        info = data['info']
        rows = []

        for idx, utterance in enumerate(data['utterances']):
            empathy_ohe = {f'empathy_{cls}': 0 for cls in empathy_classes}
            if utterance.get('listener_empathy'):
                for emp in utterance['listener_empathy']:
                    if emp in empathy_classes:
                        empathy_ohe[f'empathy_{emp}'] = 1

            role = utterance['role']

            row = {
                'data_type': data_type,
                'emotion': emotion,
                'relation': relation,
                'conv_id': conv_id,
                'utterance_num': idx + 1,
                'situation': info['situation'].strip(),
                'avg_rating': info['evaluation']['avg_rating'],
                'grade': info['evaluation']['grade'],
                'role': role,
                'text': _clean_text(utterance['text'], role),
                'terminate': utterance['terminate'],
                **empathy_ohe,
                'speaker_changeEmotion': utterance.get('speaker_changeEmotion')
            }
            rows.append(row)

        return rows
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return []


def load_json_to_dataframe(
    data_path: str | Path,
    data_type: str = 'train',
    relations: list[str] | None = None,
    max_workers: int = 8
) -> pd.DataFrame:
    data_path = Path(data_path)
    empathy_classes = ['위로', '격려', '조언', '동조']

    # 하위 폴더 탐색 (relations 필터 적용)
    if relations:
        folders = [f for f in data_path.iterdir() if f.is_dir() and any(r in f.name for r in relations)]
    else:
        folders = [f for f in data_path.iterdir() if f.is_dir()]

    # 모든 JSON 파일 수집
    all_json_files = []
    for folder in folders:
        all_json_files.extend(list(folder.glob("*.json")))

    print(f"Processing {len(all_json_files)} files from {len(folders)} folders ({data_type})...")

    # 병렬 처리
    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_json, f, data_type, empathy_classes): f
            for f in all_json_files
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            rows = future.result()
            all_data.extend(rows)

    # DataFrame 생성
    df = pd.DataFrame(all_data)

    column_order = [
        'data_type', 'emotion', 'relation', 'conv_id', 'utterance_num',
        'situation', 'avg_rating', 'grade',
        'role', 'text',
        'empathy_위로', 'empathy_격려', 'empathy_조언', 'empathy_동조',
        'speaker_changeEmotion', 'terminate'
    ]

    df = df[column_order]

    return df


def get_conversation_stats(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # text 길이 컬럼 추가
    df = df.copy()
    df['text_len'] = df['text'].str.len()

    # 대화 단위로 그룹화
    grouped = df.groupby(['data_type', 'emotion', 'relation', 'conv_id'])

    # 대화별 통계 계산
    conv_stats = grouped.agg(
        num_turns=('utterance_num', 'max'),
        total_text_len=('text_len', 'sum'),
        avg_text_len=('text_len', 'mean'),
        min_text_len=('text_len', 'min'),
        max_text_len=('text_len', 'max'),
        speaker_turns=('role', lambda x: (x == 'speaker').sum()),
        listener_turns=('role', lambda x: (x == 'listener').sum()),
    ).reset_index()

    if verbose:
        print("=" * 50)
        print("=== 대화 통계 (Conversation Stats) ===")
        print("=" * 50)

        print(f"\n총 대화 수: {len(conv_stats):,}개")

        print("\n[발화 수 (Turns per Conversation)]")
        print(f"  평균: {conv_stats['num_turns'].mean():.1f}")
        print(f"  중앙값: {conv_stats['num_turns'].median():.1f}")
        print(f"  최소: {conv_stats['num_turns'].min()} / 최대: {conv_stats['num_turns'].max()}")

        print("\n[대화당 총 글자 수 (Total Text Length)]")
        print(f"  평균: {conv_stats['total_text_len'].mean():.1f}자")
        print(f"  중앙값: {conv_stats['total_text_len'].median():.1f}자")
        print(f"  최소: {conv_stats['total_text_len'].min()}자 / 최대: {conv_stats['total_text_len'].max()}자")

        print("\n[발화당 평균 글자 수 (Avg Text Length per Turn)]")
        print(f"  평균: {conv_stats['avg_text_len'].mean():.1f}자")
        print(f"  중앙값: {conv_stats['avg_text_len'].median():.1f}자")

        # Emotion별 통계
        print("\n[Emotion별 평균 발화 수]")
        emotion_stats = conv_stats.groupby('emotion')['num_turns'].mean().round(1)
        for emotion, val in emotion_stats.items():
            print(f"  {emotion}: {val}")

        # Relation별 통계
        print("\n[Relation별 평균 발화 수]")
        relation_stats = conv_stats.groupby('relation')['num_turns'].mean().round(1)
        for relation, val in relation_stats.items():
            print(f"  {relation}: {val}")

    return conv_stats


def get_distribution_stats(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Emotion, Relation별 대화 수 분포 통계

    Args:
        df: load_json_to_dataframe으로 생성된 DataFrame
        verbose: 통계 출력 여부 (기본 True)

    Returns:
        분포 통계 딕셔너리

    Example:
        >>> df = load_json_to_dataframe("Training/02.라벨링데이터", data_type='train')
        >>> dist = get_distribution_stats(df)
    """
    # 대화 단위로 unique 추출
    conv_df = df.groupby(['data_type', 'emotion', 'relation', 'conv_id']).size().reset_index(name='turns')

    stats = {
        'total_conversations': len(conv_df),
        'by_emotion': conv_df.groupby('emotion').size().to_dict(),
        'by_relation': conv_df.groupby('relation').size().to_dict(),
        'by_data_type': conv_df.groupby('data_type').size().to_dict(),
        'by_emotion_relation': conv_df.groupby(['emotion', 'relation']).size().to_dict(),
    }

    if verbose:
        print("=" * 50)
        print("=== 대화 분포 통계 (Distribution Stats) ===")
        print("=" * 50)

        print(f"\n총 대화 수: {stats['total_conversations']:,}개")

        print("\n[Emotion별 대화 수]")
        for emotion, count in sorted(stats['by_emotion'].items()):
            print(f"  {emotion}: {count:,}개")

        print("\n[Relation별 대화 수]")
        for relation, count in sorted(stats['by_relation'].items()):
            print(f"  {relation}: {count:,}개")

        print("\n[Data Type별 대화 수]")
        for data_type, count in stats['by_data_type'].items():
            print(f"  {data_type}: {count:,}개")

        print("\n[Emotion x Relation 대화 수]")
        for (emotion, relation), count in sorted(stats['by_emotion_relation'].items()):
            print(f"  {emotion} / {relation}: {count:,}개")

    return stats
