"""
DataFrame을 window size, stride로 QA 데이터셋으로 변환하는 모듈
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm


def create_qa_dataset(
    df: pd.DataFrame,
    window_size: int = 4,
    stride: int = 1,
    response_role: str | None = 'listener',
    empathy_only: bool = True,
) -> pd.DataFrame:
    """
    DataFrame을 슬라이딩 윈도우로 멀티턴 QA 데이터셋으로 변환

    Args:
        df: 발화 DataFrame (load_json_to_dataframe 결과)
        window_size: 윈도우 크기 (context 턴 수 + 1, 기본 4 = 3턴 context + 1턴 response)
        stride: 슬라이딩 간격 (기본 1)
        response_role: 응답자 필터링
            - 'listener': listener(B)만 (기본)
            - 'speaker': speaker(A)만
            - None: 전체
        empathy_only: True면 empathy 태그가 1개 이상 있는 응답만 포함 (기본 True)

    Returns:
        QA DataFrame (context, response, response_role 컬럼 포함)

    Example:
        >>> from load_json import load_json_to_dataframe
        >>> from create_qa import create_qa_dataset
        >>>
        >>> df = load_json_to_dataframe("Training/02.라벨링데이터", data_type='train')
        >>> qa_df = create_qa_dataset(df, window_size=5, stride=2, empathy_only=True)
    """
    qa_data = []

    df = df.sort_values(['emotion', 'relation', 'conv_id', 'utterance_num'])
    grouped = df.groupby(['data_type', 'emotion', 'relation', 'conv_id'])

    desc = f"Creating QA (window={window_size}, stride={stride})"

    for (data_type, emotion, relation, conv_id), group in tqdm(grouped, desc=desc):
        utterances = group.to_dict('records')

        if len(utterances) < window_size:
            continue

        for i in range(0, len(utterances) - window_size + 1, stride):
            window = utterances[i:i + window_size]

            context_turns = window[:-1]
            response_turn = window[-1]

            resp_role = 'B' if response_turn['role'] == 'listener' else 'A'

            # response_role 필터링
            if response_role == 'listener' and response_turn['role'] != 'listener':
                continue
            if response_role == 'speaker' and response_turn['role'] != 'speaker':
                continue

            # empathy_only 필터링: 4개 태그 중 1개라도 1이면 공감발화
            if empathy_only:
                has_empathy = (
                    response_turn['empathy_위로'] == 1 or
                    response_turn['empathy_격려'] == 1 or
                    response_turn['empathy_조언'] == 1 or
                    response_turn['empathy_동조'] == 1
                )
                if not has_empathy:
                    continue

            context_lines = []
            for turn in context_turns:
                role_label = "A" if turn['role'] == 'speaker' else "B"
                context_lines.append(f"{role_label}: {turn['text']}")

            context = "\n".join(context_lines)

            qa_data.append({
                'data_type': data_type,
                'emotion': emotion,
                'relation': relation,
                'conv_id': conv_id,
                'situation': utterances[0]['situation'],
                'context': context,
                'response': response_turn['text'],
                'response_role': resp_role,
                'empathy_위로': response_turn['empathy_위로'],
                'empathy_격려': response_turn['empathy_격려'],
                'empathy_조언': response_turn['empathy_조언'],
                'empathy_동조': response_turn['empathy_동조'],
                'speaker_changeEmotion': response_turn['speaker_changeEmotion']
            })

    result_df = pd.DataFrame(qa_data)

    print(f"\n=== QA Dataset Created ===")
    print(f"Total: {len(result_df):,} samples")
    print(f"Window size: {window_size}, Stride: {stride}")
    if response_role:
        print(f"Response role filter: {response_role}")
    print(f"Empathy only: {empathy_only}")

    return result_df


def get_qa_stats(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    QA 데이터셋의 통계 확인 (빈도수, 길이 등)

    Args:
        df: create_qa_dataset으로 생성된 QA DataFrame
        verbose: 통계 출력 여부 (기본 True)

    Returns:
        통계 정보 딕셔너리

    Example:
        >>> qa_df = create_qa_dataset(df, window_size=3)
        >>> stats = get_qa_stats(qa_df)
    """
    df = df.copy()

    # 길이 계산
    df['context_len'] = df['context'].str.len()
    df['response_len'] = df['response'].str.len()
    df['situation_len'] = df['situation'].str.len()

    stats = {
        'total_samples': len(df),
        'context_len': {
            'mean': df['context_len'].mean(),
            'median': df['context_len'].median(),
            'min': df['context_len'].min(),
            'max': df['context_len'].max(),
        },
        'response_len': {
            'mean': df['response_len'].mean(),
            'median': df['response_len'].median(),
            'min': df['response_len'].min(),
            'max': df['response_len'].max(),
        },
        'by_emotion': df['emotion'].value_counts().to_dict(),
        'by_relation': df['relation'].value_counts().to_dict(),
        'by_data_type': df['data_type'].value_counts().to_dict(),
    }

    # Empathy 태그별 통계
    empathy_cols = ['empathy_위로', 'empathy_격려', 'empathy_조언', 'empathy_동조']
    stats['empathy_counts'] = {col: int(df[col].sum()) for col in empathy_cols}

    if verbose:
        print("=" * 50)
        print("=== QA 데이터셋 통계 ===")
        print("=" * 50)

        print(f"\n총 샘플 수: {stats['total_samples']:,}개")

        print("\n[Context 길이 (글자 수)]")
        print(f"  평균: {stats['context_len']['mean']:.1f}자")
        print(f"  중앙값: {stats['context_len']['median']:.1f}자")
        print(f"  최소: {stats['context_len']['min']}자 / 최대: {stats['context_len']['max']}자")

        print("\n[Response 길이 (글자 수)]")
        print(f"  평균: {stats['response_len']['mean']:.1f}자")
        print(f"  중앙값: {stats['response_len']['median']:.1f}자")
        print(f"  최소: {stats['response_len']['min']}자 / 최대: {stats['response_len']['max']}자")

        print("\n[Emotion별 샘플 수]")
        for emotion, count in stats['by_emotion'].items():
            print(f"  {emotion}: {count:,}개")

        print("\n[Relation별 샘플 수]")
        for relation, count in stats['by_relation'].items():
            print(f"  {relation}: {count:,}개")

        print("\n[Data Type별 샘플 수]")
        for data_type, count in stats['by_data_type'].items():
            print(f"  {data_type}: {count:,}개")

        print("\n[Empathy 태그별 샘플 수]")
        for tag, count in stats['empathy_counts'].items():
            print(f"  {tag}: {count:,}개")

    return stats
