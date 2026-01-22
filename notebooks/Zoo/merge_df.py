"""
Train과 Validation DataFrame을 합치는 모듈
"""

import pandas as pd


def merge_train_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    reset_index: bool = True
) -> pd.DataFrame:
    """
    Train과 Validation DataFrame을 합침

    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        reset_index: 인덱스 리셋 여부 (기본 True)

    Returns:
        합쳐진 pandas.DataFrame

    Example:
        >>> from load_json import load_json_to_dataframe
        >>> from merge_df import merge_train_valid
        >>>
        >>> train_df = load_json_to_dataframe("Training/02.라벨링데이터", data_type='train')
        >>> valid_df = load_json_to_dataframe("Validation/02.라벨링데이터", data_type='validation')
        >>> merged_df = merge_train_valid(train_df, valid_df)
    """
    merged_df = pd.concat([train_df, valid_df], ignore_index=reset_index)

    print(f"=== Merged DataFrame ===")
    print(f"Train: {len(train_df):,} rows")
    print(f"Valid: {len(valid_df):,} rows")
    print(f"Total: {len(merged_df):,} rows")

    return merged_df
