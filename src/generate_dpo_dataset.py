import os
import json
import warnings
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import PrivateAttr

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromDicts,
    MergeColumns,
    FormatTextGenerationDPO,
    PreferenceToArgilla
)
from distilabel.steps.tasks import TextGeneration 

from src.data.custom_distilabel import SNSToneFeedback, FilterNoneRatings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

load_dotenv()
API_KEY = os.getenv('UPSTAGE_API_KEY')


def main():
    args = parse_args()

    print(f"===== SFT 데이터셋 랜덤 추출 =====")
    try:
        df_raw = pd.read_csv(args.data_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(args.data_path, encoding='cp949') # 윈도우 생성 파일 대응

    sample_n = min(len(df_raw), args.sample_size)
    df_sampled = df_raw.sample(n=sample_n, random_state=42)

    df_sampled.rename(columns={"req": "instruction"}, inplace=True)
    data_dicts = df_sampled.to_dict(orient="records")
    print(f"!!!!! SFT 데이터셋 {len(data_dicts)}개 추출 완료 !!!!!")


    with Pipeline(name="sns-conversation-dpo-pipeline") as pipeline:
        load_data = LoadDataFromDicts(
            data=data_dicts,
            batch_size=5
        )
        generate_responses = [
            TextGeneration(
                llm=OpenAILLM(
                    model="solar-mini",
                    base_url="https://api.upstage.ai/v1/solar",
                    api_key="up_KyCFVhEHCYkyBTmQOxujiKk55YVAL",
                    max_retries=10,
                ),
                system_prompt="""당신은 문서체로만 텍스트를 생성해야합니다. 주 목표는
                SNS 대화체와 정반대의 성질을 가지는 말투를 가져야 합니다.""",
            )
        ]
        combine_responses = MergeColumns(
            columns=["res", "generation"],
            output_column="generations"
        )
        evaluate_responses = SNSToneFeedback(
            aspect="overall-rating",
            llm=OpenAILLM(
                model="solar-pro2",
                base_url="https://api.upstage.ai/v1/solar",
                api_key=API_KEY,
                max_retries=10,
            ),
        )
        filter_dpo = FilterNoneRatings()
        format_dpo = FormatTextGenerationDPO()

        load_data >> generate_responses >> combine_responses >> evaluate_responses >> filter_dpo >> format_dpo


    print(f"===== DPO 데이터셋 생성중 =====")
    distiset = pipeline.run(use_cache=False)
    df_distiset = distiset["default"]["train"].to_pandas()
    print(f"!!!!! DPO 데이터셋 {len(df_distiset)}개 생성 완료 !!!!!")


    print(f"===== 데이터 저장 =====")
    df_distiset = df_distiset.rename(columns={'instruction': 'prompt'})
    df_distiset = df_distiset[['prompt', 'chosen', 'rejected', 'chosen_rating', 'rejected_rating']]
    train_df, eval_df = train_test_split(df_distiset, test_size=0.1, random_state=42)

    train_dict = train_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')
    eval_dict = eval_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')
    
    combined_data = {
        "train": train_dict,
        "eval": eval_dict
    }

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    print(f"!!!!! 통합 데이터셋 저장 완료 (Train: {len(train_df)}, Eval: {len(eval_df)}) !!!!!")


def parse_args():
    parser = argparse.ArgumentParser(description="SNS DPO 데이터셋 생성 파이프라인")
    parser.add_argument("--data_path", type=str, default="./data/persona_data/AiHub_SNS.csv", help="원본 JSON 데이터 경로")
    parser.add_argument("--output_path", type=str, default="sns_dpo_combined.json", help="결과를 저장할 파일 경로")
    parser.add_argument("--sample_size", type=int, default=10, help="랜덤 추출할 데이터 개수")
    parser.add_argument("--batch_size", type=int, default=5, help="Distilabel 배치 사이즈")
    return parser.parse_args()


if __name__ == "__main__":
    main()
    