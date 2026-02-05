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
            batch_size=args.batch_size
        )
        generate_responses = TextGeneration(
                llm=OpenAILLM(
                    model="solar-pro2",
                    base_url="https://api.upstage.ai/v1/solar",
                    api_key=API_KEY,
                    max_retries=10,
                ),
                columns=["instruction", "res"],
                system_prompt = """
당신은 SNS 대화 응답을 생성하는 AI가 아닙니다.

당신의 역할은 SNS 대화 상황에서 부적절한 형식적인 문서체 응답을 생성하는 것입니다.

[주 목표]
사용자의 발화에 대해 의미적으로는 관련성이 있지만,
SNS 대화체와 정반대의 성질을 가지는 응답을 작성하십시오.

[작성 규칙]

- 반드시 문서체, 설명문, 안내문, 보고서체 중 하나의 스타일을 사용하십시오.
- 객관적이고 정보 전달 중심의 문장 구조를 사용하십시오.
- 일반적인 정보 제공 또는 분석 형태로 작성하십시오.
- 공감 표현, 감정 표현, 친근한 말투를 포함하지 마십시오.
- 사용자에게 질문하지 마십시오.
- 반말 또는 구어체 표현을 사용하지 마십시오.
- SNS 대화 특유의 자연스러운 흐름을 의도적으로 제거하십시오.

[금지 요소]

- 감탄사 또는 감정 표현
- 이모티콘 또는 반복 문자
- 사용자와의 정서적 상호작용
- 친근한 대화 흐름

[출력 규칙]

- 응답 문장만 생성하십시오.
- 불필요한 설명이나 메타 텍스트를 포함하지 마십시오.
- 1~3문장 사이로 작성하십시오.
""",      
            template = """
[사용자 발화]
{{ instruction }}

[적절한 SNS 대화 응답 예시 (chosen)]
{{ res }}

[작성 지침]
위 chosen 응답은 SNS 환경에서 자연스럽고 적절한 대화 응답입니다.
chosen의 말투, 친근함, 대화 흐름을 절대 따라하지 마십시오.

사용자의 발화와 의미적으로 관련성을 유지하면서,
SNS 대화 상황에서는 부자연스럽고 형식적인 문서체 응답을 작성하십시오.

[출력]
""".rstrip()
            )
        
        # combine_responses = MergeColumns(
        #     columns=["res", "generation"],
        #     output_column="generations"
        # )
        # evaluate_responses = SNSToneFeedback(
        #     aspect="overall-rating",
        #     llm=OpenAILLM(
        #         model="solar-pro3",
        #         base_url="https://api.upstage.ai/v1/solar",
        #         api_key=API_KEY,
        #         max_retries=10,
        #     ),
        # )
        # filter_dpo = FilterNoneRatings()
        # format_dpo = FormatTextGenerationDPO()

        load_data >> generate_responses 


    print(f"===== DPO 데이터셋 생성중 =====")
    distiset = pipeline.run(use_cache=True)
    df_distiset = distiset["default"]["train"].to_pandas()
    
    print(f"!!!!! DPO 데이터셋 {len(df_distiset)}개 생성 완료 !!!!!")


    print(f"===== 데이터 저장 =====")
    df_distiset = df_distiset.rename(columns={'instruction': 'prompt', 'res': 'chosen', 'generation': 'rejected'})
    df_distiset = df_distiset[['prompt', 'chosen', 'rejected']]
    print(df_distiset.iloc[0])
    train_df, temp_df = train_test_split(df_distiset, test_size=0.2, random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df = train_df.reset_index(drop=True)
    train_df['index'] = train_df.index

    eval_df = eval_df.reset_index(drop=True)
    eval_df['index'] = eval_df.index

    test_df = test_df.reset_index(drop=True)
    test_df['index'] = test_df.index

    train_dict = train_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')
    eval_dict = eval_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')
    test_dict = test_df.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')

    final_train_data = {
        "train": train_dict,
        "eval": eval_dict
    }

    train_output_file = Path(args.train_output_path)
    test_output_file = Path(args.test_output_path)

    with open(train_output_file, "w", encoding="utf-8") as f:
        json.dump(final_train_data, f, indent=4, ensure_ascii=False)

    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(test_dict, f, indent=4, ensure_ascii=False)

    print(f"!!!!! 통합 데이터셋 저장 완료 (Train: {len(train_df)}, Eval: {len(eval_df)}, Test: {len(test_df)}) !!!!!")


def parse_args():
    parser = argparse.ArgumentParser(description="SNS DPO 데이터셋 생성 파이프라인")
    parser.add_argument("--data_path", type=str, default="./data/persona_data/AiHub_SNS.csv", help="원본 JSON 데이터 경로")
    parser.add_argument("--train_output_path", type=str, default="dpo_dataset.json", help="학습 데이터를 저장할 파일 경로")
    parser.add_argument("--test_output_path", type=str, default="dpo_test_dataset.json", help="평가 데이터를 저장할 파일 경로")
    parser.add_argument("--sample_size", type=int, default=10, help="랜덤 추출할 데이터 개수")
    parser.add_argument("--batch_size", type=int, default=5, help="Distilabel 배치 사이즈")
    return parser.parse_args()


if __name__ == "__main__":
    main()
    