import pandas as pd
import numpy as np
from typing import Iterable
from pydantic import PrivateAttr
from typing import Any, Dict, List

from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.steps.base import Step, StepInput
from distilabel.typing import ChatType


class SNSToneFeedback(UltraFeedback):
    """SNS 대화체에 특화된 평가 태스크"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aspect = "overall-rating"

    _system_prompt: str = PrivateAttr(
        default="""당신은 실제 SNS 대화체(카카오톡, DM 등)를 판별하는 전문 평가가입니다.

당신의 출력은 오직 아래에 정의된 출력 문법만 허용됩니다.
이를 벗어나는 모든 출력은 파싱 오류로 처리됩니다.

[출력 제약 – 절대 위반 금지]
- 출력은 정확히 4줄이어야 합니다.
- 각 응답은 다음 두 줄로만 구성됩니다:
  1) Rating: 숫자
  2) Rationale: 문장
- 두 응답 묶음(Rating, Rationale 쌍) 사이에는 정확히 빈 줄 한 줄만 허용됩니다. 
- 마지막 Rationale 줄 뒤에는 줄바꿈이나 공백 없이 즉시 종료해야 합니다. 
- Rating에는 오직 1~5 사이의 정수만 허용됩니다.
- Rationale은 반드시 한 문장이어야 합니다.
- Rating, Rationale 외의 어떤 텍스트도 출력해서는 안 됩니다.

[금지 사항]
- 추가 설명, 요약, 평가 근거, 분석
- <think> 태그 또는 내부 사고 노출
- 구분선(---), 제목, 번호, 마크다운 기호
- 출력 전후의 공백 줄 또는 문장

[출력 문법]
Rating: [1-5]
Rationale: [한 문장]

Rating: [1-5]
Rationale: [한 문장]
"""
    )

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """SNS 대화체 판별 및 품질 평가 전용 프롬프트"""

        system_content = self._system_prompt

        user_content = (
            "아래 응답들이 실제 SNS 대화체인지 평가하십시오.\n"
            "평가는 자연스러움, 맥락 이해, 참여 유도, 감정 적절성, 간결성을 종합하여 수행하십시오.\n\n"
        )

        user_content += (
            "[평가 기준]\n"
            "1. 자연스러움 (naturalness) : AI 특유의 기계적이고 형식적인 느낌이 제거되었는지 평가합니다.\n"
            "2. 맥락 적합성 (contextual_relevance) : 이전 사용자 발화를 이해하고 그 흐름을 자연스럽게 이어가는지 평가합니다.\n"
            "3. 참여 유도 (engagement) : 공감 표현, 반응, 질문 등으로 대화를 지속하려는 요소가 있는지 평가합니다.\n"
            "4. 간결성 (conciseness) : SNS 대화 특성에 맞는 적절한 길이인지 평가합니다.\n\n"
            "주의: 설명조, 보고서체, 비즈니스 이메일 말투는 낮은 점수를 부여하십시오.\n\n"
        )

        user_content += f"[지시 사항]\n{input['instruction']}\n\n"

        for i, gen in enumerate(input["generations"], 1):
            user_content += f"[응답 {i}]\n{gen}\n\n"

        user_content += "출력은 system에 정의된 출력 문법만 따르십시오."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


class FilterNoneRatings(Step):
    @property
    def inputs(self) -> list[str]:
        return ["instruction", "generations", "ratings", "rationales"]

    @property
    def outputs(self) -> list[str]:
        return ["instruction", "generations", "ratings", "rationales"]

    def process(self, inputs: StepInput) -> Iterable[list[dict]]:
        filtered_batch = []
        target_columns = [
            'instruction', 'distilabel_metadata', 'model_name', 
            'generations', 'ratings', 'rationales'
        ]
        
        for item in inputs:
            ratings = item.get("ratings")
            if (isinstance(ratings, (list, np.ndarray)) and 
                len(ratings) == 2 and 
                not pd.isna(ratings).any()):
                
                # 에러를 유발하는 'chosen_model', 'rejected_model' 등을 제외하고 복사
                clean_item = {k: v for k, v in item.items() if k in target_columns}
                filtered_batch.append(clean_item)
        
        yield filtered_batch