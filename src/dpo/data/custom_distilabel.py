import pandas as pd
import numpy as np
from typing import Iterable
from pydantic import PrivateAttr
from typing import Any, Dict, List

from distilabel.steps.tasks import TextGeneration, UltraFeedback
from distilabel.steps.base import Step, StepInput
from distilabel.typing import ChatType

from src.dpo.prompt.dpo_prompts import SNS_FEEDBACK_SYSTEM_PROMPT, SNS_FEEDBACK_USER_TEMPLATE

class SNSToneFeedback(UltraFeedback):
    """SNS 대화체에 특화된 평가 태스크"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aspect = "overall-rating"

    _system_prompt: str = PrivateAttr(default=SNS_FEEDBACK_SYSTEM_PROMPT)

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """SNS 대화체 판별 및 품질 평가 전용 프롬프트"""

        system_content = self._system_prompt

        user_content = SNS_FEEDBACK_USER_TEMPLATE
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