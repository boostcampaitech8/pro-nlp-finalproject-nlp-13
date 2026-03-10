from typing import Callable, List, Tuple
from transformers import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM


# def get_collator_and_format_func(
#     tokenizer: PreTrainedTokenizer
# ) -> Tuple[DataCollatorForCompletionOnlyLM, Callable]:

#     def formatting_prompts_func(examples: dict) -> List[str]:
#         output_texts = []
#         for messages in examples["messages"]:
#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=False
#             )
#             output_texts.append(text)
#         return output_texts

#     instruction_template = "<|im_start|>user"
#     response_template = "<|im_start|>assistant"

#     # 토크나이저 이슈 방지를 위해 ID로 변환
#     instruction_template_ids = tokenizer.encode(
#         instruction_template, 
#         add_special_tokens=False
#     )
#     response_template_ids = tokenizer.encode(
#         response_template, 
#         add_special_tokens=False
#     )

#     collator = DataCollatorForCompletionOnlyLM(
#         instruction_template=instruction_template_ids,
#         response_template=response_template_ids,
#         tokenizer=tokenizer,
#         mlm=False
#     )

#     return collator, formatting_prompts_func


def get_collator_and_format_func(
    tokenizer: PreTrainedTokenizer
) -> Tuple[DataCollatorForCompletionOnlyLM, Callable]:

    def formatting_prompts_func(examples: dict) -> List[str]:
        output_texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant"

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    return collator, formatting_prompts_func