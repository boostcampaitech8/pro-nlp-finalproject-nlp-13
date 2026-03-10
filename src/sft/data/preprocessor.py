import ast
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict


def parse_and_add_system_prompt(
    dataset: Union[Dataset, DatasetDict],
    system_content: str
) -> Union[Dataset, DatasetDict]:
    
    def _process_row(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example["messages"]

        if isinstance(messages, str):
            try:
                messages = ast.literal_eval(messages)
            except (ValueError, SyntaxError):
                pass

        if messages and messages[0].get("role") != "system":
            system_message = {"role": "system", "content": system_content}
            messages = [system_message] + messages
        
        return {"messages": messages}

    print(f"[Applying preprocessing]: Parsing & Adding System Prompt ('{system_content}...')...")
    
    return dataset.map(_process_row)