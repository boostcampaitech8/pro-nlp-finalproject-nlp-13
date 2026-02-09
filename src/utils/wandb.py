from typing import Any, Dict, Optional

import wandb

def wandb_init(
        config: Dict[str, Any],
        project_name: str="my_project",
        entity: str="pro-nlp-final-13",
        run_name: Optional[str] = None,
    ):
    wandb.init(
        project=project_name, 
        entity=entity,  
        name=run_name,
        config=config
    )

def wandb_finish() -> None:
    if wandb.run is not None:
        wandb.finish()