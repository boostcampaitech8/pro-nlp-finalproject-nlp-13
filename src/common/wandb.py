from __future__ import annotations

from typing import Any, Mapping, Optional


def setup_wandb(cfg: Mapping[str, Any]) -> Optional[object]:
    wandb_cfg = cfg.get("wandb") or {}
    enabled = bool(wandb_cfg.get("enabled", False))

    if not enabled:
        print("Wandb: Disabled")
        return None

    project = wandb_cfg.get("project")
    if not project:
        raise ValueError(
            "wandb.enabled is true but wandb.project is missing in config."
        )

    entity = wandb_cfg.get("entity")
    tags = wandb_cfg.get("tags") or []

    run_name = cfg.get("experiment_name") or wandb_cfg.get("name")

    try:
        import wandb

        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            config=dict(cfg),
        )

        print("Wandb initialized!")
        return run

    except ImportError:
        print("Wandb not installed. Run: pip install wandb")
        return None
    except Exception as e:
        print(f"Wandb init failed: {e}")
        return None