def setup_wandb(config: dict):
    if not config['wandb']['enabled']:
        print("Wandb: Disabled")
        return None
    
    try:
        import wandb
        
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity'),
            name=config.get('experiment_name'),
            tags=config['wandb'].get('tags', []),
            config=config
        )
        
        print("✅ Wandb initialized!")
        return wandb
    
    except ImportError:
        print("⚠️ Wandb not installed. Run: pip install wandb")
        return None