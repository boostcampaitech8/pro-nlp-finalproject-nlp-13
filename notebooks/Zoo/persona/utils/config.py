import os
import yaml
from pathlib import Path
from typing import Union


def load_config(config_path: Union[str, Path]) -> dict:
    """
    YAML Config 로드
    
    Args:
        config_path: Config 파일 경로
    
    Returns:
        Config dictionary
    """
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    print(f"Loading config: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # config 파일이 있는 디렉토리
    config_dir = Path(config_path).parent
    
    # 상대 경로들을 절대 경로로 변환
    _resolve_paths(config, config_dir)
    
    return config

def _resolve_paths(config: dict, base_dir: Path):
    """
    Config 내의 모든 파일 경로를 절대 경로로 변환
    """
    # 데이터 파일 경로
    if 'data' in config and 'train_file' in config['data']:
        train_file = config['data']['train_file']
        if not os.path.isabs(train_file):
            # config 파일 기준 상대 경로 -> 절대 경로
            config['data']['train_file'] = str(base_dir / train_file)
            print(f"Resolved train_file: {config['data']['train_file']}")
    
    # 프롬프트 경로
    if 'model' in config and 'prompt_path' in config['model']:
        prompt_path = config['model']['prompt_path']
        if not os.path.isabs(prompt_path):
            config['model']['prompt_path'] = str(base_dir / prompt_path)
            print(f"Resolved prompt_path: {config['model']['prompt_path']}")
    
    # output 디렉토리
    if 'paths' in config:
        for key in ['output_dir', 'logging_dir']:
            if key in config['paths']:
                path = config['paths'][key]
                if not os.path.isabs(path):
                    config['paths'][key] = str(base_dir / path)
                    print(f"Resolved {key}: {config['paths'][key]}")