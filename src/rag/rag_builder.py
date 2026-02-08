from __future__ import annotations

import argparse
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.RAG.data_loader import make_docs
from src.RAG.rag import RagConfig, Rag


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="e.g. config/rag.yaml (config 폴더는 src와 같은 레벨)",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    embedding_cfg = cfg.get("embedding", {})
    file_paths = embedding_cfg.get("file_path", [])
    documents = make_docs(file_paths)
    config = RagConfig(documents=documents)
    rag = Rag(config=config)
    rag.build()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

    
if __name__ == "__main__":
    main()