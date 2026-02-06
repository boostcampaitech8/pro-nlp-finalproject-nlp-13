from src.RAG.rag import RagConfig, Rag
from src.RAG.data_loader import make_docs
from dotenv import load_dotenv
from typing import Any, Dict
import yaml

import argparse

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
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
    model_name = embedding_cfg.get("model_name", "")
    db_path = cfg.get("db_path", "")
    bm25_path = cfg.get("bm25_path", "")
    documents = make_docs(file_paths)
    config = RagConfig(db_path=db_path, bm25_path=bm25_path, embedding_model_name=model_name, documents=documents)
    rag = Rag(config=config)
    rag.build()
    
if __name__ == "__main__":
    main()