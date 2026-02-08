from __future__ import annotations

import os
import pickle
from typing import Any, Dict
from dataclasses import dataclass, field
import yaml

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
cfg = load_yaml("/Users/kyh/NaverAI/final_proj/pro-nlp-finalproject-nlp-13/config/rag.yaml")
db_cfg = cfg.get("db", {})

@dataclass(frozen=True)
class RagConfig:
    db_path: str = db_cfg.get("path", "")
    bm25_path: str = db_cfg.get("bm25_path", "")
    embedding_model_name: str = db_cfg.get("model_name", "solar-embedding-1-large")
    documents: list = field(default_factory=list)
    collection_name: str = db_cfg.get("collection_name", "")


class Rag:
    def __init__(self, config: RagConfig):
        self.config = config
        self.embeddings = UpstageEmbeddings(model=config.embedding_model_name)
        
    def build(self) -> None:
        os.makedirs(self.config.db_path, exist_ok=True)

        bm25_dir = os.path.dirname(self.config.bm25_path)
        if bm25_dir:
            os.makedirs(bm25_dir, exist_ok=True)

        vectordb = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=self.config.db_path
        )
        
        batch_size = 100
        total_docs = len(self.config.documents)

        print(f"[RAG][BUILDER] building DB with docs.. total: {total_docs}")

        for i in range(0, total_docs, batch_size):
            batch = self.config.documents[i : i + batch_size]
            vectordb.add_documents(batch)
            print(f"[RAG][BUILDER] progress {min(i + batch_size, total_docs)} / {total_docs}")
        
        print(f"[RAG][BUILDER] building vector db is completed! now progressing pickle for bm25. PATH: {self.config.bm25_path}")
        bm25 = BM25Retriever.from_documents(self.config.documents)

        with open(self.config.bm25_path, "wb") as f:
            pickle.dump(bm25, f)
            
        print(f"[RAG][BUILDER] building bm25 is completed!")

    def load(self) -> Chroma:
        print(f"[RAG][LOADER] loading DB with path: {self.config.db_path}")
        return Chroma(
            persist_directory=self.config.db_path,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
        )

    def load_bm25(self) -> BM25Retriever:
        print(f"[RAG][LOADER] loading bm25 from: {self.config.bm25_path}")
        with open(self.config.bm25_path, "rb") as f:
            return pickle.load(f)