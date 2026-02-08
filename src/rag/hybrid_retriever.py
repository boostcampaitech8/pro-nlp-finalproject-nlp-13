from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Any, Dict
import yaml

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
cfg = load_yaml("/Users/kyh/NaverAI/final_proj/pro-nlp-finalproject-nlp-13/config/rag.yaml")
db_cfg = cfg.get("db", {})
retriever_cfg = cfg.get("retriever", {})

@dataclass(frozen=True)
class RetrieverConfig:
    db: Chroma 
    bm24_path: str = db_cfg.get("bm25_path", "")
    top_k: int = retriever_cfg.get("top_k", 5)
    bm25_k: int = retriever_cfg.get("bm25_k", 20)
    search_type: str = retriever_cfg.get("search_type", "simpilarity")
    search_kwargs: dict = field(default_factory=lambda: {"k": retriever_cfg.get("dense_k", 20)})   
    weights: list = field(default_factory=lambda: retriever_cfg.get("weights", [0.6, 0.4]))
    
class HybridRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.dense_retriever = self._load_dense_retriever()
        self.bm25 = self._load_bm25_retriever()
    
    def _load_dense_retriever(self) -> VectorStoreRetriever:
        print(f"[RETRIEVER] making dense retriever with TYPE: {self.config.search_type} and ARGS: {self.config.search_kwargs}")
        dense_retriever =  self.config.db.as_retriever(
            search_type=self.config.search_type,
            search_kwargs=self.config.search_kwargs)
        return dense_retriever

    def _load_bm25_retriever(self) -> BM25Retriever:
        print(f"[RETRIEVER] making bm25 retriever with PATH: {self.config.bm24_path}")
        with open(self.config.bm24_path, "rb") as f:
            bm25 = pickle.load(f)
        bm25.k = self.config.bm25_k
        return bm25

    def retrieve(self, query: str) -> List[Document]:
        # 각각 결과 가져오기
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25.invoke(query)

        score_map = {}

        for rank, doc in enumerate(dense_docs):
            score = self.config.weights[0] * (1 / (rank + 1))
            key = doc.page_content
            if key not in score_map:
                score_map[key] = {"doc": doc, "score": 0}
            score_map[key]["score"] += score

        for rank, doc in enumerate(sparse_docs):
            score = self.config.weights[1] * (1 / (rank + 1))
            key = doc.page_content
            if key not in score_map:
                score_map[key] = {"doc": doc, "score": 0}
            score_map[key]["score"] += score

        ranked = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:self.config.top_k]]