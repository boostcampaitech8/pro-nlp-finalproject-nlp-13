from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from dataclasses import dataclass, field

import pickle

@dataclass(frozen=True)
class RetrieverConfig:
    db: Chroma
    pickle_path: str
    top_k: int = 5
    bm25_k: int = 20
    search_type: str = "similarity"
    search_kwargs: dict = field(default_factory=lambda: {"k": 20})   
    weights: list = field(default_factory=lambda: [0.6, 0.4])
    
class HybridRetriever:
    def __init__(
        self,
        config: RetrieverConfig):
        self.config = config
        self.dense_retriever = self.__load_dense_retriever__()
        self.bm_25 = self.__load_bm25_retriever__()
        
    def __load_dense_retriever__(self) -> VectorStoreRetriever:
        print(f"[RETRIEVER] making dense retriever with TYPE: {self.config.search_type} and ARGS: {self.config.search_kwargs}")
        dense_retriever =  self.config.db.as_retriever(
            search_type=self.config.search_type,
            search_kwargs=self.config.search_kwargs)
        return dense_retriever
        
    def __load_bm25_retriever__(self) -> BM25Retriever:
        print(f"[RETRIEVER] making bm25 retriever with PATH: {self.config.pickle_path}")
        with open(self.config.pickle_path, "rb") as f:
            bm25 = pickle.load(f)
        bm25.k = self.config.bm25_k
        return bm25
        
    def retrieve(self, query: str):
        # 각각 결과 가져오기
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm_25.invoke(query)

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