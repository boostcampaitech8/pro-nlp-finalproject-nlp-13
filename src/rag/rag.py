from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings


@dataclass(frozen=True)
class RagConfig:
    db_path: str = "src/db/chroma_db"
    bm25_path: str = "src/db/bm25.pkl"
    embedding_model_name: str = "solar-embedding-1-large"
    documents: list = field(default_factory=list)
    collection_name: str = "test"


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