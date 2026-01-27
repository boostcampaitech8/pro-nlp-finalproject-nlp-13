from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import pickle
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RagConfig:
    db_path: str
    embeddings: Embeddings
    bm25_path: str = ""
    documents: list = field(default_factory=list)
    collection_name: str = "test"
    
class Rag:
    def __init__(self, 
                 config: RagConfig):
        self.config = config
    
    def build(self):
        vectordb = Chroma(
            embedding_function=self.config.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=self.config.db_path)
        
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
        print(f"[RAG-BUILDER] loading DB with path: {self.config.db_path}")
        return Chroma(
        persist_directory=self.config.db_path,
        embedding_function=self.config.embeddings,
        collection_name=self.config.collection_name,)
    