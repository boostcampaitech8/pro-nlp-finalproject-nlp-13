from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class RAGBuilder:
    def __init__(self, 
                 db_path: str,
                 embeddings: Embeddings,
                 documents: list,
                 collection_name: str = "test",
                 ):
        self.db_path = db_path
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.documents = documents
    
    def build(self):
        vectordb = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.db_path)
        
        batch_size = 100
        total_docs = len(self.documents)

        print(f"[RAG-BUILDER] building DB with docs.. total: {total_docs}")

        for i in range(0, total_docs, batch_size):
            batch = self.documents[i : i + batch_size]
            vectordb.add_documents(batch)
            print(f"[RAG-BUILDER] progress {min(i + batch_size, total_docs)} / {total_docs}")
    
    def load(self):
        print(f"[RAG-BUILDER] loading DB with path: {self.db_path}")
        return Chroma(
        persist_directory=self.db_path,
        embedding_function=self.embeddings,
        collection_name=self.collection_name,)
    