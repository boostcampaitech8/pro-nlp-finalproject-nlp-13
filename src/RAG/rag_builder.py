from src.rag.rag import RagConfig, Rag
from src.rag.data_loader import make_docs
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
    
def main():
    load_dotenv() # api key load

    DB_PATH = "./db/chroma_db"
    BM25_PATH = "./db/bm25"
    embedding_model_name = "solar-embedding-1-large"
    file_path = ['data/crawling/stay_crawling.csv', 'data/crawling/sports_crawling.csv', 'data/guidebook/busan_rag_data.json']
    documents = make_docs(file_path)
    config = RagConfig(db_path=DB_PATH, bm25_path=BM25_PATH, embedding_model_name=embedding_model_name, documents=documents)
    rag = Rag(config=config)
    rag.build()
    
if __name__ == "__main__":
    main()