from rag import RagConfig, Rag
from data_loader import make_docs
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
    
def main():
    load_dotenv() # api key load

    DB_PATH = "./db/chroma_db"
    BM25_PATH = "./db/bm25"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    file_path = ['../../data/crawling/stay_crawling.csv', '../../data/crawling/sports_crawling.csv', '../../data/guidebook/busan_rag_data.json']
    documents = make_docs(file_path)
    config = RagConfig(db_path=DB_PATH, bm25_path=BM25_PATH, embeddings=embeddings, documents=documents)
    rag = Rag(config=config)
    rag.build()
    
if __name__ == "__main__":
    main()