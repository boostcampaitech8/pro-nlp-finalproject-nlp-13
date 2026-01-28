from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSerializable
from langchain_community.retrievers import BM25Retriever
from dataclasses import dataclass, field
import pickle

@dataclass(frozen=True)
class RetrieverConfig:
    db: Chroma
    pickle_path: str
    model_name: str
    top_k: int = 5
    search_type: str = "similarity"
    search_kwargs: dict = field(default_factory=lambda: {"k": 5})   
    weights: list = field(default_factory=lambda: [0.6, 0.4])
    
class HybridRetriever:
    def __init__(
        self,
        config: RetrieverConfig):
        self.config = config
        self.dense_retriever = self.__load_dense_retriever__()
        self.bm_25 = self.__load_bm25_retriever__()
        model = ChatOpenAI(
            model=self.config.model_name, 
            temperature=0)
        self.model = model
        self.lang_chain = self.__make_lang_chain__()
        
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
        bm25.k = self.config.top_k
        return bm25
        
    def __load_hybrid_retreiver__(self) -> EnsembleRetriever:
        print(f"[RETRIEVER] making ensemble retriever...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm_25],
            weights=self.config.weights
        )
        return ensemble_retriever
    
    def __make_lang_chain__(self) -> RunnableSerializable:
        print(f"[RETRIEVER] make chaining with the retriever")
        template = """
        당신은 장소 추천 전문가로서 여행 계획 플래너가 여행 계획을 세울 때 참고할 수 있는 숙박 업소를 추출하는 역할을 맡았습니다.
        아래 [정보]를 바탕으로 질문에 친절하게 답해주세요. 안 좋은 정보라도 뽑아서 답하세요.
        [정보]에 없는 내용은 지어내지 말고 "정보가 없습니다"라고 답하세요.

        [정보]:
        {context}

        질문: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        ensemble_retriever = self.__load_hybrid_retreiver__() 
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        return rag_chain
    
    def retrieve(self, query: str):
        retriever = self.__load_hybrid_retreiver__()
        docs = retriever.get_relevant_documents(query)
        return docs
    
    def retrieve_with_model(self, query: str) -> str:
        response = self.lang_chain.invoke(query)

        print(f"질문: {query}")
        print(f"답변:\n{response}")
        return response