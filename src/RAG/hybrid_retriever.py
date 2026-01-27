from typing import Dict
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pickle

class HybridRetriever:
    def __init__(
        self, 
        db: Chroma,
        pickle_path: str,
        model_name: str,
        top_k: int = 5,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 5},
        weights: list = [0.6, 0.4],
    ):
        self.model_name = model_name
        self.weights = weights
        self.dense_retriever = self.load_dense_retriever(db, search_type, search_kwargs)
        self.bm_25 = self.load_bm25_retriever(pickle_path, top_k)
        llm = ChatOpenAI(
            model=self.model_name, 
            temperature=0)
        self.model = llm
        self.lang_chain = self.make_lang_chain()
        # bm25, dense retriever 불러오기
        
    def load_dense_retriever(db, search_type, search_kwargs): # 동사로 이름 바꾸기
        print(f"[RETRIEVER] making dense retriever with TYPE: {search_type} and Args: {search_kwargs}")
        return db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs)
        
    def load_bm25_retriever(pickle_path, top_k):
        print(f"[RETRIEVER] making bm25 retriever... PATH is: {pickle_path}")
        with open(pickle_path, "rb") as f:
            bm25 = pickle.load(f)
        bm25.k = top_k
        return bm25
        
    def hybrid_retreiver(self) -> EnsembleRetriever: # 동사
        print(f"[RETRIEVER] making ensemble retriever...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm_25],
            weights=self.weights
        )
        
        return ensemble_retriever
    
    def make_lang_chain(self):
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
        
        ensemble_retriever = self.hybrid_retreiver() 
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return rag_chain
    
    def retrieve(self, query):
        response = self.lang_chain.invoke(query)

        print(f"질문: {query}")
        print(f"답변:\n{response}")
        return response