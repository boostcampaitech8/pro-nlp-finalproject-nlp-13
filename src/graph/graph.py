from __future__ import annotations

import os
from states import State, RouterDecision
from typing import Literal, List
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv

from src.rag.rag import Rag, RagConfig
from src.rag.hybrid_retriever import HybridRetriever, RetrieverConfig

load_dotenv()

VALIDATION_SYSTEM_PROMPT = """
당신은 Query Generator입니다. 사용자의 질문과 검색 결과를 비교하여 추가 정보가 필요한지 판단하세요.
결과는 **반드시** 아래 JSON 형식으로만 출력해야 합니다. (마크다운, 설명 금지)

**출력 예시:**
{
  "rag_queries": ["부산 호텔 추천", "해운대 맛집"],
  "web_queries": null,
  "weather_queries": ["부산 내일 날씨"],
  "reason": "숙소와 날씨 정보는 부족하여 추가 검색 필요"
}

**쿼리 생성 규칙:**
1. rag_queries: 장소, 숙박, 맛집, 관광 등
2. web_queries: 운영 시간, 실시간 예약, 뉴스 등
3. weather_queries: 날씨, 기온, 일출/일몰
4. 해당 사항이 없으면 null 을 입력하세요.
"""

VALIDATION_USER_PROMPT = """
사용자 질문:
{question}

rag 결과:
{documents}

web 결과:
{web_results}

weather 결과:
{weather_results}
"""

ROUTER_SYSTEM_PROMPT = """
당신은 사용자의 질문을 rag, web, weather, direct 중 하나 이상으로 분류하는 라우터입니다.

규칙:
- 단순 인사, 대화 요약, 잡담은 direct
- 날씨, 강수량, 일출/일몰 → weather
- 음식점, 카페, 관광지, 숙박, 축제, 쇼핑, 체험 → rag
- 운영시간, 교통시간, 최신 정보 → web

질문에 여러 주제가 있으면 각각 나눠 분류하세요.
쿼리는 짧은 명사 형태로 작성하세요.

반드시 JSON만 출력하세요.
None 대신 null 사용.

출력 형식:
{
  "rag_queries": [],
  "web_queries": [],
  "weather_queries": [],
  "direct": null,
  "reason": "짧은 한 줄"
}
"""

class LangGraph:
    def __init__(self):
        llm = init_chat_model(
            "solar-pro3",
            model_provider="upstage",
            temperature=0,
        )

        BM25_PATH = "db/bm25"
        DB_PATH = "db/chroma_db"

        ragconfig = RagConfig(db_path=DB_PATH)
        rag = Rag(config=ragconfig)
        db = rag.load()
        config = RetrieverConfig(db=db, pickle_path=BM25_PATH)
        retriever = HybridRetriever(config=config)

        try:
            api_key = os.getenv("TAVILY_API_KEY")
            tavily = TavilyClient(api_key=api_key)
        except Exception:
            print("Tavily API 키가 없어서 웹 검색도 가짜(Mock)로 설정합니다.")
            class MockTavily:
                def search(self, query, max_results=5):
                    return [{"title": "테스트 뉴스", "url": "http://test.com", "content": "웹 검색 결과 예시입니다."}]
            tavily = MockTavily()
            
        memory = MemorySaver()

        self.llm = llm
        self.retriever = retriever
        self.tavily = tavily
        self.app = self.build_graph(checkpointer=memory)
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")
        
        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        
        print(f"[Router] {ROUTER_SYSTEM_PROMPT}\n question")
        
        try:
            print("[Router] model is thinking...")
            decision = structured_llm.invoke(messages)
            rag_queries = decision.rag_queries
            web_queries = decision.web_queries
            weather_queries = decision.weather_queries
            direct = decision.direct
            reason = decision.route_reason
            
        except Exception as e:
            rag_queries = None
            web_queries = None
            weather_queries = None
            direct = question
            reason = f"Router fallback due to error: {type(e).__name__}"
            
        print(f"""[Router] Router analyzed the question. The result:\n[Rag]\n{rag_queries}\n[Web]\n{web_queries}\n[Weather]\n{weather_queries}\n[Direct]\n{direct}""")

        return {
            "rag_queries": rag_queries,
            "web_queries": web_queries,
            "weather_queries": weather_queries,
            "direct": direct,
            "route_reason": reason,
            }
    
    # validation
    def validate_node(self, state: State):
        print(f"[Validate] 지금까지 모은 정보를 검증합니다...")
        retried_count = state.get("retried_count", 0)
        if retried_count > 1: 
            return {
                "rag_queries": None,
                "web_queries": None,
                "weather_queries": None,
                "retried_count": 0
                }
        question = state.get("query") or state["messages"][-1].content
        structured_llm = self.llm.with_structured_output(RouterDecision)

        user_prompt = VALIDATION_USER_PROMPT.format(
            question=question,
            documents=state.get("documents", []),
            web_results=state.get("web_results", []),
            weather_results=state.get("weather_results", [])
        )
        messages = [
            SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        print(f"[Validate] the processing..: \n {VALIDATION_SYSTEM_PROMPT}")
        print(f"[Validate] the processing..: \n {user_prompt}")
        print("--------------------------------------------")        
        try:
            decision = structured_llm.invoke(messages)
            rag_queries = decision.rag_queries
            web_queries = decision.web_queries
            weather_queries = decision.weather_queries
            reason = decision.route_reason

        except Exception:
            rag_queries = None
            web_queries = None
            weather_queries = None
            
        print(f"""[Validator] Validator make another query. The result:\n[Rag]\n{rag_queries}\n[Web]\n{web_queries}\n[Weather]\n{weather_queries}""")
            
        if not rag_queries and not web_queries and not weather_queries:
            retried_count = 0
        else:
            retried_count += 1
        return {
            "rag_queries": rag_queries,
            "web_queries": web_queries,
            "weather_queries": weather_queries,
            "retried_count": retried_count
            }

    # RAG Retrieve Node
    def rag_node(self, state: State):
        print("[RAG]: 내부 문서 검색 중...")

        queries = (
            state.get("rag_queries")
            or [state.get("query")]
            or [state["messages"][-1].content]
        )
        
        rag_results = []
        builded_documents = state.get("documents", [])
        
        existing_ids = set()

        for d in builded_documents:
            meta = d.get("metadata", {})
            doc_id = meta.get("id")
            if doc_id:
                existing_ids.add(doc_id)

        for query in queries:
            docs = self.retriever.retrieve(query)
            for doc in docs:
                text = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {})
                doc_id = metadata.get("id")
                documents = [
                    {
                        "text": text,
                        "metadata": metadata,
                        "doc_id": doc_id
                    }
                ]
                
                if doc_id in existing_ids:
                    continue
                
            rag_results.extend(documents)

        return {"documents": rag_results, "route": "rag"}

    # WebSearch Node
    def web_search_node(self, state: State):
        print("[Web]: 웹 검색 중...")

        queries = (
            state.get("web_queries")
            or [state.get("query")]
            or [state["messages"][-1].content]
        )
        
        web_results = []
        builded_web_results = state.get("web_results", [])
        
        existing_urls = set()
        
        for web_reults in builded_web_results:
            url = web_reults.get("url")
            if url:
                existing_urls.add(url)

        for query in queries:
            print(f"[Web] 다음을 검색중입니다... {query}")       
            results = self.tavily.search(query=query, max_results=3)
            
            search_hits = results.get("results", []) if isinstance(results, dict) else results

            for r in search_hits:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet =  r.get("content") or r.get("snippet", "")
                snippet = snippet[:200] # 자르기
                if not url or url in existing_urls:
                    continue
                web_results.append({
                    "title": title,
                    "url": url,
                    "snippet": r.get("content") or r.get("snippet", ""),
                })

        return {"web_results": web_results}
    
    def weather_node(self, state: State):
        queries = (
            state.get("weather_queries")
            or [state.get("query")]
            or [state["messages"][-1].content]
        )
        print(f"[Weather]: 날씨 검색 중.. {queries}")
        return {"weather_results": ["화창하네~"]}

    # Answer Node
    def chatbot_node(self, state: State):
        print("[Chatbot]: 최종 답변 생성 중...")
        
        # messages = state["messages"]
        # query = state.get("query", "")
        
        # docs = state.get("documents", [])
        # context_text = "\n\n".join([d["text"] for d in docs])
        
        # webs = state.get("web_results", [])
        # web_text = "\n\n".join([f"{w['title']}: {w['snippet']}" for w in webs])
        
        # weather_results = state.get("weather_results", [])
        # weather_text = "\n".join(weather_results)
        
        # 빠른 답변을 위한 자른 버전 todo delete it by yhkim
        messages = state["messages"][-4:]
        query = state.get("query", "")

        docs = state.get("documents", [])[:3]
        context_text = "\n\n".join([d["text"][:500] for d in docs])

        webs = state.get("web_results", [])[:3]
        web_text = "\n\n".join([
            f"{w['title']}: {w['snippet'][:150]}" for w in webs
        ])

        weather_results = state.get("weather_results", [])[:3]
        weather_text = "\n".join(weather_results)


        system_prompt = f"""당신은 친절한 여행 가이드입니다.
    사용자의 질문에 대해 아래 [정보]를 바탕으로 답변하세요.
    정보가 없으면 답변하되, 진실된 정보만 답변하세요. 만들어낸 정보는 답변하지 않습니다.
    현재 질문은 다음과 같습니다: {query}
    
    [참고 문서]\n{context_text}
    [인터넷 검색 결과]\n{web_text}
    [날씨]\n{weather_text}
    """
        
        prompt_messages = [SystemMessage(content=system_prompt)] + messages
        print(f"[CHATBOT] 정리하는 prompt: {system_prompt}")
        response = self.llm.invoke(prompt_messages)
        
        return {
            "final_answer": response.content,
            "messages": [response],
            # 초기화
            "documents": [],
            "web_results": [],
            "weather_results": [],
        }

    def summarize_node(self, state: State):
        summary = state.get("summary", "")
        messages = state["messages"]

        if len(messages) > 6:
            prompt = f"""
            지금까지의 요약: {summary}
            새로운 대화:
            {messages}
            
            위 내용을 바탕으로 전체 대화 내용을 짧게 요약해줘.
            """

            response = self.llm.invoke(prompt)
            new_summary = response.content

            delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

            return {
                    "summary": new_summary, 
                    "messages": delete_messages # 삭제 명령
                }
            
        return {}

    # routing
    def route_nodes(self, state: State) -> List[str]:
        activated_nodes = []
        if state.get("rag_queries"):
            activated_nodes.append("rag")
        if state.get("web_queries"):
            activated_nodes.append("web")
        if state.get("weather_queries"):
            activated_nodes.append("weather")
        if state.get("direct") or not activated_nodes:
            return ["chatbot"]
        return activated_nodes
    
    def should_summarize(self, state: State):
        if len(state["messages"]) > 6:
            return "summarize"
        return "end"

    def build_graph(self, checkpointer=None):
        workflow = StateGraph(State)

        workflow.add_node("router", self.router_node)
        workflow.add_node("rag", self.rag_node)
        workflow.add_node("web", self.web_search_node)
        workflow.add_node("weather", self.weather_node)
        workflow.add_node("chatbot", self.chatbot_node)
        workflow.add_node("validator", self.validate_node)
        workflow.add_node("summarize", self.summarize_node)

        workflow.add_edge(START, "router")

        intermediates = ["rag", "web", "weather", "chatbot"]
        workflow.add_conditional_edges(
            "router",
            self.route_nodes,
            intermediates
        )
        
        for node in ["rag", "web", "weather"]:
            workflow.add_edge(node, "chatbot") # chatbot -> validator
        
        # todo change it by yhkim
        # workflow.add_conditional_edges(
        #     "validator",
        #     self.route_nodes,
        #     intermediates
        # )

        workflow.add_conditional_edges(
            "chatbot",        
            self.should_summarize,    
            {
                "summarize": "summarize",
                "end": END               
            }
        )

        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=checkpointer)
    
    def run(self, message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        result = self.app.invoke(inputs, config=config)
        return result.get("final_answer", "답변을 생성하지 못했습니다.")