from __future__ import annotations

from states import State, RouterDecision
from typing import Literal, List
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from src.rag.rag import Rag, RagConfig
from src.rag.hybrid_retriever import HybridRetriever, RetrieverConfig
from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

VALIDATION_SYSTEM_PROMPT = """
당신은 부족한 정보를 파악하여 검색 쿼리를 생성하는 Query Generator입니다.
제공된 [사용자 질문]과 [검색 결과(rag, web, weather)]를 비교하여, 답변에 필요한 정보가 충분한지 판단하세요.
정보가 부족하다면 아래 규칙에 따라 추가 쿼리를 생성하고, JSON 형식으로만 출력하세요.

**쿼리 생성 규칙:**
1. **중복 제거:** 한 주제당 하나의 핵심 명사형 질문만 생성하세요. (예: "부산 호텔", "부산 숙소" 중 하나만 사용)
2. **카테고리 분류:**
   - **weather_queries:** 날씨, 기온, 강수량, 일출/일몰 등.
   - **rag_queries:** 장소, 숙박(호텔/모텔 등), 음식점, 카페, 관광지, 축제, 쇼핑, 액티비티.
   - **web_queries:** rag에 없는 정보, 최신성 정보(운영 시간, 실시간 티켓/기차표 시간 등).

**출력 형식:**
- 설명, 마크다운, 특수기호를 절대 포함하지 마세요.
- 오직 아래 JSON 데이터만 반환하세요.
- 쿼리가 없는 항목은 null을 사용하세요.

{{
  "rag_queries": ["질문1", "질문2"] 또는 null,
  "web_queries": ["질문1", "질문2"] 또는 null,
  "weather_queries": ["질문1", "질문2"] 또는 null,
  "reason": "부족한 정보에 대한 짧은 이유"
}}
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
        )

        BM25_PATH = "../../db/bm25"
        DB_PATH = "../../db/chroma_db"

        ragconfig = RagConfig(db_path=DB_PATH)
        rag = Rag(config=ragconfig)
        db = rag.load()
        config = RetrieverConfig(db=db, pickle_path=BM25_PATH)
        retriever = HybridRetriever(config=config)

        try:
            tavily = TavilyClient(api_key="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")
        except Exception:
            print("Tavily API 키가 없어서 웹 검색도 가짜(Mock)로 설정합니다.")
            class MockTavily:
                def search(self, query, max_results=5):
                    return [{"title": "테스트 뉴스", "url": "http://test.com", "content": "웹 검색 결과 예시입니다."}]
            tavily = MockTavily()

        self.llm = llm
        self.retriever = retriever
        self.tavily = tavily
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")
        
        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        
        try:
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
        if retried_count > 2: 
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
        print("[Weather]: 날씨 검색 중.. {queries}")
        return {"weather_results": ["화창하네~"]}

    # Answer Node
    def chatbot_node(self, state: State):
        print("[Chatbot]: 최종 답변 생성  중...")
        
        messages = state["messages"]
        query = state.get("query", "")
        # route = state.get("route", "direct")
        
        docs = state.get("documents", [])
        context_text = "\n\n".join([d["text"] for d in docs])
        
        webs = state.get("web_results", [])
        web_text = "\n\n".join([f"{w['title']}: {w['snippet']}" for w in webs])
        
        weather_results = state.get("weather_results", [])
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
        
        response = self.llm.invoke(prompt_messages)
        
        return {
            "final_answer": response.content,
            "messages": [response] 
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
            workflow.add_edge(node, "validator") 
        
        workflow.add_conditional_edges(
            "validator",
            self.route_nodes,
            intermediates
        )

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