from __future__ import annotations

import os
from states import State, RouterDecision
from typing import List
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tavily import TavilyClient
from dotenv import load_dotenv

from src.rag.rag import Rag, RagConfig
from src.rag.hybrid_retriever import HybridRetriever, RetrieverConfig

from src.weather.weather_tools import forecast_tool
from datetime import datetime

load_dotenv()

VALIDATION_SYSTEM_PROMPT = """
당신은 검색 결과가 사용자의 질문을 해결하기에 충분한지 판단하는 **Sufficiency Validator(충족 여부 판단기)**입니다.
사용자 질문과 제공된 검색 결과(rag, web, weather)를 비교하여 판단하세요.

**★ 핵심 원칙 (가장 중요):**
1. **중복 검색 금지:** 이미 검색 결과에 답변할 수 있는 정보가 포함되어 있다면, 해당 카테고리의 쿼리는 반드시 `null`을 반환해야 합니다.
2. **완벽주의 금지:** 정보가 1개라도 확실하게 있다면 "충분하다"고 판단하세요. (예: 호텔이 하나라도 추천되었으면 추가 검색 불필요)
3. **불필요한 생성 금지:** 억지로 쿼리를 만들어내지 마세요.

**판단 기준:**
- **weather:** 질문한 날짜/지역의 기상 정보가 결과에 있는가? -> 있으면 `null`
- **rag:** 질문한 장소(숙소, 맛집 등)에 대한 정보가 1개 이상 있는가? -> 있으면 `null`
- **web:** 운영 시간, 가격 등 구체적 사실이 결과 텍스트에 포함되어 있는가? -> 있으면 `null`

**출력 형식 (JSON):**
{
  "rag_queries": ["쿼리"] 또는 null,
  "web_queries": ["쿼리"] 또는 null,
  "weather_queries": ["쿼리"] 또는 null,
  "reason": 짧은 이유
}
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

days = ["월", "화", "수", "목", "금", "토", "일"]
now = datetime.now()
day_of_week = days[now.weekday()]

current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"

ROUTER_SYSTEM_PROMPT = f"""
오늘 날짜는 다음과 같습니다: {current_time}
일주일은 다음 요일을 순서대로 포함합니다.: {days}
당신은 사용자 질문을 [rag, web, weather, direct]로 분류하는 Router입니다.
아래 규칙을 엄격히 준수하여 JSON을 생성하세요.

**★핵심 제약 사항 (위반 시 오답 처리):**
1. **개수 제한:** 각 카테고리(주제) 당 **가장 정확한 '단 하나(1개)'의 쿼리**만 생성하세요.
2. **중복 금지:** 같은 의미의 질문을 여러 번 쓰지 마세요. (유의어 나열 금지)
3. **날씨 포맷:** weather_queries는 반드시 숫자 형식이어야 합니다.

**분류 규칙:**
1. **rag:** 장소, 숙박, 맛집, 명소 정보 -> 리스트에 **핵심 키워드 1개**만.
2. **web:** 실시간 정보(교통편 시간표, 티켓 예매, 뉴스) -> 리스트에 **핵심 문장 1개**만.
3. **weather:** 날씨/기온 -> (내일=1, 오늘=0, 이틀후=2) 형식.
4. **direct:** 인사, 농담.

**출력 예시 (반드시 이 형태를 따를 것):**
User: "내일 부산 날씨랑 해운대 깨끗한 호텔 추천해주고 서울 가는 기차표 제일 빠른거 알려줘"
Output:
{{
  "rag_queries": ["해운대 깨끗한 호텔"], 
  "web_queries": ["서울행 부산 출발 기차표 최단시간"],
  "weather_queries": [1],
  "direct": null,
  "reason": 짧은 설명
}}
"""

CHATBOT_PROMPT = """
당신은 친절한 여행 가이드입니다.
사용자의 질문에 대해 아래 [정보]를 바탕으로 답변하세요.
정보가 없으면 답변하되, 진실된 정보만 답변하세요. 만들어낸 정보는 답변하지 않습니다.
출력은 무조건 줄글 형식으로 줍니다. json 이나 마크다운 형식으로 절대 주지 마십시오.
현재 질문은 다음과 같습니다: {query}

[참고 문서]\n{context_text}\n
[인터넷 검색 결과]\n{web_text}\n
[날씨]\n{weather_text}
"""

SUMMARIZE_PROMPT = """
지금까지의 요약: {summary}
새로운 대화:
{messages}

위 내용을 바탕으로 전체 대화 내용을 짧게 요약해줘.
"""

class LangGraph:
    def __init__(self):
        llm = init_chat_model(
            "gpt-4o-mini",
            # model_provider="upstage",
            temperature=0,
        )
        
        summerized_llm = init_chat_model(
            "solar-mini",
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
        
        #                 todo 타빌리 따로 빼던가?
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
        self.summerized_llm = summerized_llm
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")
        
        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content
        
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        # todo delete it
        print(f"[Router] {ROUTER_SYSTEM_PROMPT}\n question")
        
        try:
                    # todo delete it
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
            
                # todo delete it
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
        # todo delete it 
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
            
        # todo delete it
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
            results = self.tavily.search(
                query=query, 
                max_results=3,
                search_depth="advanced"
                )
            
            search_hits = results.get("results", []) if isinstance(results, dict) else results

            for r in search_hits:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet =  r.get("content") or r.get("snippet", "")
                if not url or url in existing_urls:
                    continue
                web_results.append({
                    "title": title,
                    "url": url,
                    "snippet": r.get("content") or r.get("snippet", ""),
                })

        return {"web_results": web_results}
    
    # weather node
    def weather_node(self, state: State):
        days = state.get("weather_queries")
        print(f"[Weather]: 날씨 검색 중.. {days}")
        weather_results = []
        for day in days:
            if isinstance(day, int):
                result = forecast_tool(city="부산", days=day)
                # todo delete it
                print(f"[Weather] the weather is. .. {day} and {result}")
                weather_results.append(result)
            else:
                                # todo delete it
                print(f"[Weather] the model's answer is not in int...")
                continue
            
        return {"weather_results": weather_results}

    # Answer Node
    def chatbot_node(self, state: State):
        print("[Chatbot]: 최종 답변 생성 중...")
        
        messages = state["messages"]
        query = state.get("query", "")
        
        docs = state.get("documents", [])
        context_text = "\n\n".join([d["text"] for d in docs])
        
        webs = state.get("web_results", [])
        web_text = "\n\n".join([f"{w['title']}: {w['snippet']}" for w in webs])
        
        weather_results = state.get("weather_results", [])
        weather_text = "\n".join(weather_results)
        prompt = CHATBOT_PROMPT.format(
            query=query,
            context_text=context_text,
            web_text=web_text,
            weather_text=weather_text
        )
        prompt_messages = [SystemMessage(content=prompt)] + messages
                        # todo delete it

        print(f"[CHATBOT] 정리하는 prompt: {prompt}")
        response = self.llm.invoke(prompt_messages)
        
        return {
            "final_answer": response.content,
            "messages": [response],
        }

    def summarize_node(self, state: State):
        summary = state.get("summary", "")
        messages = state["messages"]

        if len(messages) > 6:
            prompt = SUMMARIZE_PROMPT.format(
                summary=summary,
                messages=messages
            )
            response = self.summerized_llm.invoke(prompt)
            new_summary = response.content

            delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

            return {
                    "summary": new_summary, 
                    "messages": delete_messages, # 삭제 명령
                    # 초기화
                    "documents": [],
                    "web_results": [],
                    "weather_results": [],
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
    
    def run(self, message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        result = self.app.invoke(inputs, config=config)
        return result.get("final_answer", "답변을 생성하지 못했습니다.")