import os
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv, find_dotenv

if not load_dotenv(find_dotenv()):
    print("⚠️ .env 파일을 찾을 수 없습니다.")


class RagDocument(TypedDict):
    doc_id: NotRequired[str]
    text: str
    metadata: NotRequired[Dict[str, Any]]

class WebResult(TypedDict):
    title: str
    url: str
    snippet: NotRequired[str]

class RouterDecision(BaseModel):
    route: Literal["direct", "rag", "web"] = Field(description="다음 실행할 단계")
    route_reason: str = Field(description="선택한 이유")

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    summary: NotRequired[str]
    query: str
    
    route: NotRequired[Literal["direct", "rag", "web"]]
    route_reason: NotRequired[str]
    
    documents: NotRequired[List[RagDocument]]
    web_results: NotRequired[List[WebResult]]
    
    final_answer: NotRequired[str]


class TravelChatbotGraph:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
            

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=self.openai_api_key,
            temperature=0
        )
        os.environ["TAVILY_API_KEY"] = ""                                   # Tavily API 키 임시로 넣는 칸
        self.tavily_tool = TavilySearchResults(max_results=3)



        self.memory = MemorySaver()
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(State)

        workflow.add_node("router", self.router_node)
        workflow.add_node("rag", self.rag_node)
        workflow.add_node("web", self.web_search_node)
        workflow.add_node("chatbot", self.chatbot_node)
        workflow.add_node("summarize", self.summarize_node)

        workflow.add_edge(START, "router")
        
        workflow.add_conditional_edges(
            "router",
            self.get_next_step,
            {
                "rag": "web", # rag
                "web": "web",
                "direct": "web" # chatbot
            }
        )

        workflow.add_edge("rag", "chatbot")
        workflow.add_edge("web", "chatbot")

        workflow.add_conditional_edges(
            "chatbot",
            self.should_summarize,
            {
                "summarize": "summarize",
                "end": END
            }
        )
        
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.memory)


    def router_node(self, state: State):
        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content
        
        try:
            decision = structured_llm.invoke(question)
            route = decision.route
            reason = decision.route_reason
        except Exception:
            route = "direct"
            reason = "Fallback due to error"
            
        return {"route": route, "route_reason": reason}

    def rag_node(self, state: State):
        documents = [
            {"text": "부산 여행 추천: 해운대와 광안리는 필수 코스입니다. 돼지국밥이 유명합니다.", "metadata": {"source": "mock_db"}},
            {"text": "제주도 여행: 성산일출봉과 우도가 아름답습니다.", "metadata": {"source": "mock_db"}}
        ]
        return {"documents": documents}

    def web_search_node(self, state: State):
        query = state.get("query")
        if not query:
            query = state["messages"][-1].content

        # 터미널서 tavily 작동 확인 디버깅
        print(f"🔥 [Tavily 작동] 검색어: {query}")    
        
        try:
            # 리스트 형태로 결과가 반환됨 [{'url':..., 'content':...}]
            search_results = self.tavily_tool.invoke(query)
        except Exception as e:
            print(f"검색 에러: {e}")
            search_results = []

        web_results = []
        for res in search_results:
            web_results.append({
                "title": res.get("url", "검색 결과"), 
                "url": res.get("url", ""),
                "snippet": res.get("content", "") 
            })
            
        return {"web_results": web_results}

    def chatbot_node(self, state: State):
        messages = state["messages"]
        route = state.get("route", "direct")
        
        context = ""
        if route == "rag":
            docs = state.get("documents", [])
            context_text = "\n".join([d["text"] for d in docs])
            context = f"[참고 문서]\n{context_text}"
        elif route == "web":
            webs = state.get("web_results", [])
            web_text = "\n".join([f"{w['title']}: {w['snippet']}" for w in webs])
            context = f"[웹 검색 결과]\n{web_text}"

        system_prompt = f"""당신은 친절한 여행 가이드입니다.
사용자의 질문에 대해 아래 [정보]를 바탕으로 답변하세요.
정보가 없으면 아는 대로 답변하세요.

{context}
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
            새로운 대화: {messages}
            위 내용을 바탕으로 전체 대화 내용을 짧게 요약해줘.
            """
            response = self.llm.invoke(prompt)
            new_summary = response.content
            delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
            
            return {"summary": new_summary, "messages": delete_messages}
        return {}


    def get_next_step(self, state: State) -> Literal["rag", "web", "chatbot"]:
        return state["route"]

    def should_summarize(self, state: State):
        if len(state["messages"]) > 6:
            return "summarize"
        return "end"

    
    def run(self, message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "messages": [("user", message)],
            "query": message
        }
        
        result = self.app.invoke(inputs, config=config)
        return result.get("final_answer", "답변을 생성하지 못했습니다.")