from __future__ import annotations

from states import State, RouterDecision
from typing import Literal
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from src.RAG.rag import Rag, RagConfig
from src.RAG.hybrid_retriever import HybridRetriever, RetrieverConfig
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
import json

load_dotenv()

EVALUATOR_PROMPT = """
You are an evaluation agent.

User Question:
{question}

Current Draft Answer:
{draft_answer}

RAG Results:
{documents}

WEB Results:
{web_results}

Decide whether the draft answer sufficiently answers the user question.

Rules:
- sufficient: true if answer is complete and up-to-date.
- need:
  - "none" if enough
  - "rag" if more internal knowledge is needed
  - "web" if more real-time info is needed
  - "both" if both missing

Return ONLY JSON:

{{
  "sufficient": true/false,
  "need": "none|rag|web|both",
  "reason": "short explanation"
}}
"""


class LangGraph:
    def __init__(self):
        llm = init_chat_model(
            "solar-mini",
            model_provider="upstage",
        )
        
        DB_PATH = "db/chroma_db"
        BM25_PATH = "db/bm25"
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        ragconfig = RagConfig(db_path=DB_PATH, embeddings=embeddings)
        rag = Rag(config=ragconfig)
        db = rag.load()
        config = RetrieverConfig(db = db, pickle_path = BM25_PATH) # 다시 생각해보기
        retreiver = HybridRetriever(config=config)
        
        try:
            tavily = TavilyClient(api_key="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")
        except Exception:
            print("Tavily API 키가 없어서 웹 검색도 가짜(Mock)로 설정합니다.")
            class MockTavily:
                def search(self, query, max_results=5):
                    return [{"title": "테스트 뉴스", "url": "http://test.com", "content": "웹 검색 결과 예시입니다."}]
            tavily = MockTavily()
        
        self.llm = llm
        self.retriever = retreiver
        self.tavily = tavily
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")
        
        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content

        try:
            decision = structured_llm.invoke(question)
            route = decision.route
            reason = decision.route_reason
        except Exception as e:
            route = "rag"
            reason = f"Router fallback due to error: {type(e).__name__}"

        return {"route": route, "route_reason": reason}
    
    # 재평가 EVALUATOR
    def evaluator_node(self, state: State):
        print("[Evaluator]: 답변 재평가 중...")
        question = state.get("query") or state["messages"][-1].content
        prompt = EVALUATOR_PROMPT.format(
            question=question,
            draft_answer=state.get("draft_answer", ""),
            documents=state.get("documents", []),
            web_results=state.get("web_results", []),
        )

        messages = [
            SystemMessage(content="You are a strict evaluator. Return JSON only."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        # --- 여기부터 핵심 ---
        try:
            decision = json.loads(response.content)
        except Exception:
            # JSON 깨지면 기본값
            decision = {
                "sufficient": True,
                "need": "none",
                "reason": "json parse failed"
            }

        sufficient = decision.get("sufficient", True)
        need = decision.get("need", "none")

        return {
            "sufficient": sufficient,
            "next_route": need,
        }
        
        
    def evaluator_router(self, state: State):
        if state["sufficient"]:
            return "end"

        if state["next_route"] == "rag":
            return "rag_node"

        if state["next_route"] == "web":
            return "web_node"

        if state["next_route"] == "rag-web":
            return "rag_web"

        return "end"
    

    # RAG Retrieve Node
    def rag_node(self, state: State):
        print("[RAG]: 내부 문서 검색 중...")

        query = state.get("query") or state["messages"][-1].content
        docs = self.retriever.retrieve(query)

        documents = []
        for doc in docs:
            documents.append({
                "text": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {}),
            })
        
        return {"documents": documents}

    # WebSearch Node
    def web_search_node(self, state: State):
        print("[Web]: 웹 검색 중...")

        query = state.get("query") or state["messages"][-1].content
        results = self.tavily.search(query=query, max_results=5)
        
        search_hits = results.get("results", []) if isinstance(results, dict) else results

        web_results = []
        for r in search_hits:
            web_results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
            })

        return {"web_results": web_results}
    
    # RAG - WebSearch Node
    def rag_web_node(self, state: State):
        print("[Rag-Web]: 웹 검색과 내부 문서 검색 중...")
        query = state.get("query") or state["messages"][-1].content
        # RAG
        docs = self.retriever.retrieve(query)
        documents = [{
            "text": getattr(doc, "page_content", str(doc)),
            "metadata": getattr(doc, "metadata", {}),
        } for doc in docs]

        # WEB
        results = self.tavily.search(query=query, max_results=3)
        search_hits = results.get("results", []) if isinstance(results, dict) else results
        web_results = [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content") or r.get("snippet", ""),
        } for r in search_hits]
        
        print(f"[RAG-WEB] checking ${documents[0]} and ${web_results}")

        return {
            "documents": documents,
            "web_results": web_results,
            "route": "rag-web"
        }

    # Answer Node
    def chatbot_node(self, state: State):
        print("[Chatbot]: 최종 답변 생성  중...")
        
        messages = state["messages"]
        route = state.get("route", "direct")
        
        context = ""
        
        if route == "rag":
            docs = state.get("documents", [])
            context_text = "\n\n".join([d["text"] for d in docs])
            context = f"[참고 문서]\n{context_text}"
            
        elif route == "web":
            webs = state.get("web_results", [])
            web_text = "\n\n".join([f"{w['title']}: {w['snippet']}" for w in webs])
            context = f"[인터넷 검색 결과]\n{web_text}"
        
        elif route == "rag-web":
            docs = state.get("documents", [])
            webs = state.get("web_results", [])

            doc_text = "\n\n".join([d["text"] for d in docs])
            web_text = "\n\n".join([f"{w['title']}: {w['snippet']}" for w in webs])

            context = f"""
            [참고 문서]
            {doc_text}

            [인터넷 검색 결과]
            {web_text}
            """
        

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

    # edge
    def get_next_step(self, state: State) -> Literal["rag", "web", "chatbot"]:
        route = state["route"]
        
        if route == "rag":
            return "rag"      # rag_node
        elif route == "web":
            return "web"      # web_search_node
        # elif route == "rag-web":
        #     return "rag-web"
        else:
            return "chatbot"  # direct인 경우

    def should_summarize(self, state: State):
        if len(state["messages"]) > 6:
            return "summarize"
        return "end"

    def build_graph(self, checkpointer=None) -> any:
        workflow = StateGraph(State)

        workflow.add_node("router", self.router_node)
        workflow.add_node("rag", self.rag_node)
        workflow.add_node("web", self.web_search_node)
        workflow.add_node("evaluator", self.evaluator_node)
        # workflow.add_node("rag-web", self.rag_web_node)
        workflow.add_node("chatbot", self.chatbot_node)
        workflow.add_node("summarize", self.summarize_node)

        workflow.add_edge(START, "router")

        workflow.add_conditional_edges(
            "router",
            self.get_next_step,   
            {            
                "rag": "rag",
                "web": "web",
                # "rag-web": "rag-web",
                "chatbot": "chatbot"
            }
        )

        workflow.add_edge("rag", "chatbot")
        workflow.add_edge("web", "chatbot")
        # workflow.add_edge("rag-web", "chatbot")
        
        workflow.add_edge("chatbot", "evaluator")

        workflow.add_conditional_edges(
            "chatbot",        
            self.should_summarize,    
            {
                "summarize": "summarize",
                "end": END               
            }
        )
        
        workflow.add_conditional_edges(
            "evaluator",
            self.evaluator_router,
            {
                "rag_node": "rag",
                "web_node": "web",
                "end": END
            }
        )


        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=checkpointer)