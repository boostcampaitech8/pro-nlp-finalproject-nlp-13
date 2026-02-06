from __future__ import annotations

import time

from states import State, RouterDecision
from typing import List
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain_core.prompts import load_prompt
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

from src.rag.rag import Rag, RagConfig
from src.rag.hybrid_retriever import HybridRetriever, RetrieverConfig

from src.weather.weather_tools import forecast_tool
from datetime import datetime

from google import genai

load_dotenv()
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
        
        memory = MemorySaver()
        client = genai.Client()

        self.llm = llm
        self.client = client
        self.retriever = retriever
        self.app = self.build_graph(checkpointer=memory)
        self.summerized_llm = summerized_llm
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")
        
        days = ["월", "화", "수", "목", "금", "토", "일"]
        now = datetime.now()
        day_of_week = days[now.weekday()]

        current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"

        structured_llm = self.llm.with_structured_output(RouterDecision)
        question = state.get("query") or state["messages"][-1].content
        
        router_system_prompt = load_prompt("./prompt/router_system_prompt.yaml")
        prompt = router_system_prompt.format(
            current_time = current_time,
            days = days
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]
        # todo delete it
        print(f"[Router] {prompt}\n question")
        
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
        validation_system_prompt = load_prompt("./prompt/val_system_prompt.yaml")
        validation_user_prompt = load_prompt("./prompt/val_user_prompt.yaml")

        user_prompt = validation_user_prompt.format(
            question=question,
            documents=state.get("documents", []),
            web_results=state.get("web_results", []),
            weather_results=state.get("weather_results", [])
        )
        messages = [
            SystemMessage(content=validation_system_prompt),
            HumanMessage(content=user_prompt),
        ]
        # todo delete it 
        print(f"[Validate] the processing..: \n {validation_system_prompt}")
        print(f"[Validate] the processing..: \n {user_prompt}")
        print("--------------------------------------------")        
        try:
            decision = structured_llm.invoke(messages)
            rag_queries = decision.rag_queries
            web_queries = decision.web_queries
            weather_queries = decision.weather_queries

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
        for query in queries:
            results = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,)
            web_results.append(results.text)
            # time.sleep(4) # unlock it for debug by yhkim todo

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
        query = state["messages"][-1].content
        
        docs = state.get("documents", [])
        context_text = "\n\n".join([d["text"] for d in docs])
        
        webs = state.get("web_results", [])
        web_text = "\n".join(webs)
        
        weathers = state.get("weather_results", [])
        weather_text = "\n".join(weathers)
        
        chatbot_prompt = load_prompt("./prompt/chatbot_prompt.yaml")
        
        prompt = chatbot_prompt.format(
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
        messages = state["messages"]
        
        query = state["messages"][-1].content
        
        docs = state.get("documents", [])
        context_text = "\n\n".join([d["text"] for d in docs])
        
        webs = state.get("web_results", [])
        web_text = "\n".join(webs)
        
        weathers = state.get("weather_results", [])
        weather_text = "\n".join(weathers)
        
        summarize_prompt = load_prompt("./prompt/summarize_prompt.yaml")
        prompt = summarize_prompt.format(
            query=query,
            context_text=context_text,
            web_text=web_text,
            weather_text=weather_text
        )
        
        response = self.summerized_llm.invoke(prompt)
        new_summary = response.content
        delete_messages = []
        summary_message = SystemMessage(content=new_summary)
        
        print(f"[SUMMARIZE] message count: {len(messages)} make summarize: whole prompt: {prompt}")
        print(f"[SUMMARIZE] And answer: {new_summary}")

        if len(messages) > 6:
            delete_messages = [
                RemoveMessage(id=m.id)
                for m in messages[:3]
            ]
            
        return {
                "summary": new_summary, 
                "messages": delete_messages + [summary_message],
                "documents": [],
                "web_results": [],
                "weather_results": [],
                }
            
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
        workflow.add_node("validator", self.validate_node)
        workflow.add_node("chatbot", self.chatbot_node)
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
        
        workflow.add_edge("chatbot", "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=checkpointer)
    
    def run(self, message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        result = self.app.invoke(inputs, config=config)
        return result.get("final_answer", "답변을 생성하지 못했습니다.")