from __future__ import annotations

import time

from datetime import datetime

from states import State, RouterDecision
from typing import List
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

from src.RAG.rag import Rag, RagConfig
from src.RAG.hybrid_retriever import HybridRetriever, RetrieverConfig

from AGENT.weather_tools import forecast_tool
from src.AGENT.prompt.prompt import Router, ChatBot, Summarizer, Validator

from google import genai

load_dotenv()
class LangGraph:
    DAYS = ["월", "화", "수", "목", "금", "토", "일"]
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
        self.structured_llm = llm.with_structured_output(RouterDecision)
        self.summerized_llm = summerized_llm
        self.client = client
        self.retriever = retriever
        self.app = self.build_graph(checkpointer=memory)
        
    # nodes
    def router_node(self, state: State):
        print("[Router]: 경로 탐색 중...")        
        now = datetime.now()
        day_of_week = self.DAYS[now.weekday()]
        current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"
        question = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", Router.system), 
            ("human", Router.user),
        ])
        chain = prompt | self.structured_llm
        
        # todo delete it for debug!!!!!!!
        formatted = prompt.format_messages(
            current_time=current_time,
            days=self.DAYS,
            question=question,)
        print("[Router][PROMPT] system: ")
        print(formatted[-2].content)
        print("[Router][PROMPT] human: ")
        print(formatted[-1].content)
        
        try:
                    # todo delete it
            print("[Router] model is thinking...")
            decision = chain.invoke({
                "current_time": current_time,
                "days": self.DAYS,    
                "question": question   
            })
            rag_queries = decision.rag_queries
            web_queries = decision.web_queries
            weather_queries = decision.weather_queries
            direct = decision.direct
            
        except Exception as e:
            rag_queries = None
            web_queries = None
            weather_queries = None
            direct = question
            print(f"[Router][ERROR] router fallback due to error: {str(e)}")
            
                # todo delete it
        print(f"""[Router] Router analyzed the question. The result:\n[Rag]\n{rag_queries}\n[Web]\n{web_queries}\n[Weather]\n{weather_queries}\n[Direct]\n{direct}""")

        return {
            "rag_queries": rag_queries,
            "web_queries": web_queries,
            "weather_queries": weather_queries,
            "direct": direct,
            }
    
    # validation
    def validate_node(self, state: State):
        print(f"[Validate] 지금까지 모은 정보를 검증합니다...")
        retried_count = state.get("retried_count", 0)
        question = state["messages"][-1].content
        documents = state.get("documents", []),
        web_results = state.get("web_results", []),
        weather_results = state.get("weather_results", [])
    
        if retried_count > 1: 
            return {
                "rag_queries": None,
                "web_queries": None,
                "weather_queries": None,
                "retried_count": 0
                }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", Validator.system), 
            ("human", Validator.user),
        ])
        chain = prompt | self.structured_llm
        
        # todo delete it for debug!!!!!!!
        formatted = prompt.format_messages(
            question=question,
            documents=documents,
            web_results=web_results,
            weather_results=weather_results,)
        print("[Validator][PROMPT] system: ")
        print(formatted[-2].content)
        print("[Validator][PROMPT] human: ")
        print(formatted[-1].content)
        try:
            decision = chain.invoke({
                "question":question,
                "documents":documents,
                "web_results":web_results,
                "weather_results":weather_results,
            })
            rag_queries = decision.rag_queries
            web_queries = decision.web_queries
            weather_queries = decision.weather_queries

        except Exception as e:
            print(f"[Validator][ERROR] validator fallback due to error: {str(e)}")
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
            or [state["messages"][-1].content]
        )
        builded_documents = state.get("documents", [])

        rag_results = []
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
                if doc_id in existing_ids:
                    continue
                documents = [
                    {
                        "text": text,
                        "metadata": metadata,
                        "doc_id": doc_id
                    }
                ]
                
                rag_results.extend(documents)
        return {"documents": rag_results}

    # WebSearch Node
    def web_search_node(self, state: State):
        print("[Web]: 웹 검색 중...")

        queries = (
            state.get("web_queries")
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
        print(f"[Weather]: 날씨 검색 중..")
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
        
        now = datetime.now()
        day_of_week = self.DAYS[now.weekday()]
        current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"
        
        question = state["messages"][-1].content
        summary = state.get("summary", "")
        
        docs = state.get("documents", [])
        context_text = "\n\n".join([d["text"] for d in docs])
        
        webs = state.get("web_results", [])
        web_text = "\n".join(webs)
        
        weathers = state.get("weather_results", [])
        weather_text = "\n".join(weathers)
             
        prompt = ChatPromptTemplate.from_messages([
            ("system", ChatBot.system), 
            ("human", ChatBot.user),
        ])
        chain = prompt | self.llm
        
        # todo delete it for debug!!!!!!!
        formatted = prompt.format_messages(
            current_time=current_time,
            question=question,
            context_text=context_text,
            web_text=web_text,
            weather_text=weather_text,
            summary=summary,)
        print("[Chatbot][PROMPT] system: ")
        print(formatted[-2].content)
        print("[Chatbot][PROMPT] human: ")
        print(formatted[-1].content)
        
        final_answer = ""
        message = []
        
        try:
            response = chain.invoke({
                "current_time":current_time,
                "question":question,
                "context_text":context_text,
                "web_text":web_text,
                "weather_text":weather_text,
                "summary":summary,
            })
            final_answer = response.content
            message.append(response)
            
        except Exception as e:
            print(f"[Chatbot][ERROR] chatbot fallback due to error: {str(e)}")
            final_answer = "ㅋㅋㅋ 미안. 다시 한번 말해줄래?"
        
        return {
            "final_answer": final_answer,
            "messages": message,
        }

    def summarize_node(self, state: State):
        print("[Summarize]: 대화 요약 중...")
        print("[Summarize]: 메시지 확인...\n", state["messages"])

        question = state["messages"][-2].content
        bot_response = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", Summarizer.system), 
            ("human", Summarizer.user),
        ])
        chain = prompt | self.llm
        new_summary = ""
        
        # todo delete it for debug!!!!!!!
        formatted = prompt.format_messages(
            question=question,
            bot_response=bot_response)
        print("[Summarize][PROMPT] system: ")
        print(formatted[-2].content)
        print("[Summarize][PROMPT] human: ")
        print(formatted[-1].content)
        
        try: 
            response = chain.invoke({
                "question":question,
                "bot_response":bot_response,
            })
            new_summary = response.content
        
        except Exception as e:
            print(f"[Summarize][ERROR] Summarize fallback due to error: {str(e)}")

        print(f"[Summarize] And answer: {new_summary}")
            
        return {
                "summary": new_summary, 
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