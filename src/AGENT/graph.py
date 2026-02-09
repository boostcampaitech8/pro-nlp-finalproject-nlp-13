from __future__ import annotations

from datetime import datetime
from typing import List, Any, Dict
import logging

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
from google import genai

from src.RAG.rag import Rag, RagConfig
from src.RAG.hybrid_retriever import HybridRetriever, RetrieverConfig

from src.AGENT.tool.weather import forecast_tool
from src.AGENT.prompt.prompt import Router, ChatBot, Summarizer, Validator
from src.AGENT.states import State, RouterDecision

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]%(message)s",
    datefmt="%H:%M:%S",
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

class LangGraph:
    DAYS = ["월", "화", "수", "목", "금", "토", "일"]
    ERROR_RESPONSE = "ㅋㅋㅋ 미안. 다시 한번 말해줄래?"
    
    def __init__(self):    
        chatbot = init_chat_model(
            model="dpo-final-policy",
            model_provider="openai",
            api_key="none",
            base_url="http://127.0.0.1:8080/v1",
            temperature=0,
            max_tokens=2048)
        
        llm = init_chat_model(
            "gpt-4o-mini",
            temperature=0,
        )
        summerized_llm = init_chat_model(
            "solar-mini",
            model_provider="upstage",
            temperature=0,
        )

        BM25_PATH = "src/results/db/bm25.pkl"
        DB_PATH = "src/results/db/chroma_db"

        ragconfig = RagConfig(db_path=DB_PATH)
        rag = Rag(config=ragconfig)
        db = rag.load()
        config = RetrieverConfig(db=db, pickle_path=BM25_PATH)
        retriever = HybridRetriever(config=config)
        
        memory = MemorySaver()
        client = genai.Client()

        self.chatbot = chatbot
        self.structured_llm = llm.with_structured_output(RouterDecision)
        self.summerized_llm = summerized_llm
        self.web_search = client
        self.retriever = retriever
        self.app = self._build_graph(checkpointer=memory)
        
    # nodes
    def _router_node(self, state: State) -> Dict[str, Any]:
        logger.info("[ROUTER] Router try to find the suitable agent for task...")        
        now = datetime.now()
        day_of_week = self.DAYS[now.weekday()]
        current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"
        question = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", Router.system), 
            ("human", Router.user),
        ])
        chain = prompt | self.structured_llm
        
        formatted = prompt.format_messages(
            current_time=current_time,
            days=self.DAYS,
            question=question,)
        logger.debug(f"[ROUTER] Router's system propmt is ::\n{formatted[-2].content}\nRouter's human propmt is ::\n{formatted[-1].content}")
        
        try:
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
            logger.warning(f"[ROUTER][ERROR] Router fallback due to error: {str(e)}")
         
        return {
            "rag_queries": rag_queries,
            "web_queries": web_queries,
            "weather_queries": weather_queries,
            "direct": direct,
            }
    
    # validation
    def _validate_node(self, state: State) -> Dict[str, Any]:
        logger.info(f"[VALIDATOR] Verify the information collected so far...")
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
        
        formatted = prompt.format_messages(
            question=question,
            documents=documents,
            web_results=web_results,
            weather_results=weather_results,)
        logger.debug(f"[VALIDATOR] Validator's system propmt is ::\n{formatted[-2].content}\nValidator's human propmt is ::\n{formatted[-1].content}")
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
            logger.warning(f"[VALIDATOR][ERROR] Validator fallback due to error: {str(e)}")
            rag_queries = None
            web_queries = None
            weather_queries = None
            
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
    def _rag_node(self, state: State) -> Dict[str, Any]:
        logger.info("[RAG] Search for documents...")
        queries = (state.get("rag_queries") or [state["messages"][-1].content])
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
    def _web_search_node(self, state: State) -> Dict[str, Any]:
        logger.info("[WEB-SEARCH] Search the web ...")
        queries = (state.get("web_queries") or [state["messages"][-1].content])
        web_results = []
        
        for query in queries:
            results = self.web_search.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,)
            web_results.append(results.text)

        return {"web_results": web_results}
    
    # weather node
    def _weather_node(self, state: State) -> Dict[str, Any]:
        logger.info(f"[WEATHER] Search for the weather...")
        days = state.get("weather_queries")
        weather_results = []
        for day in days:
            if isinstance(day, int):
                result = forecast_tool(city="부산", days=day)
                weather_results.append(result)
            else:
                logger.warning(f"[WEATHER][WARNING] The model's query for weather is not in number format. models's query: {day}")
                continue
            
        return {"weather_results": weather_results}

    # Answer Node
    def _chatbot_node(self, state: State) -> Dict[str, Any]:
        logger.info("[CHATBOT] Generate final answer...")
        
        now = datetime.now()
        day_of_week = self.DAYS[now.weekday()]
        current_time = f"{now.strftime('%Y년 %m월 %d일')} {day_of_week}요일"
        
        question = state["messages"][-1].content        
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
        chain = prompt | self.chatbot
        
        formatted = prompt.format_messages(
            current_time=current_time,
            question=question,
            context_text=context_text,
            web_text=web_text,
            weather_text=weather_text,
            )
        logger.debug(f"[CHATBOT] Chatbot's system propmt is ::\n{formatted[-2].content}\nChatbot's human propmt is ::\n{formatted[-1].content}")
        
        final_answer = ""
        message = []
        
        try:
            response = chain.invoke({
                "current_time":current_time,
                "question":question,
                "context_text":context_text,
                "web_text":web_text,
                "weather_text":weather_text,
            })
            final_answer = response.content
            message.append(response)
            
        except Exception as e:
            logger.warning(f"[CHATBOT][ERROR] Chatbot fallback due to error: {str(e)}")
            final_answer = self.ERROR_RESPONSE
        
        return {
            "final_answer": final_answer,
            "messages": message,
        }

    def _summarize_node(self, state: State) -> Dict[str, Any]:
        logger.info("[SUMMARIZER] Summarize the conversation...")

        question = state["messages"][-2].content
        bot_response = state["messages"][-1].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", Summarizer.system), 
            ("human", Summarizer.user),
        ])
        chain = prompt | self.summerized_llm
        new_summary = ""
        
        formatted = prompt.format_messages(
            question=question,
            bot_response=bot_response)
        logger.debug(f"[SUMMARIZER] Summarizer's system propmt is ::\n{formatted[-2].content}\nSummarizer's human propmt is ::\n{formatted[-1].content}")
        
        try: 
            response = chain.invoke({
                "question":question,
                "bot_response":bot_response,
            })
            new_summary = response.content
        
        except Exception as e:
            logger.warning(f"[SUMMARIZER][ERROR] Summarizer fallback due to error: {str(e)}")

        return {
                "summary": new_summary, 
                "documents": [],
                "web_results": [],
                "weather_results": [],
                }
            
    # routing
    def _route_nodes(self, state: State) -> List[str]:
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

    def _build_graph(self, checkpointer=None) -> CompiledStateGraph:
        workflow = StateGraph(State)

        workflow.add_node("router", self._router_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("web", self._web_search_node)
        workflow.add_node("weather", self._weather_node)
        workflow.add_node("validator", self._validate_node)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("summarize", self._summarize_node)

        workflow.add_edge(START, "router")

        intermediates = ["rag", "web", "weather", "chatbot"]
        workflow.add_conditional_edges(
            "router",
            self._route_nodes,
            intermediates
        )
        
        for node in ["rag", "web", "weather"]:
            workflow.add_edge(node, "validator")
        
        workflow.add_conditional_edges(
            "validator",
            self._route_nodes,
            intermediates
        )
        
        workflow.add_edge("chatbot", END)

        return workflow.compile(checkpointer=checkpointer)
    
    def run(self, message: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        result = self.app.invoke(inputs, config=config)
        return result.get("final_answer", self.ERROR_RESPONSE)