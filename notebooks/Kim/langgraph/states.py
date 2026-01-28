from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, NotRequired, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

Route = Literal["direct", "rag", "web"]

class RagDocument(TypedDict):
    doc_id: NotRequired[str]          
    text: str                         
    score: NotRequired[float]         
    source: NotRequired[str]        
    metadata: NotRequired[Dict[str, Any]]


# WebSearch
class WebResult(TypedDict):
    title: str
    url: str
    snippet: NotRequired[str]  
    score: NotRequired[float]        
    published_date: NotRequired[str]  


# Router
class RouterDecision(BaseModel):
    route: Literal["direct", "rag", "web"] = Field(description="다음 실행할 단계")
    route_reason: str = Field(description="선택한 이유")

# Citation
class Citation(TypedDict):
    kind: Literal["rag", "web"]
    title: NotRequired[str]         
    url: NotRequired[str]       
    source: NotRequired[str]        
    doc_id: NotRequired[str]        
    quote: NotRequired[str]    
    score: NotRequired[float]


# Debug
class NodeError(TypedDict):
    node: str
    message: str
    detail: NotRequired[Dict[str, Any]]


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    summary: NotRequired[str]

    query: str
    user_info: NotRequired[Dict]

    # 라우팅 결정
    route: NotRequired[Route]
    route_reason: NotRequired[str] 

    # 검색 결과
    documents: NotRequired[List[RagDocument]]
    web_results: NotRequired[List[WebResult]]

    # 출력
    final_answer: NotRequired[str]
    citations: NotRequired[List[Citation]]

    # 운영/디버깅
    trace: NotRequired[List[str]]   
    errors: NotRequired[List[NodeError]]
    timings_ms: NotRequired[Dict[str, int]] 