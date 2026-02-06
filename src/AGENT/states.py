from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import Annotated, NotRequired, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# RAG
class RagDocument(TypedDict):
    doc_id: NotRequired[str]          
    text: str                         
    score: NotRequired[float]         
    source: NotRequired[str]        
    metadata: NotRequired[Dict[str, Any]]

# Router, Validator
class RouterDecision(BaseModel):
    rag_queries: Optional[List[str]] = None
    web_queries: Optional[List[str]] = None
    weather_queries: Optional[List[int]] = None
    direct: Optional[str] = None

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    summary: NotRequired[str]

    user_info: NotRequired[Dict]
    
    # 라우팅 결정
    rag_queries: Optional[List[str]] = None
    web_queries: Optional[List[str]] = None
    weather_queries: Optional[List[str]] = None
    direct: Optional[str] = None
    
    # 재검색 시도
    retried_count: NotRequired[int]

    # 검색 결과
    documents: NotRequired[List[RagDocument]]
    web_results: NotRequired[List[str]]
    weather_results: NotRequired[List[List[str]]]

    # 출력
    final_answer: NotRequired[str]