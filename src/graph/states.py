from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, NotRequired, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["direct", "rag", "web", "weather"]

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
    # route: Literal["direct", "rag", "web", "evaluator"] = Field(description="다음 실행할 단계")
    rag_queries: Optional[List[str]] = None
    web_queries: Optional[List[str]] = None
    weather_queries: Optional[List[int]] = None
    direct: Optional[str] = None
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
    
    # 라우팅 결정 todo delete it 
    route: NotRequired[List[Route]]
    
    # 라우팅 결정
    rag_queries: Optional[List[str]] = None
    web_queries: Optional[List[str]] = None
    weather_queries: Optional[List[str]] = None
    direct: Optional[str] = None
    route_reason: NotRequired[str]
    
    # 재검색 시도
    retried_count: NotRequired[int]

    # 검색 결과
    documents: NotRequired[List[RagDocument]]
    web_results: NotRequired[List[WebResult]]
    weather_results: NotRequired[List[List[str]]]

    # 출력
    final_answer: NotRequired[str]
    citations: NotRequired[List[Citation]]

    # 운영/디버깅
    trace: NotRequired[List[str]]   
    errors: NotRequired[List[NodeError]]
    timings_ms: NotRequired[Dict[str, int]] 