
# nodes
def router_node(state: State):
    print("[Router]: 경로 탐색 중...")
    
    structured_llm = llm.with_structured_output(RouterDecision)
    question = state.get("query") or state["messages"][-1].content

    try:
        decision = structured_llm.invoke(question)
        route = decision.route
        reason = decision.route_reason
    except Exception as e:
        route = "rag"
        reason = f"Router fallback due to error: {type(e).__name__}"

    return {"route": route, "route_reason": reason}

# RAG Retrieve Node
def rag_node(state: State):
    print("[RAG]: 내부 문서 검색 중...")

    query = state.get("query") or state["messages"][-1].content
    docs = retriever.invoke(query)

    documents = []
    for d in docs:
        documents.append({
            "text": getattr(d, "page_content", str(d)),
            "metadata": getattr(d, "metadata", {}),
        })
    
    return {"documents": documents}

# WebSearch Node
def web_search_node(state: State):
    print("[Web]: 웹 검색 중...")

    query = state.get("query") or state["messages"][-1].content
    results = tavily.search(query=query, max_results=5)
    
    search_hits = results.get("results", []) if isinstance(results, dict) else results

    web_results = []
    for r in search_hits:
        web_results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content") or r.get("snippet", ""),
        })

    return {"web_results": web_results}

# Answer Node
def chatbot_node(state: State):
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
    

    system_prompt = f"""당신은 친절한 여행 가이드입니다.
사용자의 질문에 대해 아래 [정보]를 바탕으로 답변하세요.
정보가 없으면 아는 대로 답변하세요.

{context}
"""
    
    prompt_messages = [SystemMessage(content=system_prompt)] + messages
    
    response = llm.invoke(prompt_messages)
    
    return {
        "final_answer": response.content,
        "messages": [response] 
    }

def summarize_node(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    if len(messages) > 6:
        prompt = f"""
        지금까지의 요약: {summary}
        새로운 대화:
        {messages}
        
        위 내용을 바탕으로 전체 대화 내용을 짧게 요약해줘.
        """

        response = llm.invoke(prompt)
        new_summary = response.content

        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

        return {
                "summary": new_summary, 
                "messages": delete_messages # 삭제 명령
            }
        
    return {}

# edge
def get_next_step(state: State) -> Literal["rag", "web", "chatbot"]:
    route = state["route"]
    
    if route == "rag":
        return "rag"      # rag_node
    elif route == "web":
        return "web"      # web_search_node
    else:
        return "chatbot"  # direct인 경우

def should_summarize(state: State):
    if len(state["messages"]) > 6:
        return "summarize"
    return "end"

def build_graph(checkpointer=None) -> any:
    workflow = StateGraph(State)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web", web_search_node)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        get_next_step,   
        {            
            "rag": "rag",
            "web": "web",
            "chatbot": "chatbot" 
        }
    )

    workflow.add_edge("rag", "chatbot")
    workflow.add_edge("web", "chatbot")

    workflow.add_conditional_edges(
        "chatbot",        
        should_summarize,    
        {
            "summarize": "summarize",
            "end": END               
        }
    )

    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)