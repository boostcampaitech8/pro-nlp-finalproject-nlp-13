import streamlit as st
import time
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.graph.graph import LangGraph
from dotenv import load_dotenv

load_dotenv()

# 페이지
st.set_page_config(
    page_title="여행 챗봇",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .stSpinner > div {
        border-top-color: #4B9CFF !important;
    }
    .stButton > button {
        width: 100%;
        height: auto;
        padding-top: 15px;
        padding-bottom: 15px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# 세션 초기화

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
    
if "langgraph_app" not in st.session_state:
    memory = MemorySaver()
    langgraph = LangGraph()
    app = langgraph.build_graph(checkpointer=memory)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    st.session_state.langgraph_app = app
    st.session_state.langgraph_config = config

# API 키 세션 상태 초기화
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None

def format_history_for_gemini(messages):
    history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    return history


# 로그인 화면
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("✨ Login")
        input_username = st.text_input("Username")
        input_password = st.text_input("Password", type="password")
        
        if st.button("로그인", type="primary", use_container_width=True):
            if input_username == "admin" and input_password == "boostcamp":
                st.session_state.logged_in = True
                st.session_state.username = input_username
                st.session_state.current_chat_id = None
                st.rerun()
            else:
                st.error("잘못된 사용자명 또는 비밀번호입니다.")
    st.stop()


# 로그인 후 화면

# 사이드바
with st.sidebar:
    st.title("여행 챗봇")
    
    # [변경] API 키 입력창 추가 (가장 위쪽)
    st.markdown("### API 설정")

    api_key_input = st.text_input(
        "Google API Key", 
        type="password",  # 입력 시 점(•)으로 가려짐
        placeholder="AIzaSy...",
        value=st.session_state.api_key
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        # genai.configure(api_key=st.session_state.api_key)
        
        st.success("API 키가 적용되었습니다!")
    else:
        st.caption("키를 입력하고 Enter를 누르세요.")


    
    st.divider() # 
        
    if st.button("➕ 새 채팅", use_container_width=True, type="primary"):
        st.session_state.current_chat_id = None
        st.rerun()
    
    st.markdown("### 최근 대화")
    chat_ids = list(st.session_state.chat_history.keys())
    if not chat_ids:
        st.caption("대화 기록이 없습니다.")
    else:
        for chat_id in reversed(chat_ids):
            item = st.session_state.chat_history[chat_id]
            title = item['title']
            prefix = "🟢 " if chat_id == st.session_state.current_chat_id else ""
            
            col_btn, col_del = st.columns([0.85, 0.15])
            with col_btn:
                if st.button(f"{prefix}{title}", key=f"btn_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
                    
    st.markdown("---")
    st.caption(f"Logged in as: {st.session_state.username}") 
    if st.button("로그아웃"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_chat_id = None
        st.session_state.langgraph_app = None
        st.session_state.langgraph_config = None
        st.session_state.api_key = "" # 로그아웃 시 키도 삭제
        st.rerun()


# 챗봇 메인 화면

if st.session_state.current_chat_id is None:
    messages = []
else:
    if st.session_state.current_chat_id not in st.session_state.chat_history:
        st.session_state.current_chat_id = None
        messages = []
    else:
        messages = st.session_state.chat_history[st.session_state.current_chat_id]['messages']

button_prompt = None

if not messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"## <span style='background: linear-gradient(to right, #4285F4, #9B72CB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>안녕하세요, {st.session_state.username}님</span>", unsafe_allow_html=True)
    st.markdown("### 어디로 여행을 떠나고 싶으신가요?")
    
    # API 키가 없을 때 안내 메시지
    if not st.session_state.api_key:
        st.info("**Google API Key** 미입력")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🌍 여행 계획\n\n파리 여행 일정 짜줘"): button_prompt = "파리 여행 일정 짜줘"
    with c2:
        if st.button("✈️ 항공권 검색\n\n서울-뉴욕 항공편"): button_prompt = "서울에서 뉴욕 가는 항공편 찾아줘"
    with c3:
        if st.button("🏨 호텔 추천\n\n도쿄 가성비 호텔"): button_prompt = "도쿄에서 가성비 좋은 호텔 추천해줘"
    with c4:
        if st.button("🎒 짐 싸기\n\n여름 휴가 준비물"): button_prompt = "여름 휴가 짐 싸는 법 알려줘"

for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])

chat_prompt = st.chat_input("프롬프트를 입력하세요")

if button_prompt or chat_prompt:
    
    # API 키 입력 확인
    if not st.session_state.api_key:
        st.error("⚠️ API Key가 필요합니다. 왼쪽 사이드바에 키를 입력해주세요.")
        st.stop()

    prompt = button_prompt if button_prompt else chat_prompt

    if st.session_state.current_chat_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.chat_history[new_id] = {
            "title": "새로운 대화",
            "messages": []
        }
        st.session_state.current_chat_id = new_id
        messages = st.session_state.chat_history[new_id]['messages']

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            app = st.session_state.langgraph_app
            config = st.session_state.langgraph_config

            print(prompt)
            user_input = prompt
            if user_input.lower() in ["q", "quit"]:
                print("종료합니다.")
                full_response += "종료합니다."
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            result = app.invoke(inputs, config=config)
            
            ai_msg = result["final_answer"]
            route = result.get("route", "알 수 없음")
            summary = result.get("summary", "")

            print(f"AI: {ai_msg}")
            full_response += f"AI: {ai_msg}"
            print(f" └─ [Debug] 경로: {route}")
            full_response += f" └─ [Debug] 경로: {route}"
            
            if summary:
                print(f"   └─ [Debug] 📝 요약 발생: {summary}")
                full_response += f"   └─ [Debug] 📝 요약 발생: {summary}"
                
            message_placeholder.markdown(full_response)
                
            
        except Exception as e:
            full_response = f"⚠️ 에러가 발생했습니다: {str(e)}"
            message_placeholder.error(full_response)
    
    messages.append({"role": "assistant", "content": full_response})

    if len(messages) == 2:
        new_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
        st.session_state.chat_history[st.session_state.current_chat_id]["title"] = new_title
        st.rerun()