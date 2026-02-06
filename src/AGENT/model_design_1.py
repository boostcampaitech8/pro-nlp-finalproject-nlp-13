import streamlit as st
import time
import uuid
from dotenv import load_dotenv, find_dotenv

from src.AGENT.graph import LangGraph

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="여행 챗봇",
    page_icon="✈️",
    layout="wide"
)



st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border: none;
        border-radius: 12px;
        background: #7A52F4 !important;
        color: white !important; 
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #6236D8 !important;
        transform: translateY(-2px); 
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    
    .stChatMessageAvatarUser {
        background-color: #7A52F4 !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #E0E0E0; 
        transition: all 0.2s; 
    }

    .stTextInput > div > div > input:focus {
        border-color: #7A52F4 !important; 
        box-shadow: 0 0 0 1px #7A52F4 !important; 
    }
    
    .stSpinner > div {
        border-top-color: #7A52F4 !important;
    }

</style>
""", unsafe_allow_html=True)




if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "bot_instance" not in st.session_state:
    st.session_state.bot_instance = LangGraph()

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


if not st.session_state.logged_in:
    st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True): 
            st.markdown("<h1 style='text-align: center; color: #6a11cb;'>✈️ Traveler Login</h1>", unsafe_allow_html=True)
            st.markdown("---")
            input_username = st.text_input("아이디", placeholder="ID")
            input_password = st.text_input("비밀번호", type="password", placeholder="password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("로그인", type="primary", use_container_width=True):
                if input_username == "admin" and input_password == "boostcamp":
                    st.session_state.logged_in = True
                    st.session_state.username = input_username
                    st.session_state.current_chat_id = None
                    st.rerun()
                else:
                    st.error("잘못된 정보입니다.")
    st.stop()



with st.sidebar:
    st.title("여행 챗봇")

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

            col_btn, col_del = st.columns([0.75, 0.25], vertical_alignment="center")
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
        st.session_state.api_key = ""
        st.rerun()



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

    if st.session_state.bot_instance is None:
        st.error("❌ API가 연결되지 않았습니다.")
    else:
        st.info("어떤 것이든 질문해주세요!")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("파리 여행"): button_prompt = "파리 여행 일정 짜줘"
    if c2.button("뉴욕행 항공권"): button_prompt = "서울에서 뉴욕 가는 항공편 찾아줘"
    if c3.button("도쿄 숙소"): button_prompt = "도쿄에서 가성비 좋은 호텔 추천해줘"
    if c4.button("여행 짐"): button_prompt = "여름 휴가 짐 싸는 법 알려줘"

for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🌐"):  
        st.markdown(msg["content"])

chat_prompt = st.chat_input("프롬프트를 입력하세요")
prompt = button_prompt if button_prompt else chat_prompt

if prompt:
    if st.session_state.bot_instance is None:
        st.error("⚠️ 서버 설정을 확인해주세요 (API Key Missing).")
        st.stop()

    if st.session_state.current_chat_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.chat_history[new_id] = {"title": "새로운 대화", "messages": []}
        st.session_state.current_chat_id = new_id
        messages = st.session_state.chat_history[new_id]['messages']

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🌐"):  
        message_placeholder = st.empty()

        try:
            with st.spinner("답변을 작성하고 있어요..."): 
                response_text = st.session_state.bot_instance.run(
                    message=prompt,
                    thread_id=st.session_state.current_chat_id
                )

            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.08) 
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ 에러가 발생했습니다: {str(e)}"
            message_placeholder.error(full_response)

    messages.append({"role": "assistant", "content": full_response})

    if len(messages) == 2:
        new_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
        st.session_state.chat_history[st.session_state.current_chat_id]["title"] = new_title
        st.rerun()