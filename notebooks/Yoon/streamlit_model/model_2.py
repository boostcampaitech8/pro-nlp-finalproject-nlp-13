import streamlit as st
import time
import uuid

# --- 1. 페이지 설정 ---
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
    /* 버튼 스타일 조정 */
    .stButton > button {
        width: 100%;
        height: auto;
        padding-top: 15px;
        padding-bottom: 15px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. 데이터 관리 함수 및 세션 초기화 ---

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# current_chat_id가 None이면 "새 채팅" 화면(아직 생성 안 된 상태)을 의미
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        # 현재 보고 있는 방을 삭제했다면 새 채팅 화면으로 이동
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)


# --- 3. 로그인 화면 ---
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
                # 로그인 시 항상 새 채팅 화면(None)으로 시작
                st.session_state.current_chat_id = None
                st.rerun()
            else:
                st.error("잘못된 사용자명 또는 비밀번호입니다.")
    st.stop()


# --- 4. 메인 앱 (로그인 성공 후) ---

# [사이드바]
with st.sidebar:
    st.title("여행 챗봇")
    
    # 새 채팅 버튼: 누르면 ID를 None으로 설정하여 초기 화면으로 돌아감 (실제 생성 X)
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
        st.rerun()


# [메인 화면 로직]

# 1. 현재 메시지 리스트 준비
# current_chat_id가 None이면 아직 방이 없는 상태이므로 빈 리스트 사용
if st.session_state.current_chat_id is None:
    messages = []
else:
    # 존재하지 않는 ID가 들어있을 경우 대비 (삭제 후 새로고침 등)
    if st.session_state.current_chat_id not in st.session_state.chat_history:
        st.session_state.current_chat_id = None
        messages = []
    else:
        messages = st.session_state.chat_history[st.session_state.current_chat_id]['messages']

# 버튼 클릭 여부를 저장할 변수
button_prompt = None

# A. 메시지가 없을 때 (환영 화면 + 클릭 가능한 버튼)
if not messages:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"## <span style='background: linear-gradient(to right, #4285F4, #9B72CB); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>안녕하세요, {st.session_state.username}님</span>", unsafe_allow_html=True)
    st.markdown("### 어디로 여행을 떠나고 싶으신가요?")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4개의 제안 카드 (버튼)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("🌍 여행 계획\n\n파리 여행 일정 짜줘"):
            button_prompt = "파리 여행 일정 짜줘"
    with c2:
        if st.button("✈️ 항공권 검색\n\n서울-뉴욕 항공편"):
            button_prompt = "서울에서 뉴욕 가는 항공편 찾아줘"
    with c3:
        if st.button("🏨 호텔 추천\n\n도쿄 가성비 호텔"):
            button_prompt = "도쿄에서 가성비 좋은 호텔 추천해줘"
    with c4:
        if st.button("🎒 짐 싸기\n\n여름 휴가 준비물"):
            button_prompt = "여름 휴가 짐 싸는 법 알려줘"

# B. 기존 대화 기록 표시
for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])

# C. 입력 처리 (채팅창 입력 OR 버튼 클릭)
chat_prompt = st.chat_input("프롬프트를 입력하세요")

# 버튼이나 채팅창 둘 중 하나라도 입력이 있으면 실행
if button_prompt or chat_prompt:
    
    prompt = button_prompt if button_prompt else chat_prompt

    # [핵심 변경] 첫 메시지인 경우(현재 방이 None인 경우) -> 여기서 채팅방 생성!
    if st.session_state.current_chat_id is None:
        new_id = str(uuid.uuid4())
        st.session_state.chat_history[new_id] = {
            "title": "새로운 대화",
            "messages": []
        }
        st.session_state.current_chat_id = new_id
        # 새로 만든 방의 메시지 리스트로 교체
        messages = st.session_state.chat_history[new_id]['messages']

    # 1. 사용자 메시지 추가 및 표시
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # 2. AI 응답 생성 및 표시
    with st.chat_message("assistant", avatar="✨"):
        response_text = f"Gemini 스타일 봇입니다. 입력하신 내용: **{prompt}**"
        st.write_stream(stream_data(response_text))
    
    # 3. AI 응답 저장
    messages.append({"role": "assistant", "content": response_text})

    # 4. 제목 업데이트 (첫 턴이 끝난 직후)
    # len(messages)가 2라는 것은 방금 User(1) + AI(2) 메시지가 처음 쌓였다는 뜻
    if len(messages) == 2:
        new_title = prompt[:15] + "..." if len(prompt) > 15 else prompt
        st.session_state.chat_history[st.session_state.current_chat_id]["title"] = new_title
        
        # 사이드바에 방금 만든 채팅방을 띄우기 위해 리런
        st.rerun()