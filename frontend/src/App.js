// src/App.js
import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

function App() {
  // --- 상태 관리 (Streamlit의 session_state 대응) ---
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState("");
  
  // 로그인 입력 상태
  const [inputId, setInputId] = useState("");
  const [inputPw, setInputPw] = useState("");

  // 채팅 데이터
  const [chatHistory, setChatHistory] = useState({}); // { id: { title: "", messages: [] } }
  const [currentChatId, setCurrentChatId] = useState(null);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false); // 로딩/타이핑 상태

  const messagesEndRef = useRef(null);

  // --- 스크롤 자동 이동 ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, currentChatId, isTyping]);

  // --- 로그인 핸들러 ---
  const handleLogin = () => {
    if (inputId === "admin" && inputPw === "boostcamp") {
      setLoggedIn(true);
      setUsername(inputId);
      setCurrentChatId(null);
    } else {
      alert("잘못된 정보입니다.");
    }
  };

  const handleLogout = () => {
    setLoggedIn(false);
    setUsername("");
    setChatHistory({});
    setCurrentChatId(null);
  };

  // --- 채팅 관련 함수 ---
  const createNewChat = () => {
    setCurrentChatId(null);
  };

  const deleteChat = (e, id) => {
    e.stopPropagation(); // 부모 클릭 방지
    const newHistory = { ...chatHistory };
    delete newHistory[id];
    setChatHistory(newHistory);
    if (currentChatId === id) setCurrentChatId(null);
  };

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    // 1. 채팅방 ID 설정 (없으면 생성)
    let threadId = currentChatId;
    if (!threadId) {
      threadId = uuidv4();
      setChatHistory(prev => ({
        ...prev,
        [threadId]: { title: "새로운 대화", messages: [] }
      }));
      setCurrentChatId(threadId);
    }

    // 2. 사용자 메시지 추가
    const userMsg = { role: "user", content: text };
    
    // 상태 업데이트 (불변성 유지)
    setChatHistory(prev => {
      const currentMsgs = prev[threadId]?.messages || [];
      return {
        ...prev,
        [threadId]: {
          ...prev[threadId],
          messages: [...currentMsgs, userMsg]
        }
      };
    });
    setInputValue("");
    setIsTyping(true);

    try {
      // 3. 백엔드 API 호출
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, thread_id: threadId }),
      });

      if (!response.ok) throw new Error("API Error");
      
      const data = await response.json();
      const botResponse = data.response;

      // 4. 타이핑 효과 시뮬레이션 (Streamlit의 time.sleep 모방)
      let displayedText = "";
      const words = botResponse.split(" ");
      
      // 봇 메시지 빈 껍데기 추가
      setChatHistory(prev => {
        const msgs = prev[threadId].messages;
        return {
          ...prev,
          [threadId]: { ...prev[threadId], messages: [...msgs, { role: "assistant", content: "" }] }
        };
      });

      // 단어 단위로 렌더링
      for (let i = 0; i < words.length; i++) {
        await new Promise(r => setTimeout(r, 80)); // 0.08초 대기
        displayedText += words[i] + " ";
        
        setChatHistory(prev => {
           const msgs = [...prev[threadId].messages];
           msgs[msgs.length - 1].content = displayedText + "▌"; // 커서 효과
           return {
             ...prev,
             [threadId]: { ...prev[threadId], messages: msgs }
           };
        });
      }

      // 5. 완료 후 커서 제거 및 제목 업데이트
      setChatHistory(prev => {
        const msgs = [...prev[threadId].messages];
        msgs[msgs.length - 1].content = displayedText.trim();
        
        // 두 번째 메시지(첫 턴 완료)일 때 제목 업데이트
        let newTitle = prev[threadId].title;
        if (msgs.length === 2) {
           newTitle = text.length > 15 ? text.substring(0, 15) + "..." : text;
        }

        return {
          ...prev,
          [threadId]: { title: newTitle, messages: msgs }
        };
      });

    } catch (error) {
      alert("에러 발생: " + error.message);
    } finally {
      setIsTyping(false);
    }
  };

  // --- 렌더링 ---

  // 1. 로그인 전 화면
  if (!loggedIn) {
    return (
      <div className="login-wrapper">
        <div className="login-box">
          <h1 className="login-title">✈️ Traveler Login</h1>
          <hr style={{margin: "20px 0", borderTop: "1px solid #eee"}}/>
          <input 
            className="input-field" 
            placeholder="아이디" 
            value={inputId} onChange={(e) => setInputId(e.target.value)}
          />
          <input 
            className="input-field" 
            type="password" 
            placeholder="비밀번호" 
            value={inputPw} onChange={(e) => setInputPw(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
          />
          <div style={{height: '20px'}}></div>
          <button className="btn-primary" onClick={handleLogin}>로그인</button>
        </div>
      </div>
    );
  }

  // 2. 로그인 후 채팅 화면
  const currentMessages = currentChatId ? chatHistory[currentChatId]?.messages || [] : [];

  return (
    <div className="app-container">
      {/* 사이드바 */}
      <div className="sidebar">
        <h2>여행 챗봇</h2>
        <button className="btn-primary" onClick={createNewChat}>➕ 새 채팅</button>
        
        <div className="history-list">
          <h3>최근 대화</h3>
          {Object.keys(chatHistory).length === 0 && <p style={{color: '#888', fontSize: '0.8rem'}}>대화 기록이 없습니다.</p>}
          {Object.keys(chatHistory).reverse().map(id => (
            <div 
              key={id} 
              className={`history-item ${currentChatId === id ? 'active' : ''}`}
              onClick={() => setCurrentChatId(id)}
            >
              <span style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px'}}>
                {currentChatId === id ? "🟢 " : ""}{chatHistory[id].title}
              </span>
              <button className="delete-btn" onClick={(e) => deleteChat(e, id)}>🗑️</button>
            </div>
          ))}
        </div>

        <hr style={{width: '100%', borderColor: '#ddd'}} />
        <p style={{fontSize: '0.9rem', color: '#666'}}>Logged in as: {username}</p>
        <button onClick={handleLogout} style={{padding: '5px', width: '100%', cursor: 'pointer'}}>로그아웃</button>
      </div>

      {/* 메인 영역 */}
      <div className="chat-area">
        {/* 대화 내용이 없을 때 (Welcome Screen) */}
        {(!currentChatId || currentMessages.length === 0) ? (
          <div className="welcome-screen">
            <h2 className="gradient-text">안녕하세요, {username}님</h2>
            <h3>어디로 여행을 떠나고 싶으신가요?</h3>
            <div className="suggestion-grid">
              <button className="btn-primary" onClick={() => sendMessage("파리 여행 일정 짜줘")}>파리 여행</button>
              <button className="btn-primary" onClick={() => sendMessage("서울에서 뉴욕 가는 항공편 찾아줘")}>뉴욕행 항공권</button>
              <button className="btn-primary" onClick={() => sendMessage("도쿄에서 가성비 좋은 호텔 추천해줘")}>도쿄 숙소</button>
              <button className="btn-primary" onClick={() => sendMessage("여름 휴가 짐 싸는 법 알려줘")}>여행 짐</button>
            </div>
          </div>
        ) : (
          /* 대화 리스트 */
          <div className="messages-list">
            {currentMessages.map((msg, idx) => (
              <div key={idx} className="message">
                <div className={`avatar ${msg.role === 'user' ? 'user' : 'bot'}`}>
                  {msg.role === 'user' ? '🧑‍💻' : '🌐'}
                </div>
                <div className="msg-content">
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* 입력창 (항상 하단 표시) */}
        {(currentChatId || currentMessages.length > 0) && (
          <div className="input-container">
            <input 
              className="chat-input"
              placeholder="프롬프트를 입력하세요"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputValue)}
              disabled={isTyping}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;