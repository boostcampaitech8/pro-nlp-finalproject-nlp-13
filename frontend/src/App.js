import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

import botProfile from './bot_profile.png'; 

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState("");
  const [inputId, setInputId] = useState("");
  const [inputPw, setInputPw] = useState("");

  const [chatHistory, setChatHistory] = useState({});
  const [currentChatId, setCurrentChatId] = useState(null);
  const [inputValue, setInputValue] = useState("");
  
  const [isTyping, setIsTyping] = useState(false);
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 768);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, currentChatId, isTyping]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 768) {
        setIsSidebarOpen(false);
      } else {
        setIsSidebarOpen(true);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleLogin = () => {
    if (inputId === "admin" && inputPw === "boostcamp") {
      setLoggedIn(true);
      setUsername(inputId);
      if (window.innerWidth <= 768) setIsSidebarOpen(false);
    } else {
      alert("로그인 실패: admin / boostcamp");
    }
  };

  const handleLogout = () => {
    setLoggedIn(false);
    setUsername("");
    setChatHistory({});
    setCurrentChatId(null);
    setInputValue("");
    if (window.innerWidth <= 768) setIsSidebarOpen(false); 
  };

  const createNewChat = () => {
    setCurrentChatId(null);
    if (window.innerWidth <= 768) setIsSidebarOpen(false);
  };

  const selectChat = (id) => {
    setCurrentChatId(id);
    if (window.innerWidth <= 768) setIsSidebarOpen(false);
  };

  const deleteChat = (e, id) => {
    e.stopPropagation();
    const newHistory = { ...chatHistory };
    delete newHistory[id];
    setChatHistory(newHistory);
    if (currentChatId === id) setCurrentChatId(null);
  };

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    let threadId = currentChatId;
    if (!threadId) {
      threadId = uuidv4();
      setChatHistory(prev => ({
        ...prev,
        [threadId]: { title: "새로운 대화", messages: [] }
      }));
      setCurrentChatId(threadId);
    }

    const userMsg = { role: "user", content: inputValue };
    setChatHistory(prev => ({
      ...prev,
      [threadId]: {
        ...prev[threadId],
        messages: [...(prev[threadId]?.messages || []), userMsg]
      }
    }));
    
    setInputValue("");
    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.content, thread_id: threadId }),
      });

      if (!response.ok) throw new Error("Server Error");
      const data = await response.json();
      const fullResponse = data.response;

      setIsTyping(false); 

      setChatHistory(prev => ({
        ...prev,
        [threadId]: {
          ...prev[threadId],
          messages: [...prev[threadId].messages, { role: "bot", content: "" }]
        }
      }));

      let currentText = "";
      for (let i = 0; i < fullResponse.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 20));
        currentText += fullResponse[i];
        setChatHistory(prev => {
          const msgs = [...prev[threadId].messages];
          msgs[msgs.length - 1] = { role: "bot", content: currentText };
          return { ...prev, [threadId]: { ...prev[threadId], messages: msgs } };
        });
      }

      setChatHistory(prev => {
        const msgs = prev[threadId].messages;
        let newTitle = prev[threadId].title;
        if (msgs.length === 2) {
           newTitle = userMsg.content.length > 12 ? userMsg.content.substring(0, 12) + "..." : userMsg.content;
        }
        return { ...prev, [threadId]: { title: newTitle, messages: msgs } };
      });

    } catch (error) {
      setIsTyping(false);
      alert("서버 연결 실패 (backend가 켜져있나요?)");
    }
  };

  if (!loggedIn) {
    return (
      <div className="login-wrapper">
        <div className="login-box">
          <h2 style={{color: '#7A52F4', marginBottom: '20px'}}>✈️ 여행 친구</h2>
          <input className="login-input" placeholder="ID" value={inputId} onChange={e=>setInputId(e.target.value)}/>
          <input className="login-input" type="password" placeholder="PW" value={inputPw} onChange={e=>setInputPw(e.target.value)} onKeyPress={e=>e.key==='Enter' && handleLogin()}/>
          <button className="btn-primary" onClick={handleLogin}>여행 시작하기</button>
        </div>
      </div>
    );
  }

  const currentMessages = currentChatId ? chatHistory[currentChatId]?.messages || [] : [];

  return (
    <div className="app-container">
      {isSidebarOpen && window.innerWidth <= 768 && (
        <div 
          style={{position:'absolute', top:0, left:0, right:0, bottom:0, background:'rgba(0,0,0,0.5)', zIndex:900}}
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <div className={`sidebar ${isSidebarOpen ? '' : 'closed'}`}>
        <div className="sidebar-content">
          <div className="sidebar-header">
            <h2 style={{margin: 0, fontSize: '1.2rem'}}>🎒 여행 친구</h2>
            <button className="close-btn" onClick={() => setIsSidebarOpen(false)}>✕</button>
          </div>
          <button className="btn-primary" onClick={createNewChat} style={{marginBottom: '15px'}}>+ 새 여행 계획</button>
          
          <div className="history-list">
            {Object.keys(chatHistory).length === 0 && <p style={{color:'#999', fontSize:'0.9rem', textAlign:'center'}}>여행 기록이 없어요.</p>}
            {Object.keys(chatHistory).reverse().map(id => (
              <div key={id} className={`history-item ${currentChatId === id ? 'active' : ''}`} onClick={() => selectChat(id)}>
                <span style={{overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:'180px'}}>
                  {chatHistory[id].title}
                </span>
                <button className="delete-btn" onClick={(e) => deleteChat(e, id)}>×</button>
              </div>
            ))}
          </div>
          
          <div className="sidebar-footer">
            <div className="user-info"><span>👤 {username}님</span></div>
            <button className="btn-logout" onClick={handleLogout}>로그아웃</button>
          </div>
        </div>
      </div>

      <div className="chat-area">
        <div className="chat-header">
          <button className="menu-btn" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>☰</button>
          <span className="header-title">
            {currentChatId ? chatHistory[currentChatId]?.title : "여행 친구"}
          </span>
        </div>

        <div className="messages-list">
          {currentMessages.length === 0 ? (
            <div style={{display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', color:'#000000', textAlign:'center', textShadow: '0 0 10px rgba(255,255,255, 0.8)'}}>
              <div style={{fontSize: '3rem', marginBottom: '10px'}}>✈️</div>
              <h3>여행 친구와 함께 떠나볼까요?</h3>
              <p style={{fontSize: '0.9rem'}}>가고 싶은 곳을 말해보세요!</p>
            </div>
          ) : (
            currentMessages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>
                  {msg.role === 'bot' ? <img src={botProfile} alt="bot" /> : '🙂'}
                </div>
                <div className="msg-content">{msg.content}</div>
              </div>
            ))
          )}
          
          {isTyping && (
            <div className="message bot">
              <div className="avatar bot">
                <img src={botProfile} alt="bot" />
              </div>
              <div className="msg-content loading">
                <div className="loading-spinner"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <input 
              className="chat-input"
              placeholder="여행 친구에게 물어보기..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button className="send-btn" onClick={sendMessage} disabled={!inputValue.trim()}>➤</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;