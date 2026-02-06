# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.get("/")
def read_root():
    return {"message": "서버가 정상적으로 켜져 있습니다! 이제 React(localhost:3000)를 실행하세요."}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    time.sleep(0.5) 
    return {
        "response": f"✅ [서버 연결 성공] 입력하신 내용: '{req.message}'\n\n현재 LangGraph 연결 없이 UI 테스트 중입니다."
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 실행 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)