# server.py
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv

# 현재 폴더를 경로에 추가 (혹시 모를 import 에러 방지)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.AGENT.graph import LangGraph
except ImportError as e:
    print(f"Import Error: {e}")
    print("src 폴더가 server.py와 같은 위치에 있는지 확인해주세요.")
    class LangGraph:
        def run(self, message, thread_id):
            return f"테스트 응답: {message}"

load_dotenv(find_dotenv())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot_instance = LangGraph()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response_text = bot_instance.run(
            message=req.message,
            thread_id=req.thread_id
        )
        return {"response": response_text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)