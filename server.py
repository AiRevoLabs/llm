import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import json

app = FastAPI()

# Initialize model (will be loaded on first request)
llm = None

class GenerateRequest(BaseModel):
    model: str = "qwen-career"
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen-career"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

def load_model():
    global llm
    if llm is None:
        print("🔄 Loading GGUF model...")
        try:
            llm = Llama(
                model_path="/app/model/qwen-career.gguf",
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            print("✅ GGUF model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return llm

@app.get("/")
async def root():
    return {"message": "llama.cpp server is running", "status": "ready"}

@app.get("/api/version")
async def version():
    return {"version": "llama.cpp-python", "status": "ready"}

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        model = load_model()

        # Format prompt with career counselor context - simplified format
        system_prompt = "You are a career counselor and recruitment expert. Provide helpful career advice, resume tips, and job search guidance."
        formatted_prompt = f"System: {system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"

        response = model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["User:", "System:", "#"] + request.stop,
            repeat_penalty=1.1,
            echo=False
        )

        return {
            "model": request.model,
            "response": response["choices"][0]["text"].strip(),
            "done": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        model = load_model()

        # Build conversation prompt - simplified format
        conversation = "System: You are a career counselor and recruitment expert. Provide helpful career advice, resume tips, and job search guidance.\n\n"

        for message in request.messages:
            role = message.role.capitalize()
            conversation += f"{role}: {message.content}\n\n"

        conversation += "Assistant:"

        response = model(
            conversation,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["User:", "System:", "#"],
            repeat_penalty=1.1,
            echo=False
        )

        return {
            "model": request.model,
            "message": {
                "role": "assistant",
                "content": response["choices"][0]["text"].strip()
            },
            "done": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting llama.cpp server...")
    uvicorn.run(app, host="0.0.0.0", port=11434)