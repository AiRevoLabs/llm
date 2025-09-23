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
        print("🔄 Loading model...")
        # Try to load the model from safetensors
        try:
            llm = Llama(
                model_path="/app/model",
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback: try to find a specific model file
            model_files = [f for f in os.listdir("/app/model") if f.endswith('.safetensors')]
            if model_files:
                print(f"📁 Found model files: {model_files}")
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

        # Format prompt with career counselor context
        system_prompt = "You are a career counselor and recruitment expert. Provide helpful career advice, resume tips, and job search guidance."
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{request.prompt}<|im_end|>\n<|im_start|>assistant\n"

        response = model(
            formatted_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["<|im_end|>", "<|im_start|>"] + request.stop,
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

        # Build conversation prompt
        conversation = "<|im_start|>system\nYou are a career counselor and recruitment expert. Provide helpful career advice, resume tips, and job search guidance.<|im_end|>\n"

        for message in request.messages:
            conversation += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"

        conversation += "<|im_start|>assistant\n"

        response = model(
            conversation,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["<|im_end|>", "<|im_start|>"],
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