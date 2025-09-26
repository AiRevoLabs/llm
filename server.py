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
            # Optimize for high-memory system (32GB available)
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()

            llm = Llama(
                model_path="/app/model/qwen-career.gguf",
                n_ctx=8192,  # Increased context window for longer conversations
                n_threads=min(cpu_count, 16),  # Use more CPU threads for parallel processing
                n_batch=1024,  # Large batch size for faster prompt processing
                use_mmap=True,  # Memory-map the model file
                use_mlock=True,  # Lock model in memory to prevent swapping
                verbose=False,
                # High-memory optimizations
                rope_freq_base=10000.0,  # RoPE frequency base for better context handling
                rope_freq_scale=1.0,  # RoPE frequency scaling
            )
            print("✅ GGUF model loaded successfully")
            print(f"📊 Using {cpu_count} CPU cores, max threads: {min(cpu_count, 16)}")
            print(f"🧠 Context window: 8192 tokens, Batch size: 1024")
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
            echo=False,
            # High-performance generation settings
            threads=min(cpu_count, 16),  # Use more threads for generation
            batch_size=1024  # Large batch for faster processing
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
            echo=False,
            # High-performance generation settings
            threads=min(cpu_count, 16),  # Use more threads for generation
            batch_size=1024  # Large batch for faster processing
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