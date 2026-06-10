from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time
import uvicorn

from app.core.config import DEVICE, API_HOST, API_PORT
from app.core.model_manager import model_manager
from app.schemas.comment import CommentCheckRequest, CommentCheckResponse
from app.services.detector import run_inference
from app.utils.gpu import get_gpu_diagnostics

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    model_manager.load_model()
    yield
    # Clean up on shutdown
    model_manager.unload_model()

app = FastAPI(
    title="Vietnamese Toxic Comment Detector API",
    description="API for detecting hate speech, toxic speech, and hate spans in Vietnamese text using ViHateT5.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/comment/check", response_model=CommentCheckResponse)
async def check_comment(payload: CommentCheckRequest):
    if model_manager.model is None or model_manager.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    
    start_time = time.time()
    task = payload.task
    
    try:
        if task == "all":
            tasks = ["hate-speech-detection", "toxic-speech-detection", "hate-spans-detection"]
            results = {}
            for t in tasks:
                results[t] = run_inference(payload.text, t)
            result = None
        else:
            results = None
            result = run_inference(payload.text, task)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
        
    end_time = time.time()
    
    return CommentCheckResponse(
        text=payload.text,
        task=task,
        result=result,
        results=results,
        time_taken_seconds=round(end_time - start_time, 4)
    )

@app.get("/health")
async def health_check():
    gpu_info = get_gpu_diagnostics()
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model_manager.model is not None,
        "gpu_diagnostics": gpu_info
    }

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
