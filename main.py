from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException
import uvicorn

from app.core.config import API_HOST, API_PORT, DEVICE, LOG_LEVEL
from app.core.model_manager import model_manager
from app.schemas.comment import CommentCheckRequest, CommentCheckResponse
from app.services.detector import run_inference, extract_toxic_spans
from app.services.detector import run_inference
from app.services.rabbitmq import rabbitmq_comment_detector
from app.utils.gpu import get_gpu_diagnostics

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
for handler in logging.getLogger().handlers:
    handler.setLevel(LOG_LEVEL)
logging.getLogger("app").setLevel(LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    model_manager.load_model()
    await rabbitmq_comment_detector.start()
    try:
        yield
    finally:
        # Clean up on shutdown
        await rabbitmq_comment_detector.stop()
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
    toxic_spans = None
    
    try:
        if task == "all":
            tasks = ["hate-speech-detection", "toxic-speech-detection", "hate-spans-detection"]
            results = {}
            for t in tasks:
                results[t] = run_inference(payload.text, t)
            result = None
            toxic_spans = extract_toxic_spans(results["hate-spans-detection"])
        else:
            results = None
            result = run_inference(payload.text, task)
            if task == "hate-spans-detection":
                toxic_spans = extract_toxic_spans(result)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
        
    end_time = time.time()
    
    return CommentCheckResponse(
        text=payload.text,
        task=task,
        result=result,
        results=results,
        toxicSpans=toxic_spans,
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
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
