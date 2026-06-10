# Vietnamese Toxic Comment Detector API

This repository contains a modular FastAPI-based web service to serve the **ViHateT5** hate and toxic speech detection model for the Vietnamese language. 

> [!WARNING]
> **Disclaimer:** This project and the underlying model contain examples and predictions from actual content on social media platforms that could be considered toxic, hateful, and offensive. 

## Model Attribution

- **Model Source:** [tarudesu/ViHateT5-base-HSD on Hugging Face](https://huggingface.co/tarudesu/ViHateT5-base-HSD)
- **Authors:** The model is introduced in the research paper: *"ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model"*, accepted at **ACL 2024 (Findings)**.
- **Pre-trained Foundation:** Fine-tuned on the VOZ-HSD dataset (comprising over 10.7 million comments) utilizing a unified text-to-text T5 transformer model architecture.

---

## Features

- **Efficient Startup:** Loads model weights and the tokenizer onto the target computing device (GPU/CUDA if available, or CPU) once during application startup.
- **Multi-task Detection Endpoints**:
  - `toxic-speech-detection`: Checks if a comment is clean or toxic.
  - `hate-speech-detection`: Checks if a comment contains hate speech, offensive speech, or is clean.
  - `hate-spans-detection`: Extracts and marks offensive spans within the input text using inline `[hate]...[hate]` tags.
  - `all`: Sequentially executes all three tasks and returns the combined results mapping.
- **Diagnostics**: Health check endpoint displays environment information, including GPU memory availability and CUDA version properties.
- **Configurable**: Environment-driven parameters loaded from a `.env` file via `python-dotenv`.

---

## Project Structure

```
toxic-comment-detector/
├── main.py (API Entrypoint)
├── requirements.txt
├── .env (Server host and port configurations)
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py (Config parser & global settings)
│   │   └── model_manager.py (Model lifecycle controller)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── comment.py (Pydantic payload schemas)
│   ├── services/
│   │   ├── __init__.py
│   │   └── detector.py (Inference processing logic)
│   └── utils/
│       ├── __init__.py
│       └── gpu.py (GPU diagnostic utilities)
```

---

## Quick Start

### 1. Installation

Install the project dependencies in your virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root folder of the project (already ignored in `.gitignore`):
```env
API_HOST=127.0.0.1
API_PORT=8000
```

### 3. Launch Server

Run the server using python:
```bash
python main.py
```

### 4. Interactive API Documentation

Once started, access the interactive Swagger UI to test requests:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc Page: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Documentation

### Check Comment (`POST /api/v1/comment/check`)

#### Request Payload
```json
{
  "text": "Đm nó, phim lởm quá trời, diễn viên diễn như cc, nội dung thì xàm, đạo diễn như ***, làm phim dở ẹc, tốn tiền vé vcl",
  "task": "all"
}
```

#### Response Payload
```json
{
  "text": "Đm nó, phim lởm quá trời, diễn viên diễn như cc, nội dung thì xàm, đạo diễn như ***, làm phim dở ẹc, tốn tiền vé vcl",
  "task": "all",
  "result": null,
  "results": {
    "hate-speech-detection": "hate",
    "toxic-speech-detection": "toxic",
    "hate-spans-detection": "[hate]đm[hate] [hate]nó[hate], phim lởm quá trời, diễn viên diễn như [hate]cc[hate], nội dung thì [hate]xàm[hate], [hate]đạo diễn như ***[hate], làm phim dở ẹc, tốn tiền vé [hate]vcl[hate]"
  },
  "time_taken_seconds": 1.733
}
```

### Health Check (`GET /health`)

#### Response Payload
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "gpu_diagnostics": {
    "cuda_available": true,
    "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "device_count": 1,
    "memory_capacity_gb": 4,
    "cuda_version": "12.1"
  }
}
```
