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
- **Robust Text Normalization**: Automatically converts unaccented Vietnamese slang and teencode into proper accented forms using a dedicated dataset (`toxic_dataset.json`) containing hundreds of mappings before inference.
- **Multi-task Detection Endpoints**:
  - `toxic-speech-detection`: Checks if a comment is clean or toxic.
  - `hate-speech-detection`: Checks if a comment contains hate speech, offensive speech, or is clean.
  - `hate-spans-detection`: Extracts and marks offensive spans within the input text using inline `[hate]...[hate]` tags.
  - `all`: Sequentially executes all three tasks and returns the combined results mapping.
- **Diagnostics**: Health check endpoint displays environment information, including GPU memory availability and CUDA version properties.
- **Configurable**: Environment-driven parameters loaded from a `.env` file via `python-dotenv`.
- **RabbitMQ Integration**: Consumes comment detection jobs from RabbitMQ and publishes toxic detection results for backend processing.

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
│   ├── data/
│   │   └── toxic_dataset.json (Dictionary for teencode/unaccented normalization)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── comment.py (Pydantic HTTP payload schemas)
│   │   └── rabbitmq.py (Pydantic RabbitMQ payload schemas)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detector.py (Inference processing logic)
│   │   └── rabbitmq.py (RabbitMQ consumer and publisher)
│   └── utils/
│       ├── __init__.py
│       ├── text_normalizer.py (Text preprocessing logic)
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
LOG_LEVEL=INFO
RABBITMQ_ENABLED=true
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_RECONNECT_SECONDS=5
TOXIC_COMMENT_DETECTOR_QUEUE=TOXIC_COMMENT_DETECTOR_QUEUE
UPDATE_VIDEO_QUEUE=UPDATE_VIDEO_QUEUE
RABBITMQ_RESPONSE_APP=TOXIC_COMMENT_DETECTOR
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

### RabbitMQ Comment Detection

When `RABBITMQ_ENABLED=true`, the service consumes messages from `TOXIC_COMMENT_DETECTOR_QUEUE`.
RabbitMQ payloads use the generic envelope `BaseMessage[T]`, where `data` is typed per command.

Input message:
```json
{
  "cmd": "CMD_DETECTOR_COMMENT",
  "app": "BACKEND_APP",
  "data": {
    "content": "Phim hay 1",
    "comment_id": 9567938084306944
  }
}
```

The service first runs `toxic-speech-detection`. If the result is `toxic`, it runs `hate-spans-detection` and publishes this message to `UPDATE_VIDEO_QUEUE`:
```json
{
  "cmd": "CMD_DONE_DETECTOR_COMMENT",
  "app": "TOXIC_COMMENT_DETECTOR",
  "data": {
    "comment_id": 9567938084306944,
    "content": "[hate]...detected toxic spans...[hate]"
  }
}
```
