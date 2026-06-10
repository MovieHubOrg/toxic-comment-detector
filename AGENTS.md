# AGENTS.md — Vietnamese Toxic Comment Detector API

This file provides context and rules for AI coding agents working in this repository.

---

## Project Overview

A modular **FastAPI** service that exposes the **ViHateT5** model (Seq2Seq T5 architecture, fine-tuned on VOZ-HSD) for Vietnamese hate/toxic speech detection. The model is loaded once on startup and served via REST endpoints. It supports GPU acceleration via CUDA.

- **Model:** [`tarudesu/ViHateT5-base-HSD`](https://huggingface.co/tarudesu/ViHateT5-base-HSD) (Hugging Face)
- **Paper:** *"ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model"* — ACL 2024 (Findings)
- **Language:** Vietnamese

---

## Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Web framework | FastAPI ≥ 0.110.0                |
| ASGI server  | Uvicorn ≥ 0.28.0                 |
| ML framework | PyTorch 2.5.1 + CUDA 12.1        |
| NLP          | Hugging Face Transformers 4.46.3 |
| Validation   | Pydantic ≥ 2.6.0                 |
| Config       | python-dotenv                     |
| Runtime      | Python 3.x, virtual env in `.venv/` |

---

## Directory Structure

```
toxic-comment-detector/
├── main.py                   # FastAPI app, lifespan, route definitions
├── requirements.txt          # Pinned dependencies (CUDA-specific torch index)
├── .env                      # Local env vars (gitignored) — API_HOST, API_PORT
├── .env.example              # Template for .env
├── AGENTS.md                 # This file
├── README.md                 # Human-facing documentation
└── app/
    ├── __init__.py
    ├── core/
    │   ├── config.py         # Global constants & env var loading
    │   └── model_manager.py  # ModelManager class: load/unload lifecycle
    ├── schemas/
    │   └── comment.py        # Pydantic request/response schemas
    ├── services/
    │   └── detector.py       # run_inference() — tokenize, generate, decode
    └── utils/
        └── gpu.py            # get_gpu_diagnostics() — CUDA info dict
```

---

## Key Modules

### `app/core/config.py`
Loads `.env` via `python-dotenv`. Exposes:
- `MODEL_NAME` — HuggingFace model ID
- `DEVICE` — `torch.device("cuda")` if available, else `"cpu"`
- `MAX_GENERATION_LENGTH = 256`
- `API_HOST`, `API_PORT` — from env or defaults `127.0.0.1:8000`

### `app/core/model_manager.py`
Singleton `ModelManager` instance (`model_manager`). Responsible for:
- `load_model()` — downloads/loads tokenizer & model, moves to `DEVICE`, sets `model.eval()`
- `unload_model()` — sets to `None`, flushes `torch.cuda.empty_cache()`

### `app/services/detector.py`
`run_inference(input_text: str, prefix: str) -> str`
- Prepends the task prefix: `"<prefix>: <text>"`
- Tokenizes, runs `model.generate()` with `torch.no_grad()`
- Decodes output with `skip_special_tokens=True`

### `app/schemas/comment.py`
- `CommentCheckRequest`: `text: str`, `task: Literal["toxic-speech-detection", "hate-speech-detection", "hate-spans-detection", "all"]`
- `CommentCheckResponse`: `text`, `task`, `result` (single task), `results` (all-task dict), `time_taken_seconds`

---

## API Endpoints

| Method | Path                    | Description                        |
|--------|-------------------------|------------------------------------|
| POST   | `/api/v1/comment/check` | Run detection on a Vietnamese comment |
| GET    | `/health`               | Health check + GPU diagnostics     |
| GET    | `/docs`                 | Swagger UI (auto-generated)        |
| GET    | `/redoc`                | ReDoc UI (auto-generated)          |

### Detection Tasks & Prefixes (used as T5 task prefixes)

| Task string               | Output                                         |
|---------------------------|------------------------------------------------|
| `toxic-speech-detection`  | `"clean"` or `"toxic"`                         |
| `hate-speech-detection`   | `"clean"`, `"offensive"`, or `"hate"`          |
| `hate-spans-detection`    | Original text with `[hate]...[hate]` markers   |
| `all`                     | Dict of all three tasks above                  |

---

## Running Locally

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env (copy from example)
copy .env.example .env

# 4. Start the server
python main.py
```

The server will be available at `http://127.0.0.1:8000` (or as configured in `.env`).

> **Note:** First startup downloads the model from Hugging Face (~900 MB). Subsequent startups load from the local HuggingFace cache.

---

## Environment Variables (`.env`)

```env
API_HOST=127.0.0.1
API_PORT=8000
```

---

## Coding Conventions

- **Python style:** Follow PEP 8. Use type hints on all function signatures.
- **Imports:** Group as stdlib → third-party → local (`app.*`), separated by blank lines.
- **Schemas:** All request/response models go in `app/schemas/`. Use Pydantic `Field()` for descriptions and examples.
- **Config:** All constants and env vars belong in `app/core/config.py`. Never hardcode model names, ports, or device strings elsewhere.
- **Inference logic:** All model I/O lives in `app/services/detector.py`. Routes in `main.py` must not directly touch the tokenizer or model.
- **Error handling:** Use `HTTPException` at the route level. Raise plain `ValueError` from service/utility functions; routes catch and convert them.
- **Model state:** Always guard inference with `if model_manager.model is None` checks before calling `run_inference()`.
- **No tests directory yet** — if adding tests, place them in `tests/` at the project root and use `pytest`.

---

## Agent Rules

1. **Do not change `MODEL_NAME`** in `config.py` — it must remain `"tarudesu/ViHateT5-base-HSD"`.
2. **Do not modify the task prefix strings** (`hate-speech-detection`, `toxic-speech-detection`, `hate-spans-detection`) — these are the exact prefixes the model was trained on.
3. **Do not add synchronous blocking I/O** inside async route handlers. Use `run_in_executor` if needed.
4. **Do not commit `.env`** — it is in `.gitignore`. Only update `.env.example` when adding new variables.
5. **Preserve the singleton pattern** for `model_manager` — it must not be re-instantiated per request.
6. **GPU memory:** If adding batch inference, always wrap in `torch.no_grad()` and be mindful of VRAM limits (tested on 4 GB VRAM).
7. When adding new endpoints, follow the existing prefix pattern: `/api/v1/...`.
