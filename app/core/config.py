import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "tarudesu/ViHateT5-base-HSD"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU Utility Configurations
CUDA_DEVICE_INDEX = 0
BYTES_TO_GB_DIVISOR = 1024 ** 3

# Model Inference Configurations
MAX_GENERATION_LENGTH = 256

# Server Configurations
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# RabbitMQ Configurations
RABBITMQ_ENABLED = os.getenv("RABBITMQ_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
RABBITMQ_RECONNECT_SECONDS = float(os.getenv("RABBITMQ_RECONNECT_SECONDS", "5"))
TOXIC_COMMENT_DETECTOR_QUEUE = os.getenv(
    "TOXIC_COMMENT_DETECTOR_QUEUE",
    "TOXIC_COMMENT_DETECTOR_QUEUE",
)
UPDATE_VIDEO_QUEUE = os.getenv("UPDATE_VIDEO_QUEUE", "UPDATE_VIDEO_QUEUE")
DETECTOR_REQUEST_CMD = "CMD_DETECTOR_COMMENT"
DETECTOR_RESPONSE_CMD = "CMD_DONE_DETECTOR_COMMENT"
RABBITMQ_RESPONSE_APP = os.getenv(
    "RABBITMQ_RESPONSE_APP",
    "TOXIC_COMMENT_DETECTOR",
)

# Dataset Paths
DATASET_PATH = Path(__file__).parent.parent / "data" / "toxic_dataset.json"

