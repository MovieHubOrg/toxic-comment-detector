import os
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
