import torch
from math import ceil
from app.core.config import CUDA_DEVICE_INDEX, BYTES_TO_GB_DIVISOR

def get_gpu_diagnostics() -> dict:
    if torch.cuda.is_available():
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(CUDA_DEVICE_INDEX),
            "device_count": torch.cuda.device_count(),
            "memory_capacity_gb": ceil(torch.cuda.get_device_properties(CUDA_DEVICE_INDEX).total_memory / BYTES_TO_GB_DIVISOR),
            "cuda_version": torch.version.cuda
        }
    else:
        return {
            "cuda_available": False
        }
