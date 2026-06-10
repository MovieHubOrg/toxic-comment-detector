import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import MODEL_NAME, DEVICE

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading model '{MODEL_NAME}' on device: {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def unload_model(self):
        print("Unloading model and cleaning cache...")
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

model_manager = ModelManager()
