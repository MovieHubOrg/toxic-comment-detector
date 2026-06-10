import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import MODEL_NAME, DEVICE, ACCENT_MODEL_NAME

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.accent_tokenizer = None
        self.accent_model = None

    def load_model(self):
        print(f"Loading model '{MODEL_NAME}' on device: {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

        print(f"Loading accent restoration model '{ACCENT_MODEL_NAME}' on device: {DEVICE}...")
        self.accent_tokenizer = AutoTokenizer.from_pretrained(ACCENT_MODEL_NAME)
        self.accent_model = AutoModelForSeq2SeqLM.from_pretrained(ACCENT_MODEL_NAME).to(DEVICE)
        self.accent_model.eval()
        print("Accent restoration model and tokenizer loaded successfully.")

    def unload_model(self):
        print("Unloading models and cleaning cache...")
        self.model = None
        self.tokenizer = None
        self.accent_model = None
        self.accent_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

model_manager = ModelManager()
