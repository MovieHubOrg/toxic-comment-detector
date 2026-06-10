import torch
from app.core.config import DEVICE, MAX_GENERATION_LENGTH
from app.core.model_manager import model_manager
from app.utils.text_normalizer import normalize_text

def run_inference(input_text: str, prefix: str) -> str:
    # Ensure model and tokenizer are loaded
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("Model is not loaded.")
        
    # Normalize text to handle unaccented/teencode words
    normalized_text = normalize_text(input_text)
        
    # Add prefix
    prefixed_input_text = prefix + ': ' + normalized_text

    # Tokenize input text
    inputs = model_manager.tokenizer(prefixed_input_text, return_tensors="pt").to(DEVICE)

    # Generate output
    with torch.no_grad():
        output_ids = model_manager.model.generate(**inputs, max_length=MAX_GENERATION_LENGTH)

    # Decode the generated output
    output_text = model_manager.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text
