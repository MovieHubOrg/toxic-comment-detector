import torch
from app.core.config import DEVICE, MAX_GENERATION_LENGTH
from app.core.model_manager import model_manager
from app.utils.text_normalizer import normalize_text
import difflib

def restore_original_casing(original_text: str, generated_text: str) -> str:
    if "[hate]" not in generated_text:
        return original_text
        
    gen_no_tags = generated_text.replace("[hate]", "")
    matcher = difflib.SequenceMatcher(None, gen_no_tags.lower(), original_text.lower())
    opcodes = matcher.get_opcodes()
    
    tag_positions = []
    curr_pos = 0
    for i, chunk in enumerate(generated_text.split("[hate]")):
        if i > 0:
            tag_positions.append(curr_pos)
        curr_pos += len(chunk)
        
    orig_tag_positions = []
    for p in tag_positions:
        mapped_p = len(original_text)
        for idx_op, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            if i1 <= p < i2:
                if tag == 'equal':
                    mapped_p = j1 + (p - i1)
                else:
                    mapped_p = j1
                break
            elif p == i2 and idx_op == len(opcodes) - 1:
                mapped_p = j2
        orig_tag_positions.append(mapped_p)
        
    orig_tag_positions.sort(reverse=True)
    
    result = original_text
    for p in orig_tag_positions:
        result = result[:p] + "[hate]" + result[p:]
        
    while "[hate][hate]" in result:
        result = result.replace("[hate][hate]", "")
        
    return result

def extract_toxic_spans(tagged_text: str) -> list[dict]:
    spans = []
    clean_idx = 0
    i = 0
    start_idx = -1
    in_span = False
    
    while i < len(tagged_text):
        if tagged_text[i:].startswith("[hate]"):
            if not in_span:
                start_idx = clean_idx
                in_span = True
            else:
                spans.append({"start": start_idx, "end": clean_idx})
                in_span = False
                start_idx = -1
            i += 6
        else:
            clean_idx += 1
            i += 1
            
    if in_span:
        spans.append({"start": start_idx, "end": clean_idx})
        
    return spans

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
    
    # Restore original casing for spans task
    if prefix == "hate-spans-detection":
        output_text = restore_original_casing(input_text, output_text)
        
    return output_text
