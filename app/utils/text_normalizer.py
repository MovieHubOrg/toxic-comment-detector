import json
import re
from app.core.config import DATASET_PATH

# Load dictionary mapping unaccented/teencode words to their proper accented forms
try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        TOXIC_DICTIONARY = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load toxic dictionary from {DATASET_PATH}: {e}")
    TOXIC_DICTIONARY = {}

# Compile regex patterns for better performance
# Using \b to ensure we only match whole words
COMPILED_TOXIC_DICT = {
    re.compile(rf"\b{pattern}\b", re.IGNORECASE): replacement
    for pattern, replacement in TOXIC_DICTIONARY.items()
}

def normalize_text(text: str) -> str:
    """Replace unaccented/teencode phrases with accented versions."""
    for pattern, replacement in COMPILED_TOXIC_DICT.items():
        text = pattern.sub(replacement, text)
    return text
