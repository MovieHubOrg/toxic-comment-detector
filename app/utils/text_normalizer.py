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
# Sort patterns by length in descending order to match longer phrases first
COMPILED_TOXIC_DICT = {
    re.compile(rf"\b{pattern}\b", re.IGNORECASE): replacement
    for pattern, replacement in sorted(TOXIC_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)
}

NORMALIZED_TOXIC_DICTIONARY = {
    pattern.casefold(): replacement
    for pattern, replacement in TOXIC_DICTIONARY.items()
}
SORTED_TOXIC_PATTERNS = sorted(
    TOXIC_DICTIONARY,
    key=len,
    reverse=True,
)
COMPILED_TOXIC_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(pattern) for pattern in SORTED_TOXIC_PATTERNS)
    + r")(?!\w)",
    re.IGNORECASE,
) if SORTED_TOXIC_PATTERNS else None

def normalize_text(text: str) -> str:
    """Replace unaccented/teencode phrases with accented versions."""
    if COMPILED_TOXIC_PATTERN is None:
        return text

    return COMPILED_TOXIC_PATTERN.sub(
        lambda match: NORMALIZED_TOXIC_DICTIONARY[match.group(0).casefold()],
        text,
    )
