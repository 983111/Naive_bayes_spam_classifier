import re

def clean_text(text: str) -> list[str]:
    """Lowercase text, strip special characters/punctuation, and tokenize."""
    if not isinstance(text, str):
        return []
    # Keep alphanumeric characters and spaces
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.strip().split()
    return tokens