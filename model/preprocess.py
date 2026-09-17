import re

STOPWORDS = {
    "a","an","the","is","are","was","were","of","to","in",
    "on","at","for","with","this","that","it","and","or",
    "be","as","by","from","you","your","i","me","my","we",
    "our","they","them","he","she","his","her"
}

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # Remove phone numbers
    text = re.sub(r"\\d+", "", text)

    # Keep letters only
    text = re.sub(r"[^a-z\\s]", " ", text)

    words = text.split()

    words = [w for w in words if w not in STOPWORDS]

    return words