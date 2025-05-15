import re

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.strip()
