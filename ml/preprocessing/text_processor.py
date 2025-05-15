import re
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text by removing URLs, special characters, and extra whitespace."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_dataset(df: pd.DataFrame, model_name: str = "distilbert-base-uncased") -> List[Dict]:
    """Tokenize texts and prepare dataset for BERT training."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        label_map = {"positive": 1, "negative": 0, "neutral": 2}
        encodings = []
        
        for _, row in df.iterrows():
            text = clean_text(row["text"])
            label = label_map.get(row["sentiment"], 2)  # Default to neutral if unknown
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            encodings.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": label
            })
        logger.info(f"Prepared {len(encodings)} samples for training")
        return encodings
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        raise