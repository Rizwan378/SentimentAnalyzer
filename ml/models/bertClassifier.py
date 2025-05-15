from transformers import AutoModelForSequenceClassification
import torch
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_bert_model(model_path: str = "ml/models/sentiment_model", model_name: str = "distilbert-base-uncased") -> AutoModelForSequenceClassification:
    """Load or initialize a BERT model for sentiment classification."""
    try:
        model_dir = Path(model_path)
        if model_dir.exists():
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # Positive, negative, neutral
            )
            logger.info(f"Initialized new model from {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def save_bert_model(model: AutoModelForSequenceClassification, model_path: str = "ml/models/sentiment_model") -> None:
    """Save the trained BERT model to disk."""
    try:
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        logger.info(f"Saved model to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise