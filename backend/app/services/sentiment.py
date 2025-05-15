import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..models.sentiment import SentimentResponse
from ...ml.preprocessing.text_processor import clean_text
from ...ml.models.bert_classifier import load_bert_model

logger = logging.getLogger(__name__)

async def analyze_sentiment(texts: list) -> SentimentResponse:
    """Analyze sentiment using a custom fine-tuned BERT model."""
    try:
        # Load model and tokenizer
        model = load_bert_model()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model.eval()

        results = []
        for text in texts:
            cleaned_text = clean_text(text)
            inputs = tokenizer(
                cleaned_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Map probabilities to sentiment scores
            positive_score = probs[0][1].item()
            negative_score = probs[0][0].item()
            neutral_score = probs[0][2].item()
            
            results.append({
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            })
        
        logger.info(f"Analyzed sentiment for {len(texts)} texts")
        return SentimentResponse(results=results)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise