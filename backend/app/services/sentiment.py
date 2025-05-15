import asyncio
import logging
from transformers import pipeline
from ..models.sentiment import SentimentResponse
from ...ml.preprocessing.text_processor import clean_text

logger = logging.getLogger(__name__)

async def analyze_sentiment(texts: list) -> SentimentResponse:
    try:
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        cleaned_texts = [clean_text(text) for text in texts]
        
        results = []
        for text in cleaned_texts:
            prediction = classifier(text)[0]
            score = prediction['score'] if prediction['label'] == 'POSITIVE' else 1 - prediction['score']
            results.append({
                "positive": score,
                "negative": 1 - score
            })
        
        return SentimentResponse(results=results)
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise
