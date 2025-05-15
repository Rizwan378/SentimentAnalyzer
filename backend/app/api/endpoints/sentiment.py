from fastapi import APIRouter, HTTPException
from ...services.sentiment import analyze_sentiment
from ...models.sentiment import SentimentRequest, SentimentResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=SentimentResponse)
async def analyze(request: SentimentRequest):
    try:
        logger.info(f"Analyzing sentiment for {len(request.texts)} texts")
        result = await analyze_sentiment(request.texts)
        return result
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
