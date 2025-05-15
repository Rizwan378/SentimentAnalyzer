from fastapi import FastAPI
from .api.endpoints import sentiment
from .core.config import settings
from .core.logger import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analyzer API")

app.include_router(sentiment.router, prefix="/api/v1")
