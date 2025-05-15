from pydantic import BaseModel
from typing import List, Dict

class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    results: List[Dict[str, float]]
