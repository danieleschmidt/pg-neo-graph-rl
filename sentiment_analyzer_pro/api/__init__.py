"""FastAPI-based REST API for sentiment analysis."""

from .app import create_app, app
from .models import SentimentRequest, SentimentResponse, BatchSentimentRequest

__all__ = ["create_app", "app", "SentimentRequest", "SentimentResponse", "BatchSentimentRequest"]