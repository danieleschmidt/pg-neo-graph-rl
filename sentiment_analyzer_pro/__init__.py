"""
Sentiment Analyzer Pro: Enterprise-grade sentiment analysis toolkit

A production-ready sentiment analysis system that combines state-of-the-art
transformer models with federated learning capabilities for privacy-preserving,
scalable sentiment analysis across distributed environments.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .core.analyzer import SentimentAnalyzer, SentimentResult
from .models.bert import BERTSentimentModel
from .preprocessing.text_processor import TextPreprocessor
from .federated.trainer import FederatedSentimentTrainer

__all__ = [
    "SentimentAnalyzer",
    "SentimentResult", 
    "BERTSentimentModel",
    "TextPreprocessor",
    "FederatedSentimentTrainer"
]