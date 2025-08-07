"""Base classes for sentiment analysis models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    
    text: str
    label: str
    confidence: float
    probabilities: np.ndarray
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": float(self.confidence),
            "probabilities": self.probabilities.tolist() if isinstance(self.probabilities, np.ndarray) else self.probabilities,
            "processing_time": self.processing_time,
            "metadata": self.metadata or {}
        }


class BaseSentimentModel(ABC):
    """Abstract base class for sentiment analysis models."""
    
    def __init__(self, model_name: str, num_classes: int = 3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"] if num_classes == 3 else [f"CLASS_{i}" for i in range(num_classes)]
        self.model = None
        self.tokenizer = None
        self.device = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the sentiment analysis model."""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """Predict sentiment for a batch of texts."""
        pass
    
    @abstractmethod
    def predict_single(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        if len(text) == 0:
            text = "neutral"
            
        return text
    
    def validate_input(self, text: str) -> str:
        """Validate and sanitize input text."""
        if not text:
            raise ValueError("Input text cannot be empty")
        
        if len(text) > 10000:  # Reasonable limit
            text = text[:10000]
            
        return self.preprocess_text(text)