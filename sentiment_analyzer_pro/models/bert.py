"""BERT-based sentiment analysis model implementation."""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

from ..core.base import BaseSentimentModel, SentimentResult
from ..utils.exceptions import ModelLoadError, PredictionError


logger = logging.getLogger(__name__)


class BERTSentimentModel(BaseSentimentModel):
    """
    BERT-based sentiment analysis model.
    
    Supports various BERT variants including:
    - BERT (bert-base-uncased, bert-large-uncased)
    - RoBERTa (roberta-base, roberta-large)  
    - DistilBERT (distilbert-base-uncased)
    - Custom fine-tuned models
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        num_classes: int = 3,
        device: Optional[torch.device] = None,
        max_length: int = 512,
        use_fast_tokenizer: bool = True
    ):
        """
        Initialize BERT sentiment model.
        
        Args:
            model_name: Hugging Face model name or local path
            num_classes: Number of sentiment classes
            device: PyTorch device (auto-detected if None)
            max_length: Maximum sequence length
            use_fast_tokenizer: Whether to use fast tokenizer
        """
        super().__init__(model_name, num_classes)
        
        self.max_length = max_length
        self.use_fast_tokenizer = use_fast_tokenizer
        
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        logger.info(f"Initialized BERTSentimentModel with {model_name} on {self.device}")
    
    def load_model(self) -> None:
        """Load the BERT model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast_tokenizer
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                return_dict=True
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Update labels based on model config if available
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                self.labels = list(self.model.config.id2label.values())
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info(f"Model loaded successfully. Labels: {self.labels}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def predict_single(self, text: str) -> SentimentResult:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            SentimentResult with prediction
        """
        start_time = time.time()
        
        try:
            # Validate input
            text = self.validate_input(text)
            
            # Get prediction using pipeline
            outputs = self.pipeline(text)
            
            # Extract results (pipeline returns list of dicts with label and score)
            scores = outputs[0]  # First (and only) text
            
            # Convert to numpy array and find best prediction
            probabilities = np.array([score['score'] for score in scores])
            label_names = [score['label'] for score in scores]
            
            # Find the highest scoring label
            max_idx = np.argmax(probabilities)
            predicted_label = label_names[max_idx]
            confidence = probabilities[max_idx]
            
            # Create result
            result = SentimentResult(
                text=text,
                label=predicted_label,
                confidence=float(confidence),
                probabilities=probabilities,
                processing_time=time.time() - start_time,
                metadata={
                    "model_name": self.model_name,
                    "device": str(self.device),
                    "all_scores": scores
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for text: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
            
        start_time = time.time()
        
        try:
            # Validate inputs
            validated_texts = [self.validate_input(text) for text in texts]
            
            # Get predictions using pipeline
            outputs = self.pipeline(validated_texts)
            
            results = []
            for i, (text, output) in enumerate(zip(texts, outputs)):
                # Extract scores for this text
                scores = output
                probabilities = np.array([score['score'] for score in scores])
                label_names = [score['label'] for score in scores]
                
                # Find the highest scoring label
                max_idx = np.argmax(probabilities)
                predicted_label = label_names[max_idx]
                confidence = probabilities[max_idx]
                
                # Create result
                result = SentimentResult(
                    text=text,
                    label=predicted_label,
                    confidence=float(confidence),
                    probabilities=probabilities,
                    processing_time=(time.time() - start_time) / len(texts),
                    metadata={
                        "model_name": self.model_name,
                        "device": str(self.device),
                        "batch_index": i,
                        "all_scores": scores
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise PredictionError(f"Batch prediction failed: {e}")
    
    def get_model_details(self) -> Dict[str, Any]:
        """Get detailed model information."""
        details = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "labels": self.labels,
            "device": str(self.device),
            "max_length": self.max_length,
            "model_loaded": self.model is not None
        }
        
        if self.model is not None:
            details.update({
                "model_type": self.model.config.model_type,
                "hidden_size": self.model.config.hidden_size,
                "num_layers": self.model.config.num_hidden_layers,
                "num_attention_heads": self.model.config.num_attention_heads,
                "vocab_size": self.model.config.vocab_size,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
            
        return details
    
    def save_model(self, path: str) -> None:
        """Save model and tokenizer to path."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
            
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        num_classes: int = 3,
        device: Optional[torch.device] = None
    ) -> 'BERTSentimentModel':
        """Load model from saved path."""
        model = cls(path, num_classes, device)
        model.load_model()
        return model