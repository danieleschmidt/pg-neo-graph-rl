"""Main sentiment analyzer implementation."""

import asyncio
import logging
import time
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
try:
    import torch
except ImportError:
    torch = None

from ..models.bert import BERTSentimentModel
from ..preprocessing.text_processor import TextPreprocessor
from .base import SentimentResult, BaseSentimentModel
from ..utils.device_manager import DeviceManager
from ..utils.exceptions import SentimentAnalysisError


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Main sentiment analyzer class providing sync and async analysis capabilities.
    
    Features:
    - Multiple model support (BERT, RoBERTa, DistilBERT)
    - Automatic device detection (GPU/CPU)
    - Batch processing optimization
    - Async processing for high throughput
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        num_classes: int = 3,
        enable_preprocessing: bool = True
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model: Model name or path
            device: Device to use ("auto", "cpu", "cuda", "mps")
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            num_classes: Number of sentiment classes
            enable_preprocessing: Whether to enable text preprocessing
        """
        self.model_name = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_classes = num_classes
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize device manager
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device(device)
        
        # Initialize text preprocessor
        if enable_preprocessing:
            self.preprocessor = TextPreprocessor(
                lowercase=True,
                remove_urls=True,
                handle_emojis=True,
                normalize_text=True,
                remove_special_chars=False  # Keep some special chars for sentiment
            )
        else:
            self.preprocessor = None
            
        # Initialize model
        self._load_model()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"SentimentAnalyzer initialized with model '{model}' on device '{self.device}'")
    
    def _load_model(self) -> None:
        """Load the sentiment analysis model."""
        try:
            # For now, use BERT as the default model
            self.model = BERTSentimentModel(
                model_name=self.model_name,
                num_classes=self.num_classes,
                device=self.device,
                max_length=self.max_length
            )
            self.model.load_model()
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise SentimentAnalysisError(f"Model loading failed: {e}")
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with analysis results
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        start_time = time.time()
        
        try:
            # Preprocess text if enabled
            if self.preprocessor:
                processed_text = self.preprocessor.process(text)
            else:
                processed_text = text.strip()
            
            # Get prediction from model
            result = self.model.predict_single(processed_text)
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise SentimentAnalysisError(f"Analysis failed: {e}")
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
            
        start_time = time.time()
        
        try:
            # Preprocess texts if enabled
            if self.preprocessor:
                processed_texts = [
                    self.preprocessor.process(text) if text and text.strip() else "neutral"
                    for text in texts
                ]
            else:
                processed_texts = [
                    text.strip() if text and text.strip() else "neutral"
                    for text in texts
                ]
            
            # Process in batches for memory efficiency
            results = []
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                batch_originals = texts[i:i + self.batch_size]
                
                batch_results = self.model.predict(batch)
                
                # Update with original texts and timing
                for result, original_text in zip(batch_results, batch_originals):
                    result.text = original_text
                    result.processing_time = (time.time() - start_time) / len(texts)
                
                results.extend(batch_results)
            
            logger.info(f"Processed {len(texts)} texts in {time.time() - start_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise SentimentAnalysisError(f"Batch analysis failed: {e}")
    
    async def analyze_async(self, text: str) -> SentimentResult:
        """
        Asynchronously analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze, text)
    
    async def analyze_batch_async(self, texts: List[str]) -> List[SentimentResult]:
        """
        Asynchronously analyze sentiment of multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of SentimentResult objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_batch, texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "preprocessing_enabled": self.enable_preprocessing,
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("SentimentAnalyzer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()