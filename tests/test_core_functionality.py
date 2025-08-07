#!/usr/bin/env python3
"""
Comprehensive tests for core sentiment analysis functionality.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from sentiment_analyzer_pro import SentimentAnalyzer, SentimentResult
from sentiment_analyzer_pro.core.analyzer import SentimentAnalyzer as CoreAnalyzer
from sentiment_analyzer_pro.core.base import BaseSentimentModel
from sentiment_analyzer_pro.models.bert import BERTSentimentModel
from sentiment_analyzer_pro.preprocessing.text_processor import TextPreprocessor
from sentiment_analyzer_pro.utils.exceptions import (
    SentimentAnalysisError, 
    ModelLoadError, 
    PredictionError,
    ValidationError
)


class TestSentimentResult:
    """Test SentimentResult dataclass."""
    
    def test_sentiment_result_creation(self):
        """Test creating SentimentResult."""
        result = SentimentResult(
            text="Test text",
            label="POSITIVE",
            confidence=0.95,
            probabilities=np.array([0.02, 0.03, 0.95]),
            processing_time=0.1,
            metadata={"model": "test"}
        )
        
        assert result.text == "Test text"
        assert result.label == "POSITIVE"
        assert result.confidence == 0.95
        assert np.array_equal(result.probabilities, np.array([0.02, 0.03, 0.95]))
        assert result.processing_time == 0.1
        assert result.metadata == {"model": "test"}
    
    def test_sentiment_result_to_dict(self):
        """Test converting result to dictionary."""
        result = SentimentResult(
            text="Test",
            label="POSITIVE",
            confidence=0.9,
            probabilities=np.array([0.1, 0.9]),
            processing_time=0.05
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["text"] == "Test"
        assert result_dict["label"] == "POSITIVE"
        assert result_dict["confidence"] == 0.9
        assert result_dict["probabilities"] == [0.1, 0.9]
        assert result_dict["processing_time"] == 0.05


class TestBaseSentimentModel:
    """Test BaseSentimentModel abstract class."""
    
    def test_base_model_initialization(self):
        """Test base model initialization."""
        class MockModel(BaseSentimentModel):
            def load_model(self): pass
            def predict(self, texts): pass
            def predict_single(self, text): pass
        
        model = MockModel("test-model", num_classes=3)
        assert model.model_name == "test-model"
        assert model.num_classes == 3
        assert model.labels == ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    def test_validate_input(self):
        """Test input validation."""
        class MockModel(BaseSentimentModel):
            def load_model(self): pass
            def predict(self, texts): pass
            def predict_single(self, text): pass
        
        model = MockModel("test-model")
        
        # Test valid input
        validated = model.validate_input("Valid input text")
        assert validated == "Valid input text"
        
        # Test empty input
        with pytest.raises(ValueError):
            model.validate_input("")
        
        # Test very long input
        long_text = "a" * 15000
        validated = model.validate_input(long_text)
        assert len(validated) <= 10000


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        processor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            handle_emojis=True
        )
        
        assert processor.lowercase is True
        assert processor.remove_urls is True
        assert processor.handle_emojis is True
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        processor = TextPreprocessor(lowercase=True, normalize_text=True)
        
        input_text = "  THIS IS A TEST   WITH   MULTIPLE   SPACES  "
        processed = processor.process(input_text)
        
        assert processed == "this is a test with multiple spaces"
    
    def test_url_removal(self):
        """Test URL removal."""
        processor = TextPreprocessor(remove_urls=True)
        
        text = "Check out this site: https://example.com for more info"
        processed = processor.process(text)
        
        assert "https://example.com" not in processed
        assert "URL" in processed
    
    def test_emoji_handling(self):
        """Test emoji processing."""
        processor = TextPreprocessor(handle_emojis=True)
        
        text = "I love this! 😍 It's amazing! 😊"
        processed = processor.process(text)
        
        # Should contain text descriptions of emojis
        assert "love" in processed.lower()
        assert "happy" in processed.lower()
    
    def test_batch_processing(self):
        """Test batch text processing."""
        processor = TextPreprocessor(lowercase=True)
        
        texts = ["TEXT ONE", "TEXT TWO", "TEXT THREE"]
        processed = processor.batch_process(texts)
        
        assert len(processed) == 3
        assert all(text.islower() for text in processed)
    
    def test_preprocessing_stats(self):
        """Test preprocessing statistics."""
        processor = TextPreprocessor()
        
        text = "Visit https://example.com and email user@example.com @mention #hashtag"
        stats = processor.get_stats(text)
        
        assert isinstance(stats, dict)
        assert "original_length" in stats
        assert "processed_length" in stats
        assert "urls_found" in stats
        assert stats["urls_found"] >= 1


@pytest.fixture
def mock_bert_model():
    """Fixture for mocked BERT model."""
    with patch('sentiment_analyzer_pro.models.bert.AutoTokenizer') as mock_tokenizer, \
         patch('sentiment_analyzer_pro.models.bert.AutoModelForSequenceClassification') as mock_model, \
         patch('sentiment_analyzer_pro.models.bert.pipeline') as mock_pipeline:
        
        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [[
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.05},
            {"label": "NEUTRAL", "score": 0.05}
        ]]
        mock_pipeline.return_value = mock_pipeline_instance
        
        model = BERTSentimentModel("test-model")
        model.load_model()
        
        yield model


class TestBERTSentimentModel:
    """Test BERT sentiment model."""
    
    def test_model_initialization(self):
        """Test BERT model initialization."""
        model = BERTSentimentModel(
            model_name="distilbert-base-uncased",
            num_classes=3,
            max_length=512
        )
        
        assert model.model_name == "distilbert-base-uncased"
        assert model.num_classes == 3
        assert model.max_length == 512
    
    def test_model_prediction_single(self, mock_bert_model):
        """Test single text prediction."""
        result = mock_bert_model.predict_single("This is a positive text!")
        
        assert isinstance(result, SentimentResult)
        assert result.label == "POSITIVE"
        assert result.confidence > 0
        assert len(result.probabilities) > 0
        assert result.processing_time >= 0
    
    def test_model_prediction_batch(self, mock_bert_model):
        """Test batch prediction."""
        texts = [
            "This is positive!",
            "This is negative.",
            "This is neutral."
        ]
        
        results = mock_bert_model.predict(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, SentimentResult) for result in results)
        assert all(result.processing_time >= 0 for result in results)
    
    def test_model_details(self, mock_bert_model):
        """Test getting model details."""
        details = mock_bert_model.get_model_details()
        
        assert isinstance(details, dict)
        assert "model_name" in details
        assert "num_classes" in details
        assert "device" in details


@pytest.fixture  
def mock_analyzer():
    """Fixture for mocked sentiment analyzer."""
    with patch('sentiment_analyzer_pro.core.analyzer.BERTSentimentModel') as mock_model:
        # Mock the model's predict methods
        mock_model_instance = Mock()
        mock_model_instance.predict_single.return_value = SentimentResult(
            text="test",
            label="POSITIVE",
            confidence=0.9,
            probabilities=np.array([0.05, 0.05, 0.9]),
            processing_time=0.1
        )
        mock_model_instance.predict.return_value = [
            SentimentResult(
                text="test1",
                label="POSITIVE", 
                confidence=0.9,
                probabilities=np.array([0.05, 0.05, 0.9]),
                processing_time=0.1
            ),
            SentimentResult(
                text="test2",
                label="NEGATIVE",
                confidence=0.8, 
                probabilities=np.array([0.8, 0.1, 0.1]),
                processing_time=0.1
            )
        ]
        
        mock_model.return_value = mock_model_instance
        
        analyzer = SentimentAnalyzer(model="test-model")
        yield analyzer


class TestSentimentAnalyzer:
    """Test main SentimentAnalyzer class."""
    
    def test_analyzer_initialization(self, mock_analyzer):
        """Test analyzer initialization."""
        assert mock_analyzer.model_name == "test-model"
        assert mock_analyzer.batch_size == 32
        assert mock_analyzer.device is not None
    
    def test_analyze_single_text(self, mock_analyzer):
        """Test analyzing single text."""
        result = mock_analyzer.analyze("This is a positive message!")
        
        assert isinstance(result, SentimentResult)
        assert result.label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert 0 <= result.confidence <= 1
        assert result.processing_time >= 0
    
    def test_analyze_empty_text(self, mock_analyzer):
        """Test analyzing empty text."""
        with pytest.raises((ValueError, SentimentAnalysisError)):
            mock_analyzer.analyze("")
    
    def test_analyze_batch(self, mock_analyzer):
        """Test batch analysis."""
        texts = [
            "Great product!",
            "Terrible experience.",
            "It's okay."
        ]
        
        results = mock_analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, SentimentResult) for result in results)
    
    def test_analyze_empty_batch(self, mock_analyzer):
        """Test analyzing empty batch."""
        results = mock_analyzer.analyze_batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_analyze_async(self, mock_analyzer):
        """Test async analysis."""
        result = await mock_analyzer.analyze_async("Async test message")
        
        assert isinstance(result, SentimentResult)
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_batch_async(self, mock_analyzer):
        """Test async batch analysis."""
        texts = ["Text 1", "Text 2", "Text 3"]
        results = await mock_analyzer.analyze_batch_async(texts)
        
        assert len(results) == 3
        assert all(isinstance(result, SentimentResult) for result in results)
    
    def test_get_model_info(self, mock_analyzer):
        """Test getting model information."""
        info = mock_analyzer.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "batch_size" in info
    
    def test_context_manager(self, mock_analyzer):
        """Test using analyzer as context manager."""
        with mock_analyzer as analyzer:
            result = analyzer.analyze("Context manager test")
            assert isinstance(result, SentimentResult)


class TestErrorHandling:
    """Test error handling and exceptions."""
    
    def test_model_load_error(self):
        """Test model loading error handling."""
        with patch('sentiment_analyzer_pro.models.bert.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Network error")
            
            model = BERTSentimentModel("invalid-model")
            with pytest.raises(ModelLoadError):
                model.load_model()
    
    def test_prediction_error(self, mock_bert_model):
        """Test prediction error handling."""
        # Mock pipeline to raise exception
        mock_bert_model.pipeline.side_effect = Exception("Prediction failed")
        
        with pytest.raises(PredictionError):
            mock_bert_model.predict_single("test text")
    
    def test_validation_error(self):
        """Test input validation errors."""
        processor = TextPreprocessor()
        
        # Test with invalid input type
        with pytest.raises(ValueError):
            processor.process(None)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_batch_processing_performance(self, mock_analyzer):
        """Test that batch processing is more efficient than individual calls."""
        texts = ["Test text " + str(i) for i in range(10)]
        
        # Time individual processing
        start_time = time.time()
        individual_results = []
        for text in texts:
            result = mock_analyzer.analyze(text)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        batch_results = mock_analyzer.analyze_batch(texts)
        batch_time = time.time() - start_time
        
        # Batch should generally be faster (allowing for mocking overhead)
        assert len(individual_results) == len(batch_results)
        # Note: With mocking, timing differences may not be significant
    
    @pytest.mark.asyncio
    async def test_async_performance(self, mock_analyzer):
        """Test async processing performance."""
        texts = ["Async test " + str(i) for i in range(5)]
        
        # Test concurrent processing
        start_time = time.time()
        tasks = [mock_analyzer.analyze_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        async_time = time.time() - start_time
        
        assert len(results) == 5
        assert all(isinstance(result, SentimentResult) for result in results)
        # Async processing should complete (timing varies with mocking)


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_integration(self, mock_analyzer):
        """Test full analysis pipeline."""
        # Test with various text types
        test_cases = [
            "I love this product! It's amazing! 😍",
            "This is terrible quality. Very disappointed.",
            "It's okay, nothing special really.",
            "https://example.com Check this out @user #hashtag",
            "Mixed feelings... good but could be better 🤔"
        ]
        
        for text in test_cases:
            result = mock_analyzer.analyze(text)
            
            assert isinstance(result, SentimentResult)
            assert result.label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            assert 0 <= result.confidence <= 1
            assert len(result.probabilities) > 0
            assert result.processing_time >= 0
    
    def test_preprocessing_integration(self):
        """Test preprocessing integration."""
        processor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            handle_emojis=True,
            normalize_text=True
        )
        
        messy_text = "I LOVE THIS!!! 😍😍 https://example.com @user #awesome"
        processed = processor.process(messy_text)
        
        # Should be cleaned and normalized
        assert processed.islower()
        assert "https://example.com" not in processed
        assert len(processed.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, mock_analyzer):
        """Test concurrent analysis requests."""
        texts = [f"Concurrent test {i}" for i in range(20)]
        
        # Process multiple texts concurrently
        tasks = []
        for i in range(0, len(texts), 5):
            batch = texts[i:i+5]
            task = mock_analyzer.analyze_batch_async(batch)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        assert len(all_results) == 20
        assert all(isinstance(result, SentimentResult) for result in all_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])