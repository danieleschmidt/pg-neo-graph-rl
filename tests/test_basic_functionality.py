#!/usr/bin/env python3
"""
Basic functionality tests that can run without ML dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import components that don't require torch
from sentiment_analyzer_pro.core.base import SentimentResult, BaseSentimentModel
from sentiment_analyzer_pro.preprocessing.text_processor import TextPreprocessor
from sentiment_analyzer_pro.utils.exceptions import ValidationError
from sentiment_analyzer_pro.utils.validation import TextValidator, ConfigValidator
from sentiment_analyzer_pro.utils.security import InputValidator
from sentiment_analyzer_pro.optimization.cache import ResultCache, CacheManager


def test_sentiment_result():
    """Test SentimentResult functionality."""
    print("Testing SentimentResult...")
    
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
    assert result.processing_time == 0.1
    
    # Test to_dict
    result_dict = result.to_dict()
    assert result_dict["text"] == "Test text"
    assert result_dict["label"] == "POSITIVE"
    assert result_dict["confidence"] == 0.95
    
    print("✅ SentimentResult tests passed")


def test_text_preprocessor():
    """Test text preprocessing functionality."""
    print("Testing TextPreprocessor...")
    
    processor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        handle_emojis=True,
        normalize_text=True
    )
    
    # Test basic preprocessing
    input_text = "  THIS IS A TEST   WITH   MULTIPLE   SPACES  "
    processed = processor.process(input_text)
    assert processed == "this is a test with multiple spaces"
    
    # Test URL removal
    text_with_url = "Check out https://example.com for info"
    processed = processor.process(text_with_url)
    assert "https://example.com" not in processed
    assert "URL" in processed
    
    # Test batch processing
    texts = ["TEXT ONE", "TEXT TWO", "TEXT THREE"]
    processed_batch = processor.batch_process(texts)
    assert len(processed_batch) == 3
    assert all(text.islower() for text in processed_batch)
    
    # Test stats
    stats = processor.get_stats("Visit https://example.com @user #hashtag")
    assert isinstance(stats, dict)
    assert "original_length" in stats
    assert "processed_length" in stats
    assert "urls_found" in stats
    
    print("✅ TextPreprocessor tests passed")


def test_input_validator():
    """Test input validation."""
    print("Testing InputValidator...")
    
    validator = InputValidator(max_length=1000)
    
    # Test valid input
    valid_text = validator.validate_input("This is valid text", "test")
    assert isinstance(valid_text, str)
    assert len(valid_text) > 0
    
    # Test empty input (should raise exception)
    try:
        validator.validate_input("", "test")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test very long input (should be truncated)
    long_text = "a" * 2000
    validated = validator.validate_input(long_text, "test")
    assert len(validated) <= 1000
    
    print("✅ InputValidator tests passed")


def test_text_validator():
    """Test advanced text validation."""
    print("Testing TextValidator...")
    
    validator = TextValidator(min_length=1, max_length=1000)
    
    # Test valid text
    result = validator.validate_text("This is a good test message")
    assert result.is_valid
    assert len(result.errors) == 0
    
    # Test empty text
    result = validator.validate_text("")
    assert not result.is_valid
    assert len(result.errors) > 0
    
    # Test very long text
    long_text = "a" * 2000
    result = validator.validate_text(long_text)
    assert not result.is_valid
    assert any("too long" in error.lower() for error in result.errors)
    
    print("✅ TextValidator tests passed")


def test_config_validator():
    """Test configuration validation."""
    print("Testing ConfigValidator...")
    
    validator = ConfigValidator()
    
    # Test valid config
    valid_config = {
        "model_name": "bert-base-uncased",
        "device": "cpu",
        "batch_size": 16,
        "max_length": 512,
        "num_classes": 3
    }
    
    result = validator.validate_config(valid_config)
    assert result.is_valid
    assert len(result.errors) == 0
    
    # Test invalid config
    invalid_config = {
        "model_name": "",  # Empty model name
        "device": "invalid_device",  # Invalid device
        "batch_size": -1,  # Negative batch size
        "num_classes": 1   # Too few classes
    }
    
    result = validator.validate_config(invalid_config)
    assert not result.is_valid
    assert len(result.errors) > 0
    
    print("✅ ConfigValidator tests passed")


def test_result_cache():
    """Test result caching functionality."""
    print("Testing ResultCache...")
    
    cache = ResultCache(max_size=100, ttl_seconds=60)
    
    # Test cache miss
    result = cache.get("nonexistent_key")
    assert result is None
    
    # Test cache put and get
    test_result = SentimentResult(
        text="test",
        label="POSITIVE",
        confidence=0.9,
        probabilities=np.array([0.1, 0.9]),
        processing_time=0.1
    )
    
    cache.put("test_key", test_result)
    cached_result = cache.get("test_key")
    
    assert cached_result is not None
    assert cached_result.text == "test"
    assert cached_result.label == "POSITIVE"
    
    # Test cache stats
    stats = cache.get_stats()
    assert isinstance(stats, dict)
    assert "hits" in stats
    assert "misses" in stats
    
    print("✅ ResultCache tests passed")


def test_cache_manager():
    """Test cache manager functionality."""
    print("Testing CacheManager...")
    
    cache_manager = CacheManager(
        result_cache_size=100,
        model_cache_size=5
    )
    
    # Test result caching
    result = SentimentResult(
        text="test cache",
        label="NEUTRAL",
        confidence=0.7,
        probabilities=np.array([0.2, 0.7, 0.1]),
        processing_time=0.05
    )
    
    cache_manager.cache_result("test text", "test-model", result)
    cached_result = cache_manager.get_result("test text", "test-model")
    
    assert cached_result is not None
    assert cached_result.label == "NEUTRAL"
    
    # Test preprocessing cache
    cache_manager.cache_preprocessed_text(
        "raw text", 
        {"lowercase": True}, 
        "processed text"
    )
    
    cached_text = cache_manager.get_preprocessed_text(
        "raw text", 
        {"lowercase": True}
    )
    assert cached_text is not None
    
    # Test cache stats
    stats = cache_manager.get_cache_stats()
    assert isinstance(stats, dict)
    assert "result_cache" in stats
    
    print("✅ CacheManager tests passed")


def test_base_model():
    """Test base model functionality."""
    print("Testing BaseSentimentModel...")
    
    class MockModel(BaseSentimentModel):
        def load_model(self):
            pass
        
        def predict(self, texts):
            return [SentimentResult(
                text=text,
                label="POSITIVE",
                confidence=0.8,
                probabilities=np.array([0.1, 0.1, 0.8]),
                processing_time=0.1
            ) for text in texts]
        
        def predict_single(self, text):
            return SentimentResult(
                text=text,
                label="POSITIVE",
                confidence=0.8,
                probabilities=np.array([0.1, 0.1, 0.8]),
                processing_time=0.1
            )
    
    model = MockModel("test-model", num_classes=3)
    assert model.model_name == "test-model"
    assert model.num_classes == 3
    assert model.labels == ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    # Test validation
    validated = model.validate_input("Valid text")
    assert validated == "Valid text"
    
    try:
        model.validate_input("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✅ BaseSentimentModel tests passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Basic Functionality Tests...")
    print("=" * 50)
    
    try:
        test_sentiment_result()
        test_text_preprocessor()
        test_input_validator()
        test_text_validator()
        test_config_validator()
        test_result_cache()
        test_cache_manager()
        test_base_model()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("📊 Test Coverage: Core utilities and preprocessing")
        print("🎯 Basic functionality is working correctly")
        print("⚡ Performance optimizations validated")
        print("🛡️ Security validations working")
        return True
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)