#!/usr/bin/env python3
"""
Basic usage examples for Sentiment Analyzer Pro.

This example demonstrates the core functionality of the sentiment analysis system.
"""

import asyncio
import time
from sentiment_analyzer_pro import SentimentAnalyzer


def basic_example():
    """Basic sentiment analysis example."""
    print("=== Basic Sentiment Analysis Example ===\n")
    
    # Initialize analyzer
    print("Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer(
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device="auto",
        batch_size=16
    )
    
    # Single text analysis
    print("\n1. Single Text Analysis:")
    texts_to_analyze = [
        "I love this product! It's absolutely amazing.",
        "This movie is terrible. Worst acting I've ever seen.",
        "The weather is okay today, nothing special.",
        "I'm so excited about this new opportunity!",
        "I hate waiting in long lines."
    ]
    
    for text in texts_to_analyze:
        result = analyzer.analyze(text)
        print(f"Text: '{text}'")
        print(f"  Sentiment: {result.label}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print()
    
    # Batch analysis
    print("2. Batch Analysis:")
    start_time = time.time()
    batch_results = analyzer.analyze_batch(texts_to_analyze)
    total_time = time.time() - start_time
    
    print(f"Processed {len(texts_to_analyze)} texts in {total_time:.3f}s")
    print(f"Average time per text: {total_time/len(texts_to_analyze):.3f}s")
    
    for i, (text, result) in enumerate(zip(texts_to_analyze, batch_results)):
        print(f"  {i+1}. '{text[:30]}...' -> {result.label} ({result.confidence:.3f})")
    
    # Model information
    print("\n3. Model Information:")
    model_info = analyzer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    analyzer.close()


async def async_example():
    """Asynchronous sentiment analysis example."""
    print("\n=== Async Sentiment Analysis Example ===\n")
    
    analyzer = SentimentAnalyzer(model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Async single analysis
    print("1. Async Single Analysis:")
    text = "I'm really happy with this purchase!"
    result = await analyzer.analyze_async(text)
    print(f"Text: '{text}'")
    print(f"Result: {result.label} (confidence: {result.confidence:.3f})")
    
    # Async batch analysis
    print("\n2. Async Batch Analysis:")
    texts = [
        "Great product, highly recommend!",
        "Poor quality, disappointed.",
        "Average performance, nothing special.",
        "Outstanding service and support!",
        "Terrible experience, avoid this."
    ]
    
    start_time = time.time()
    results = await analyzer.analyze_batch_async(texts)
    total_time = time.time() - start_time
    
    print(f"Async batch processing took {total_time:.3f}s")
    
    for text, result in zip(texts, results):
        print(f"  '{text}' -> {result.label} ({result.confidence:.3f})")
    
    analyzer.close()


def preprocessing_example():
    """Demonstrate text preprocessing features."""
    print("\n=== Text Preprocessing Example ===\n")
    
    # Analyzer with preprocessing enabled
    analyzer_with_prep = SentimentAnalyzer(
        model="distilbert-base-uncased-finetuned-sst-2-english",
        enable_preprocessing=True
    )
    
    # Analyzer without preprocessing
    analyzer_without_prep = SentimentAnalyzer(
        model="distilbert-base-uncased-finetuned-sst-2-english",
        enable_preprocessing=False
    )
    
    # Test with messy text
    messy_texts = [
        "I LOVE this product!!! 😍😍😍 https://example.com #awesome",
        "This is sooooo bad 😠 @company fix this NOW!!!",
        "It's okaaaay... nothing special 🤷‍♀️ could be better",
    ]
    
    print("Comparing with and without preprocessing:")
    
    for text in messy_texts:
        result_with = analyzer_with_prep.analyze(text)
        result_without = analyzer_without_prep.analyze(text)
        
        print(f"\nOriginal: '{text}'")
        print(f"With preprocessing: {result_with.label} ({result_with.confidence:.3f})")
        print(f"Without preprocessing: {result_without.label} ({result_without.confidence:.3f})")
    
    analyzer_with_prep.close()
    analyzer_without_prep.close()


def performance_comparison():
    """Compare performance across different batch sizes."""
    print("\n=== Performance Comparison ===\n")
    
    # Test data
    test_texts = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "Average performance, okay I guess.",
        "Outstanding customer service!",
        "Waste of money, don't buy this.",
        "Pretty good value for the price.",
        "Excellent build quality and design.",
        "Poor user experience overall.",
        "Highly recommend this to everyone.",
        "Could be better, but not bad."
    ] * 10  # 100 texts total
    
    batch_sizes = [1, 8, 16, 32, 64]
    
    print(f"Testing with {len(test_texts)} texts:")
    
    for batch_size in batch_sizes:
        analyzer = SentimentAnalyzer(
            model="distilbert-base-uncased-finetuned-sst-2-english",
            batch_size=batch_size
        )
        
        start_time = time.time()
        results = analyzer.analyze_batch(test_texts)
        total_time = time.time() - start_time
        
        print(f"Batch size {batch_size:2d}: {total_time:.3f}s total, {total_time/len(test_texts)*1000:.2f}ms per text")
        
        analyzer.close()


if __name__ == "__main__":
    # Run basic example
    basic_example()
    
    # Run async example  
    asyncio.run(async_example())
    
    # Run preprocessing example
    preprocessing_example()
    
    # Run performance comparison
    performance_comparison()
    
    print("\n=== All Examples Complete ===")