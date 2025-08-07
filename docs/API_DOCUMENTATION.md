# Sentiment Analyzer Pro - API Documentation

## Overview

The Sentiment Analyzer Pro API provides real-time sentiment analysis capabilities through a RESTful interface. It supports both single text analysis and batch processing with comprehensive security and monitoring features.

## Base URL

**Production**: `https://localhost/api`  
**Development**: `http://localhost:8000`

## Authentication

Currently, the API operates without authentication for simplicity. In production environments, consider implementing:
- API key authentication
- JWT tokens
- OAuth2 integration

## Rate Limiting

- **Default**: 10 requests per second per IP
- **Burst**: Up to 20 requests in burst mode
- **Headers**: Rate limit status included in response headers

## Endpoints

### 1. Analyze Single Text

Analyze sentiment for a single text input.

**Endpoint**: `POST /analyze`

**Request Body**:
```json
{
  "text": "string",
  "options": {
    "include_probabilities": true,
    "include_metadata": false,
    "model": "bert-base-uncased"
  }
}
```

**Response**:
```json
{
  "text": "I love this product!",
  "label": "POSITIVE",
  "confidence": 0.95,
  "probabilities": [0.02, 0.03, 0.95],
  "processing_time": 0.045,
  "metadata": {
    "model_used": "bert-base-uncased",
    "text_length": 19,
    "preprocessing_applied": true
  }
}
```

**Status Codes**:
- `200 OK`: Successful analysis
- `400 Bad Request`: Invalid input
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### 2. Batch Analysis

Analyze sentiment for multiple texts in a single request.

**Endpoint**: `POST /analyze/batch`

**Request Body**:
```json
{
  "texts": ["Great product!", "Not satisfied", "It's okay"],
  "options": {
    "include_probabilities": true,
    "parallel_processing": true,
    "batch_size": 32
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "Great product!",
      "label": "POSITIVE",
      "confidence": 0.98,
      "probabilities": [0.01, 0.01, 0.98],
      "processing_time": 0.032
    },
    {
      "text": "Not satisfied", 
      "label": "NEGATIVE",
      "confidence": 0.87,
      "probabilities": [0.87, 0.08, 0.05],
      "processing_time": 0.029
    }
  ],
  "summary": {
    "total_texts": 3,
    "total_processing_time": 0.156,
    "average_confidence": 0.89,
    "sentiment_distribution": {
      "POSITIVE": 1,
      "NEGATIVE": 1,
      "NEUTRAL": 1
    }
  }
}
```

### 3. Health Check

Check API health and system status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "3.0.0",
  "uptime": 3600,
  "checks": {
    "database": "healthy",
    "model": "loaded",
    "cache": "operational",
    "memory": "normal"
  },
  "metrics": {
    "requests_processed": 15420,
    "average_response_time": 0.089,
    "cache_hit_rate": 0.78,
    "memory_usage_mb": 1024
  }
}
```

### 4. Model Information

Get information about available models and current configuration.

**Endpoint**: `GET /models`

**Response**:
```json
{
  "available_models": [
    {
      "name": "bert-base-uncased",
      "type": "transformer",
      "status": "loaded",
      "accuracy": 0.94,
      "languages": ["en"],
      "classes": ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    },
    {
      "name": "roberta-base",
      "type": "transformer", 
      "status": "available",
      "accuracy": 0.96,
      "languages": ["en"],
      "classes": ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    }
  ],
  "current_model": "bert-base-uncased",
  "model_cache_size": "2.1GB",
  "last_updated": "2024-01-15T09:00:00Z"
}
```

### 5. Performance Metrics

Get system performance and usage metrics.

**Endpoint**: `GET /metrics`

**Response** (Prometheus format):
```
# HELP sentiment_requests_total Total number of sentiment analysis requests
# TYPE sentiment_requests_total counter
sentiment_requests_total{method="analyze"} 15420

# HELP sentiment_response_time_seconds Response time in seconds
# TYPE sentiment_response_time_seconds histogram
sentiment_response_time_seconds_bucket{le="0.1"} 12500
sentiment_response_time_seconds_bucket{le="0.5"} 15200
sentiment_response_time_seconds_bucket{le="+Inf"} 15420

# HELP sentiment_cache_hits_total Cache hits
# TYPE sentiment_cache_hits_total counter
sentiment_cache_hits_total 12028

# HELP memory_usage_bytes Memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes 1073741824
```

## Request/Response Details

### Request Headers
```
Content-Type: application/json
Accept: application/json
User-Agent: YourApp/1.0
X-Request-ID: unique-request-identifier (optional)
```

### Response Headers
```
Content-Type: application/json
X-Request-ID: unique-request-identifier
X-Response-Time: 0.045
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1642248600
```

## Input Validation

### Text Requirements
- **Minimum length**: 1 character
- **Maximum length**: 10,000 characters
- **Supported characters**: UTF-8 encoded text
- **Languages**: English (primary support)

### Security Filtering
- SQL injection pattern detection
- XSS attack prevention
- Command injection protection
- Content filtering for inappropriate material

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Input text exceeds maximum length",
    "details": {
      "field": "text",
      "max_length": 10000,
      "provided_length": 15000
    }
  },
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Codes
- `VALIDATION_ERROR`: Input validation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MODEL_ERROR`: ML model processing error
- `INTERNAL_ERROR`: Server-side error
- `SECURITY_VIOLATION`: Security check failed

## Code Examples

### Python
```python
import requests

# Single text analysis
response = requests.post(
    "https://localhost/api/analyze",
    json={"text": "I love this product!"},
    headers={"Content-Type": "application/json"}
)
result = response.json()
print(f"Sentiment: {result['label']} (confidence: {result['confidence']:.2f})")
```

### JavaScript
```javascript
// Batch analysis
const response = await fetch('https://localhost/api/analyze/batch', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    texts: ['Great!', 'Not good', 'Okay'],
    options: { include_probabilities: true }
  })
});

const data = await response.json();
console.log('Results:', data.results);
```

### cURL
```bash
# Health check
curl -X GET "https://localhost/api/health" \
  -H "Accept: application/json"

# Analyze sentiment
curl -X POST "https://localhost/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

## SDK Integration

### Python SDK
```python
from sentiment_analyzer_pro import SentimentClient

client = SentimentClient(base_url="https://localhost/api")
result = client.analyze("I love this!")
print(result.label, result.confidence)
```

## Monitoring and Observability

### Health Monitoring
- Endpoint: `/health`
- Frequency: Every 30 seconds
- Timeout: 10 seconds

### Metrics Collection
- Request counts and latencies
- Error rates by type
- Cache performance
- Memory and CPU usage
- Model inference times

### Logging
All requests are logged with:
- Request ID
- Timestamp
- Input length
- Processing time
- Response status
- Error details (if any)

## Performance

### Response Time Targets
- **Single analysis**: < 100ms (95th percentile)
- **Batch analysis**: < 50ms per item
- **Health check**: < 10ms

### Throughput
- **Single requests**: 100+ RPS
- **Batch processing**: 1000+ texts per second
- **Concurrent connections**: 1000+

### Caching
- **Cache TTL**: 1 hour
- **Cache size**: 10,000 entries
- **Hit rate target**: >70%

## Version History

- **v3.0**: Added batch processing, improved caching
- **v2.0**: Enhanced security, validation, monitoring
- **v1.0**: Initial API release with basic functionality