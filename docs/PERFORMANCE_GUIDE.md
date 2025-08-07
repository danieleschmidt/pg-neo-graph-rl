# Sentiment Analyzer Pro - Performance Guide

## Overview

This guide provides comprehensive information on optimizing, monitoring, and maintaining the performance of the Sentiment Analyzer Pro system. It covers configuration tuning, resource optimization, and scaling strategies.

## Performance Architecture

### System Components Performance Impact

```
┌─────────────────────────────────────────────────────────────┐
│                    Performance Stack                        │
├─────────────────────────────────────────────────────────────┤
│ Load Balancer (Nginx)          │ Response Time: ~2ms       │
├─────────────────────────────────────────────────────────────┤
│ API Layer (FastAPI)             │ Response Time: ~5ms       │
├─────────────────────────────────────────────────────────────┤
│ Caching Layer (Redis/Memory)    │ Hit Time: ~1ms            │
├─────────────────────────────────────────────────────────────┤
│ ML Model (BERT/RoBERTa)         │ Inference: ~40-80ms       │
├─────────────────────────────────────────────────────────────┤
│ Text Processing                 │ Processing: ~2-5ms        │
└─────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Metric | Target | Measurement |
|--------|---------|------------|
| Single Request Response Time | < 100ms (95th percentile) | End-to-end API response |
| Batch Processing Throughput | > 1000 texts/second | Concurrent batch processing |
| Cache Hit Rate | > 70% | Cache performance ratio |
| Memory Usage | < 4GB per instance | Peak memory consumption |
| CPU Usage | < 80% average | Sustained load performance |
| Startup Time | < 30 seconds | Model loading + service ready |

## Configuration Optimization

### Application Configuration

**Core Settings** (`sentiment_analyzer_pro/config.py`):
```python
class PerformanceConfig:
    # Worker configuration
    MAX_WORKERS = 4  # Adjust based on CPU cores
    BATCH_SIZE = 32  # Optimal for most hardware
    
    # Memory management
    MODEL_CACHE_SIZE = "2GB"
    RESULT_CACHE_SIZE = 10000
    CACHE_TTL = 3600  # 1 hour
    
    # Processing optimization
    ENABLE_GPU = False  # Enable if GPU available
    ASYNC_PROCESSING = True
    PREFETCH_MODELS = True
    
    # Resource limits
    MAX_INPUT_LENGTH = 10000
    MAX_BATCH_SIZE = 100
    REQUEST_TIMEOUT = 30
```

**Environment Variables**:
```bash
# Resource allocation
export MAX_WORKERS=4
export BATCH_SIZE=32
export MEMORY_LIMIT="4g"

# Performance features
export ENABLE_CACHING=true
export ENABLE_GPU=false
export ASYNC_PROCESSING=true

# Optimization settings
export PREFETCH_MODELS=true
export AGGRESSIVE_CACHING=true
export MEMORY_OPTIMIZATION=true
```

### Docker Configuration

**Resource Limits** (`docker-compose.production.yml`):
```yaml
services:
  sentiment-analyzer:
    deploy:
      resources:
        limits:
          cpus: '2.0'      # Adjust based on available cores
          memory: 4G       # 4GB for BERT models + overhead
        reservations:
          cpus: '1.0'      # Minimum guaranteed resources
          memory: 2G
    environment:
      - OMP_NUM_THREADS=2  # Control threading for NumPy/PyTorch
      - TOKENIZERS_PARALLELISM=false  # Prevent tokenizer threading issues
```

**GPU Configuration** (if available):
```yaml
  sentiment-analyzer-gpu:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Memory Optimization

### Model Memory Management

**Model Loading Optimization**:
```python
# sentiment_analyzer_pro/optimization/memory.py
class ModelMemoryManager:
    def __init__(self):
        self.model_cache = {}
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def load_model_efficiently(self, model_name: str):
        # Check memory usage before loading
        if self._get_memory_usage() > self.memory_threshold:
            self._cleanup_unused_models()
        
        # Load model with memory mapping
        model = transformers.AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto"
        )
        return model
```

**Memory Monitoring**:
```python
import psutil
import gc

def monitor_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    # Trigger cleanup if memory usage is high
    if memory_percent > 80:
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU
```

### Garbage Collection Optimization

**Automatic Memory Cleanup**:
```python
import gc
import threading

class MemoryManager:
    def __init__(self, cleanup_interval=300):  # 5 minutes
        self.cleanup_interval = cleanup_interval
        self.start_cleanup_timer()
    
    def cleanup_memory(self):
        # Force garbage collection
        gc.collect()
        
        # Clear model caches if needed
        if self._memory_usage_high():
            self._clear_model_cache()
        
        # Clear result cache partially
        self.result_cache.cleanup_expired()
```

## Caching Strategies

### Multi-Level Caching

**L1 Cache - In-Memory Results**:
```python
# sentiment_analyzer_pro/optimization/cache.py
class InMemoryCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0
    
    def get_hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0
```

**L2 Cache - Redis (Optional)**:
```python
import redis

class RedisCache:
    def __init__(self, redis_url="redis://redis:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[SentimentResult]:
        cached_data = await self.redis_client.get(key)
        if cached_data:
            return SentimentResult.from_json(cached_data)
        return None
```

**Cache Warming**:
```python
class CacheWarmer:
    def __init__(self, analyzer, cache):
        self.analyzer = analyzer
        self.cache = cache
    
    def warm_cache_with_common_phrases(self):
        common_phrases = [
            "great product", "not satisfied", "excellent service",
            "poor quality", "love it", "hate it", "okay", "good"
        ]
        
        for phrase in common_phrases:
            result = self.analyzer.analyze(phrase)
            self.cache.put(phrase, result)
```

## Model Performance Optimization

### Model Selection and Configuration

**Performance vs Accuracy Trade-offs**:
```python
MODEL_CONFIGS = {
    "fast": {
        "model": "distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 64,
        "fp16": True,
        "expected_latency": "20-30ms",
        "accuracy": "~0.90"
    },
    "balanced": {
        "model": "bert-base-uncased", 
        "max_length": 512,
        "batch_size": 32,
        "fp16": True,
        "expected_latency": "40-60ms",
        "accuracy": "~0.94"
    },
    "accurate": {
        "model": "roberta-large",
        "max_length": 512,
        "batch_size": 16,
        "fp16": True,
        "expected_latency": "80-120ms",
        "accuracy": "~0.96"
    }
}
```

### Batch Processing Optimization

**Dynamic Batch Sizing**:
```python
class AdaptiveBatchProcessor:
    def __init__(self):
        self.base_batch_size = 32
        self.max_batch_size = 128
        self.min_batch_size = 8
        self.performance_history = []
    
    def get_optimal_batch_size(self, queue_size: int) -> int:
        # Adjust batch size based on queue length and performance
        if queue_size > 100:
            return min(self.max_batch_size, queue_size // 4)
        elif queue_size < 10:
            return self.min_batch_size
        else:
            return self.base_batch_size
    
    async def process_batch_optimally(self, texts: List[str]) -> List[SentimentResult]:
        optimal_size = self.get_optimal_batch_size(len(texts))
        results = []
        
        for i in range(0, len(texts), optimal_size):
            batch = texts[i:i + optimal_size]
            batch_results = await self.process_batch(batch)
            results.extend(batch_results)
        
        return results
```

## Network and I/O Optimization

### FastAPI Performance Tuning

**Async Request Handling**:
```python
from fastapi import FastAPI
import asyncio

app = FastAPI(
    title="Sentiment Analyzer Pro",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Connection pooling for better performance
@app.on_event("startup")
async def startup_event():
    # Initialize connection pools
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    
    # Pre-load models
    await app.state.analyzer.load_models()
```

**Response Optimization**:
```python
from fastapi.responses import ORJSONResponse

@app.post("/analyze", response_class=ORJSONResponse)
async def analyze_sentiment(request: TextInput):
    # Use orjson for faster JSON serialization
    result = await analyzer.analyze_async(request.text)
    return result.to_dict()
```

### Nginx Performance Configuration

**Nginx Optimization** (`nginx/nginx.conf`):
```nginx
# Worker processes optimization
worker_processes auto;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
    accept_mutex off;
}

http {
    # Connection optimization
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 30;
    keepalive_requests 1000;
    
    # Buffer optimization
    client_body_buffer_size 16K;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 8k;
    client_max_body_size 10M;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript;
    
    # Upstream optimization
    upstream sentiment_api {
        server sentiment-analyzer:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
        keepalive_requests 1000;
        keepalive_timeout 60s;
    }
}
```

## Monitoring and Profiling

### Performance Metrics Collection

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('sentiment_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('sentiment_request_duration_seconds', 'Request latency')

# Model performance metrics
MODEL_INFERENCE_TIME = Histogram('sentiment_model_inference_seconds', 'Model inference time')
CACHE_HIT_RATE = Gauge('sentiment_cache_hit_rate', 'Cache hit rate')

# Resource metrics
MEMORY_USAGE = Gauge('sentiment_memory_usage_bytes', 'Memory usage')
CPU_USAGE = Gauge('sentiment_cpu_usage_percent', 'CPU usage percentage')

@REQUEST_LATENCY.time()
async def analyze_with_metrics(text: str):
    start_time = time.time()
    
    # Check cache first
    cached_result = cache.get(text)
    if cached_result:
        CACHE_HIT_RATE.inc()
        return cached_result
    
    # Perform analysis
    with MODEL_INFERENCE_TIME.time():
        result = await model.analyze(text)
    
    # Update metrics
    REQUEST_COUNT.labels(method='POST', endpoint='/analyze').inc()
    
    return result
```

### Performance Profiling

**Python Profiling**:
```python
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def performance_profiler(filename="profile_output"):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(20)
        stats.dump_stats(f"{filename}.prof")

# Usage
async def analyze_with_profiling(text: str):
    with performance_profiler("sentiment_analysis"):
        return await analyzer.analyze(text)
```

**Memory Profiling**:
```python
from memory_profiler import profile
import tracemalloc

@profile
def analyze_memory_usage(text: str):
    tracemalloc.start()
    
    result = analyzer.analyze(text)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
    return result
```

## Load Testing

### Performance Testing Setup

**Locust Load Testing** (`tests/load_test.py`):
```python
from locust import HttpUser, task, between

class SentimentAnalysisUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.test_texts = [
            "I love this product!",
            "This is terrible quality",
            "It's okay, nothing special",
            "Amazing service and fast delivery",
            "Not worth the money"
        ]
    
    @task(3)
    def analyze_single_text(self):
        text = self.random.choice(self.test_texts)
        self.client.post("/api/analyze", json={"text": text})
    
    @task(1)
    def analyze_batch(self):
        texts = self.random.choices(self.test_texts, k=5)
        self.client.post("/api/analyze/batch", json={"texts": texts})
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
```

**Artillery.js Load Testing** (`tests/load_test.yml`):
```yaml
config:
  target: 'https://localhost'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 300
      arrivalRate: 50
      name: "Sustained load"
    - duration: 120
      arrivalRate: 100
      name: "Peak load"
  processor: "./load_test_functions.js"

scenarios:
  - name: "Sentiment Analysis Load Test"
    weight: 100
    flow:
      - post:
          url: "/api/analyze"
          json:
            text: "{{ generateRandomText() }}"
      - think: 1
```

### Performance Benchmarks

**Baseline Performance Tests**:
```bash
# Single instance performance
ab -n 1000 -c 10 -T application/json -p test_data.json https://localhost/api/analyze

# Sustained load testing
artillery run tests/load_test.yml

# Memory leak testing
python -m pytest tests/test_memory_leaks.py --duration=3600
```

## Scaling Strategies

### Horizontal Scaling

**Load Balancer Configuration**:
```yaml
# docker-compose.scale.yml
services:
  sentiment-analyzer:
    scale: 3
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure

  nginx:
    depends_on:
      - sentiment-analyzer
    volumes:
      - ./nginx/nginx-scaled.conf:/etc/nginx/nginx.conf
```

**Nginx Load Balancing** (`nginx/nginx-scaled.conf`):
```nginx
upstream sentiment_api {
    least_conn;
    server sentiment-analyzer-1:8000 max_fails=3 fail_timeout=30s;
    server sentiment-analyzer-2:8000 max_fails=3 fail_timeout=30s;
    server sentiment-analyzer-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 64;
}
```

### Auto-Scaling

**Docker Swarm Auto-scaling**:
```yaml
# docker-stack.yml
services:
  sentiment-analyzer:
    image: sentiment-analyzer-pro:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      update_config:
        parallelism: 1
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

**Kubernetes HPA** (Horizontal Pod Autoscaler):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analyzer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analyzer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting Performance Issues

### Common Performance Problems

**High Memory Usage**:
```bash
# Diagnose memory issues
docker stats sentiment-analyzer-api
htop -p $(pgrep -f sentiment)

# Check for memory leaks
python -m tracemalloc analyze_memory_usage.py
```

**Slow Response Times**:
```bash
# Check model loading time
time curl -X POST "https://localhost/api/analyze" -d '{"text":"test"}'

# Monitor request queues
docker-compose logs -f sentiment-analyzer | grep "queue"
```

**Cache Inefficiency**:
```python
# Monitor cache performance
def check_cache_performance():
    hit_rate = cache.get_hit_rate()
    print(f"Cache hit rate: {hit_rate:.2%}")
    
    if hit_rate < 0.5:  # Less than 50% hit rate
        print("Consider:")
        print("- Increasing cache size")
        print("- Adjusting TTL settings") 
        print("- Implementing cache warming")
```

### Performance Tuning Checklist

**Application Level**:
- [ ] Model loading optimization
- [ ] Batch size tuning
- [ ] Cache configuration
- [ ] Memory management
- [ ] Async processing enablement

**Infrastructure Level**:
- [ ] Resource allocation (CPU/Memory)
- [ ] Network optimization
- [ ] Storage I/O optimization
- [ ] Load balancing configuration
- [ ] Auto-scaling setup

**Monitoring**:
- [ ] Performance metrics collection
- [ ] Alerting configuration
- [ ] Load testing setup
- [ ] Profiling implementation
- [ ] Capacity planning

## Best Practices Summary

1. **Resource Management**:
   - Monitor memory usage continuously
   - Implement proper garbage collection
   - Use appropriate batch sizes

2. **Caching Strategy**:
   - Implement multi-level caching
   - Monitor cache hit rates
   - Use cache warming for common requests

3. **Model Optimization**:
   - Choose appropriate model size for requirements
   - Use FP16 precision when possible
   - Implement model quantization for production

4. **Infrastructure**:
   - Configure proper resource limits
   - Implement load balancing
   - Use auto-scaling when appropriate

5. **Monitoring**:
   - Collect comprehensive performance metrics
   - Set up alerting for performance degradation
   - Regular performance testing and profiling