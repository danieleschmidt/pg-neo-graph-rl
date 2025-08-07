"""FastAPI application for sentiment analysis."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from ..core.analyzer import SentimentAnalyzer
from ..utils.exceptions import SentimentAnalysisError, ValidationError
from .models import (
    SentimentRequest,
    SentimentResponse, 
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    ErrorResponse
)


logger = logging.getLogger(__name__)

# Global analyzer instance
analyzer: SentimentAnalyzer = None
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global analyzer
    
    # Startup
    logger.info("Starting Sentiment Analyzer API...")
    
    try:
        analyzer = SentimentAnalyzer(
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="auto",
            batch_size=32
        )
        logger.info("Sentiment analyzer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load sentiment analyzer: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analyzer API...")
    if analyzer:
        analyzer.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Sentiment Analyzer Pro API",
        description="Enterprise-grade sentiment analysis API with advanced ML models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Exception handlers
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request, exc):
        return HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="ValidationError",
                message=str(exc),
                details={"request_path": str(request.url)}
            ).model_dump()
        )
    
    @app.exception_handler(SentimentAnalysisError) 
    async def analysis_error_handler(request, exc):
        return HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="SentimentAnalysisError",
                message=str(exc),
                details={"request_path": str(request.url)}
            ).model_dump()
        )
    
    # Dependency to get analyzer
    async def get_analyzer() -> SentimentAnalyzer:
        if analyzer is None:
            raise HTTPException(status_code=503, detail="Analyzer not initialized")
        return analyzer
    
    # Routes
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint."""
        return {
            "name": "Sentiment Analyzer Pro API",
            "version": "1.0.0", 
            "status": "running",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Health check endpoint."""
        try:
            model_info = analyzer.get_model_info()
            
            return HealthResponse(
                status="healthy",
                model_loaded=model_info["model_loaded"],
                gpu_available=model_info["gpu_available"], 
                memory_usage={
                    "gpu_count": model_info["gpu_count"],
                    "device": model_info["device"]
                },
                uptime=time.time() - startup_time,
                version="1.0.0"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Health check failed: {str(e)}"
            )
    
    @app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
    async def analyze_sentiment(
        request: SentimentRequest,
        analyzer: SentimentAnalyzer = Depends(get_analyzer)
    ):
        """Analyze sentiment of a single text."""
        try:
            start_time = time.time()
            result = await analyzer.analyze_async(request.text)
            
            return SentimentResponse(
                text=result.text,
                label=result.label,
                confidence=result.confidence,
                probabilities=result.probabilities.tolist() if hasattr(result.probabilities, 'tolist') else result.probabilities,
                processing_time=time.time() - start_time,
                metadata=result.metadata
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
    
    @app.post("/analyze/batch", response_model=BatchSentimentResponse, tags=["Analysis"])
    async def analyze_batch_sentiment(
        request: BatchSentimentRequest,
        analyzer: SentimentAnalyzer = Depends(get_analyzer)
    ):
        """Analyze sentiment of multiple texts."""
        try:
            start_time = time.time()
            results = await analyzer.analyze_batch_async(request.texts)
            
            sentiment_responses = [
                SentimentResponse(
                    text=result.text,
                    label=result.label,
                    confidence=result.confidence,
                    probabilities=result.probabilities.tolist() if hasattr(result.probabilities, 'tolist') else result.probabilities,
                    processing_time=result.processing_time,
                    metadata=result.metadata
                ) for result in results
            ]
            
            return BatchSentimentResponse(
                results=sentiment_responses,
                total_processing_time=time.time() - start_time,
                batch_size=len(request.texts)
            )
            
        except Exception as e:
            logger.error(f"Batch analysis error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch analysis failed: {str(e)}"
            )
    
    @app.get("/models/info", tags=["Models"])
    async def get_model_info(analyzer: SentimentAnalyzer = Depends(get_analyzer)):
        """Get information about the loaded model."""
        try:
            return analyzer.get_model_info()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get model info: {str(e)}"
            )
    
    @app.post("/models/reload", tags=["Models"])
    async def reload_model(
        background_tasks: BackgroundTasks,
        analyzer: SentimentAnalyzer = Depends(get_analyzer)
    ):
        """Reload the sentiment analysis model."""
        def reload_analyzer():
            global analyzer
            try:
                analyzer.close()
                analyzer = SentimentAnalyzer(
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device="auto"
                )
                logger.info("Model reloaded successfully")
            except Exception as e:
                logger.error(f"Model reload failed: {e}")
        
        background_tasks.add_task(reload_analyzer)
        return {"message": "Model reload initiated"}
    
    return app


# Create app instance
app = create_app()


def start_server():
    """Start the API server."""
    uvicorn.run(
        "sentiment_analyzer_pro.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()