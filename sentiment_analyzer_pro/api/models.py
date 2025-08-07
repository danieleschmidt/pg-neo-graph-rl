"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SentimentRequest(BaseModel):
    """Request model for single text sentiment analysis."""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "I love this product! It's amazing."
            }
        }
    )


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "I love this product!",
                    "This is terrible.",
                    "It's okay, nothing special."
                ]
            }
        }
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    
    text: str = Field(..., description="Original input text")
    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probabilities: List[float] = Field(..., description="Probabilities for each class")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "I love this product!",
                "label": "POSITIVE", 
                "confidence": 0.95,
                "probabilities": [0.02, 0.03, 0.95],
                "processing_time": 0.045,
                "metadata": {"model_name": "distilbert-base-uncased-finetuned-sst-2-english"}
            }
        }
    )


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    
    results: List[SentimentResponse] = Field(..., description="List of sentiment analysis results")
    total_processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    batch_size: int = Field(..., ge=0, description="Number of texts processed")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "text": "I love this product!",
                        "label": "POSITIVE",
                        "confidence": 0.95,
                        "probabilities": [0.02, 0.03, 0.95],
                        "processing_time": 0.045,
                        "metadata": {}
                    }
                ],
                "total_processing_time": 0.123,
                "batch_size": 1
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage information")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": True,
                "memory_usage": {"used_gb": 2.1, "total_gb": 8.0},
                "uptime": 3600.0,
                "version": "1.0.0"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Input text cannot be empty",
                "details": {"field": "text", "input": ""}
            }
        }
    )