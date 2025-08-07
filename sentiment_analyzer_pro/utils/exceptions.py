"""Custom exceptions for sentiment analysis."""


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""
    pass


class ModelLoadError(SentimentAnalysisError):
    """Exception raised when model loading fails."""
    pass


class PredictionError(SentimentAnalysisError):
    """Exception raised when prediction fails."""
    pass


class ValidationError(SentimentAnalysisError):
    """Exception raised when input validation fails."""
    pass


class ConfigurationError(SentimentAnalysisError):
    """Exception raised when configuration is invalid."""
    pass


class ResourceError(SentimentAnalysisError):
    """Exception raised when system resources are insufficient."""
    pass


class NetworkError(SentimentAnalysisError):
    """Exception raised for network-related issues."""
    pass


class SecurityError(SentimentAnalysisError):
    """Exception raised for security-related issues."""
    pass