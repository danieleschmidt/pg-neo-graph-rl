"""Advanced validation utilities for sentiment analysis."""

import re
import json
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
import logging

from .exceptions import ValidationError, ConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A validation rule with predicate and error message."""
    name: str
    predicate: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    validated_value: Any = None


class ConfigValidator:
    """
    Advanced configuration validator for sentiment analysis settings.
    
    Features:
    - Type validation
    - Range validation
    - Dependency validation
    - Custom rule validation
    - Configuration sanitization
    """
    
    def __init__(self):
        """Initialize configuration validator."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default validation rules for common configurations."""
        
        # Model configuration rules
        self.add_rule("model_name", ValidationRule(
            name="model_name_not_empty",
            predicate=lambda x: isinstance(x, str) and len(x.strip()) > 0,
            error_message="Model name cannot be empty"
        ))
        
        # Device configuration rules
        self.add_rule("device", ValidationRule(
            name="device_valid",
            predicate=lambda x: x in ["auto", "cpu", "cuda", "mps"] or 
                              (isinstance(x, str) and x.startswith("cuda:")),
            error_message="Device must be 'auto', 'cpu', 'cuda', 'mps', or 'cuda:N'"
        ))
        
        # Batch size rules
        self.add_rule("batch_size", ValidationRule(
            name="batch_size_positive",
            predicate=lambda x: isinstance(x, int) and x > 0,
            error_message="Batch size must be a positive integer"
        ))
        
        self.add_rule("batch_size", ValidationRule(
            name="batch_size_reasonable",
            predicate=lambda x: isinstance(x, int) and x <= 128,
            error_message="Batch size > 128 may cause memory issues",
            severity="warning"
        ))
        
        # Max length rules
        self.add_rule("max_length", ValidationRule(
            name="max_length_positive",
            predicate=lambda x: isinstance(x, int) and x > 0,
            error_message="Max length must be a positive integer"
        ))
        
        self.add_rule("max_length", ValidationRule(
            name="max_length_reasonable",
            predicate=lambda x: isinstance(x, int) and x <= 10000,
            error_message="Max length > 10000 may cause performance issues",
            severity="warning"
        ))
        
        # Number of classes rules
        self.add_rule("num_classes", ValidationRule(
            name="num_classes_valid",
            predicate=lambda x: isinstance(x, int) and x >= 2,
            error_message="Number of classes must be at least 2"
        ))
        
        # Confidence threshold rules
        self.add_rule("confidence_threshold", ValidationRule(
            name="confidence_threshold_range",
            predicate=lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            error_message="Confidence threshold must be between 0.0 and 1.0"
        ))
        
    def add_rule(self, field: str, rule: ValidationRule):
        """Add a validation rule for a field."""
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)
        
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        info = []
        
        # Validate each field
        for field, value in config.items():
            if field in self.rules:
                field_result = self._validate_field(field, value)
                errors.extend(field_result.errors)
                warnings.extend(field_result.warnings)
                info.extend(field_result.info)
        
        # Cross-field validation
        cross_validation_result = self._validate_cross_field_dependencies(config)
        errors.extend(cross_validation_result.errors)
        warnings.extend(cross_validation_result.warnings)
        info.extend(cross_validation_result.info)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            validated_value=config
        )
    
    def _validate_field(self, field: str, value: Any) -> ValidationResult:
        """Validate a single field against its rules."""
        errors = []
        warnings = []
        info = []
        
        for rule in self.rules[field]:
            try:
                if not rule.predicate(value):
                    if rule.severity == "error":
                        errors.append(f"{field}: {rule.error_message}")
                    elif rule.severity == "warning":
                        warnings.append(f"{field}: {rule.error_message}")
                    else:
                        info.append(f"{field}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{field}: Validation rule '{rule.name}' failed: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def _validate_cross_field_dependencies(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate dependencies between fields."""
        errors = []
        warnings = []
        info = []
        
        # Check device and GPU-related settings
        if config.get("device") == "cuda" and config.get("batch_size", 0) > 64:
            warnings.append("Large batch size with CUDA may cause out-of-memory errors")
        
        # Check model size vs memory constraints
        if ("bert-large" in config.get("model_name", "").lower() and 
            config.get("batch_size", 0) > 16):
            warnings.append("Large model with big batch size may require significant memory")
        
        # Check preprocessing vs performance
        if (config.get("enable_preprocessing") is False and 
            config.get("batch_size", 0) > 32):
            info.append("Disabling preprocessing with large batches may improve performance")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )


class TextValidator:
    """
    Advanced text validation for sentiment analysis input.
    
    Features:
    - Content validation
    - Encoding validation
    - Language detection
    - Quality scoring
    - Metadata extraction
    """
    
    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        allowed_languages: Optional[List[str]] = None,
        quality_threshold: float = 0.5
    ):
        """
        Initialize text validator.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            allowed_languages: List of allowed language codes
            quality_threshold: Minimum quality score (0-1)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_languages = allowed_languages
        self.quality_threshold = quality_threshold
        
        # Compile patterns for text analysis
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for text analysis."""
        # Non-printable characters
        self.non_printable_pattern = re.compile(r'[^\x20-\x7E\n\r\t]')
        
        # Repeated characters/words
        self.repeated_chars_pattern = re.compile(r'(.)\1{5,}')
        self.repeated_words_pattern = re.compile(r'\b(\w+)(\s+\1){3,}\b', re.IGNORECASE)
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Word pattern
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Sentence pattern
        self.sentence_pattern = re.compile(r'[.!?]+')
        
    def validate_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate input text for sentiment analysis.
        
        Args:
            text: Text to validate
            metadata: Optional metadata about the text
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        info = []
        
        # Basic type check
        if not isinstance(text, str):
            errors.append(f"Text must be a string, got {type(text)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, info=info)
        
        # Length validation
        if len(text) < self.min_length:
            errors.append(f"Text too short: {len(text)} < {self.min_length}")
        
        if len(text) > self.max_length:
            errors.append(f"Text too long: {len(text)} > {self.max_length}")
        
        # Content validation
        content_result = self._validate_content(text)
        errors.extend(content_result.errors)
        warnings.extend(content_result.warnings)
        info.extend(content_result.info)
        
        # Quality validation
        quality_result = self._validate_quality(text)
        warnings.extend(quality_result.warnings)
        info.extend(quality_result.info)
        
        # Language validation
        if self.allowed_languages:
            lang_result = self._validate_language(text)
            if lang_result.errors:
                errors.extend(lang_result.errors)
            warnings.extend(lang_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            validated_value=text
        )
    
    def _validate_content(self, text: str) -> ValidationResult:
        """Validate text content."""
        errors = []
        warnings = []
        info = []
        
        # Check for empty/whitespace-only text
        if not text.strip():
            errors.append("Text contains only whitespace")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, info=info)
        
        # Check encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            errors.append(f"Text encoding error: {e}")
        
        # Check for excessive non-printable characters
        non_printable_count = len(self.non_printable_pattern.findall(text))
        if non_printable_count > len(text) * 0.1:  # More than 10% non-printable
            warnings.append(f"High number of non-printable characters: {non_printable_count}")
        
        # Check for repeated patterns
        if self.repeated_chars_pattern.search(text):
            warnings.append("Text contains excessively repeated characters")
        
        if self.repeated_words_pattern.search(text):
            warnings.append("Text contains excessively repeated words")
        
        # Check content composition
        url_count = len(self.url_pattern.findall(text))
        if url_count > 5:
            warnings.append(f"High number of URLs: {url_count}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def _validate_quality(self, text: str) -> ValidationResult:
        """Assess text quality for sentiment analysis."""
        warnings = []
        info = []
        
        # Word count
        words = self.word_pattern.findall(text)
        word_count = len(words)
        
        if word_count < 3:
            warnings.append(f"Very short text: {word_count} words")
        elif word_count > 500:
            info.append(f"Long text: {word_count} words")
        
        # Sentence structure
        sentences = self.sentence_pattern.split(text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count == 0:
            warnings.append("No sentence structure detected")
        
        # Average word length
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2:
                warnings.append("Very short average word length")
            elif avg_word_length > 10:
                info.append("High average word length - may be technical text")
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            warnings.append("Low character diversity")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text, word_count, sentence_count)
        
        if quality_score < self.quality_threshold:
            warnings.append(f"Low quality score: {quality_score:.2f}")
        else:
            info.append(f"Quality score: {quality_score:.2f}")
        
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            info=info,
            validated_value=quality_score
        )
    
    def _calculate_quality_score(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calculate a quality score for the text."""
        score = 1.0
        
        # Penalize very short texts
        if word_count < 3:
            score *= 0.3
        elif word_count < 5:
            score *= 0.7
        
        # Penalize lack of sentence structure
        if sentence_count == 0:
            score *= 0.5
        
        # Penalize excessive repetition
        if self.repeated_chars_pattern.search(text):
            score *= 0.6
        
        if self.repeated_words_pattern.search(text):
            score *= 0.7
        
        # Penalize excessive non-printable characters
        non_printable_ratio = len(self.non_printable_pattern.findall(text)) / len(text)
        if non_printable_ratio > 0.1:
            score *= (1.0 - non_printable_ratio)
        
        # Boost for good length
        if 10 <= word_count <= 100:
            score *= 1.1
        
        return max(0.0, min(1.0, score))
    
    def _validate_language(self, text: str) -> ValidationResult:
        """Validate text language (basic heuristic-based detection)."""
        errors = []
        warnings = []
        
        # Simple language detection based on character sets
        # This is a basic implementation - consider using langdetect library for production
        
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cyrillic = bool(re.search(r'[а-яА-Я]', text))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        has_japanese = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))
        
        detected_scripts = []
        if has_latin:
            detected_scripts.append("latin")
        if has_cyrillic:
            detected_scripts.append("cyrillic")
        if has_chinese:
            detected_scripts.append("chinese")
        if has_japanese:
            detected_scripts.append("japanese")
        if has_arabic:
            detected_scripts.append("arabic")
        
        if not detected_scripts:
            warnings.append("No recognizable script detected")
        elif len(detected_scripts) > 1:
            warnings.append(f"Multiple scripts detected: {', '.join(detected_scripts)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=[f"Detected scripts: {', '.join(detected_scripts)}"] if detected_scripts else []
        )


class ModelValidator:
    """Validator for model-related configurations and states."""
    
    def __init__(self):
        """Initialize model validator."""
        self.supported_models = {
            "bert-base-uncased",
            "bert-large-uncased", 
            "distilbert-base-uncased",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "roberta-base",
            "roberta-large"
        }
    
    def validate_model_config(self, model_name: str, num_classes: int) -> ValidationResult:
        """Validate model configuration."""
        errors = []
        warnings = []
        info = []
        
        # Check if model is supported
        if model_name not in self.supported_models:
            warnings.append(f"Model {model_name} is not in tested models list")
        
        # Check class count
        if num_classes < 2:
            errors.append("Number of classes must be at least 2")
        elif num_classes > 10:
            warnings.append("High number of classes may reduce accuracy")
        
        # Model-specific checks
        if "large" in model_name.lower() and num_classes > 5:
            warnings.append("Large model with many classes requires significant memory")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )