"""Text preprocessing for sentiment analysis."""

import re
import logging
from typing import Dict, List, Optional, Union
import html


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Advanced text preprocessor for sentiment analysis.
    
    Features:
    - URL removal/replacement
    - Emoji handling (preservation or removal)
    - Text normalization
    - HTML entity decoding
    - Special character handling
    - Case normalization
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        handle_emojis: bool = True,
        normalize_text: bool = True,
        remove_special_chars: bool = False,
        preserve_sentiment_chars: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            handle_emojis: Convert emojis to text descriptions
            normalize_text: Normalize whitespace and special characters
            remove_special_chars: Remove special characters
            preserve_sentiment_chars: Keep sentiment-relevant punctuation
            max_length: Maximum text length (truncate if longer)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.handle_emojis = handle_emojis
        self.normalize_text = normalize_text
        self.remove_special_chars = remove_special_chars
        self.preserve_sentiment_chars = preserve_sentiment_chars
        self.max_length = max_length
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Emoji mappings for sentiment analysis
        self.emoji_sentiment_map = {
            '😀': ' happy ', '😃': ' happy ', '😄': ' happy ', '😁': ' happy ',
            '😊': ' happy ', '🙂': ' happy ', '😉': ' happy ', '😍': ' love ',
            '🥰': ' love ', '😘': ' love ', '😗': ' happy ', '☺️': ' happy ',
            '😚': ' happy ', '😙': ' happy ', '😋': ' happy ', '😛': ' playful ',
            '😜': ' playful ', '🤪': ' crazy ', '😝': ' playful ', '🤗': ' happy ',
            '🤩': ' excited ', '🥳': ' celebration ', '😎': ' cool ',
            '😔': ' sad ', '😞': ' sad ', '😟': ' worried ', '😕': ' confused ',
            '🙁': ' sad ', '☹️': ' sad ', '😣': ' frustrated ', '😖': ' frustrated ',
            '😫': ' frustrated ', '😩': ' tired ', '🥺': ' sad ', '😢': ' crying ',
            '😭': ' crying ', '😤': ' angry ', '😠': ' angry ', '😡': ' angry ',
            '🤬': ' angry ', '🤯': ' shocked ', '😱': ' shocked ', '😨': ' scared ',
            '😰': ' worried ', '😥': ' sad ', '😓': ' worried ', '🤔': ' thinking ',
            '🤨': ' suspicious ', '😐': ' neutral ', '😑': ' neutral ',
            '🙄': ' annoyed ', '😬': ' awkward ', '🤐': ' quiet ', '😷': ' sick ',
            '🤢': ' sick ', '🤮': ' sick ', '🤧': ' sick ', '🥴': ' dizzy ',
            '😵': ' dizzy ', '🤠': ' happy ', '🥸': ' sneaky ', '😇': ' innocent ',
            '🤓': ' nerdy ', '🧐': ' thinking ', '👍': ' good ', '👎': ' bad ',
            '👌': ' okay ', '✌️': ' peace ', '🤞': ' hopeful ', '🤟': ' love ',
            '🤘': ' rock ', '🤙': ' cool ', '👏': ' applause ', '🙌': ' celebration ',
            '👐': ' open ', '🤲': ' prayer ', '🙏': ' prayer ', '✍️': ' writing ',
            '💪': ' strong ', '🦾': ' strong ', '🦿': ' strong ', '🦵': ' strong ',
            '🦶': ' foot ', '👂': ' listen ', '🦻': ' listen ', '👃': ' smell ',
            '🫀': ' heart ', '🫁': ' lungs ', '🧠': ' brain ', '🦷': ' tooth ',
            '🦴': ' bone ', '👀': ' eyes ', '👁️': ' eye ', '👅': ' tongue ',
            '👄': ' lips ', '💋': ' kiss ', '🩸': ' blood ', '❤️': ' love ',
            '🧡': ' love ', '💛': ' love ', '💚': ' love ', '💙': ' love ',
            '💜': ' love ', '🖤': ' dark ', '🤍': ' pure ', '🤎': ' brown ',
            '💔': ' heartbreak ', '❣️': ' love ', '💕': ' love ', '💞': ' love ',
            '💓': ' love ', '💗': ' love ', '💖': ' love ', '💘': ' love ',
            '💝': ' love ', '💟': ' love ', '☮️': ' peace ', '✝️': ' faith ',
            '☪️': ' faith ', '🕉️': ' faith ', '☸️': ' faith ', '✡️': ' faith ',
            '🔯': ' faith ', '🕎': ' faith ', '☯️': ' balance ', '☦️': ' faith ',
            '🛐': ' faith ', '⛎': ' zodiac ', '♈': ' zodiac ', '♉': ' zodiac ',
            '♊': ' zodiac ', '♋': ' zodiac ', '♌': ' zodiac ', '♍': ' zodiac ',
            '♎': ' zodiac ', '♏': ' zodiac ', '♐': ' zodiac ', '♑': ' zodiac ',
            '♒': ' zodiac ', '♓': ' zodiac ', '🆔': ' id ', '⚛️': ' science ',
            '🉑': ' accept ', '☢️': ' danger ', '☣️': ' danger ', '📴': ' off ',
            '📳': ' vibrate ', '🈶': ' charge ', '🈚': ' free ', '🈸': ' apply ',
            '🈺': ' open ', '🈷️': ' month ', '✴️': ' star ', '🆚': ' versus ',
            '💮': ' flower ', '🉐': ' bargain ', '㊙️': ' secret ', '㊗️': ' congratulations ',
            '🈴': ' together ', '🈵': ' full ', '🈹': ' discount ', '🈲': ' prohibit ',
            '🅰️': ' a ', '🅱️': ' b ', '🆎': ' ab ', '🆑': ' clear ', '🅾️': ' o ',
            '🆘': ' help ', '❌': ' no ', '⭕': ' heavy ', '🛑': ' stop ',
            '⛔': ' no ', '📛': ' name ', '🚫': ' prohibited ', '💯': ' hundred ',
            '💢': ' anger ', '♨️': ' hot ', '🚷': ' no ', '🚯': ' no ',
            '🚳': ' no ', '🚱': ' no ', '🔞': ' adult ', '📵': ' no ',
            '🚭': ' no ', '❗': ' exclamation ', '❕': ' exclamation ', '❓': ' question ',
            '❔': ' question ', '‼️': ' exclamation ', '⁉️': ' question ',
            '🔅': ' dim ', '🔆': ' bright ', '〽️': ' part ', '⚠️': ' warning ',
            '🚸': ' children ', '🔱': ' trident ', '⚜️': ' fleur ', '🔰': ' beginner ',
            '♻️': ' recycle ', '✅': ' check ', '🈯': ' finger ', '💹': ' chart ',
            '❇️': ' sparkle ', '✳️': ' eight ', '❎': ' cross ', '🌐': ' globe ',
            '💠': ' diamond ', 'Ⓜ️': ' m ', '🌀': ' cyclone ', '💤': ' sleep ',
            '🏧': ' atm ', '🚾': ' wc ', '♿': ' wheelchair ', '🅿️': ' parking ',
            '🈳': ' vacancy ', '🈂️': ' sa ', '🛂': ' passport ', '🛃': ' customs ',
            '🛄': ' baggage ', '🛅': ' left '
        }
        
        logger.info("TextPreprocessor initialized")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient text processing."""
        # URL patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special characters (keeping sentiment-relevant ones if specified)
        if self.preserve_sentiment_chars:
            # Keep !?.,;:"'-() and other punctuation that affects sentiment
            self.special_chars_pattern = re.compile(r'[^\w\s!?.,;:"\'()\-/]')
        else:
            self.special_chars_pattern = re.compile(r'[^\w\s]')
            
        # Repeated characters pattern (e.g., "soooo good" -> "so good")
        self.repeated_chars_pattern = re.compile(r'(.)\1{2,}')
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+')
    
    def process(self, text: str) -> str:
        """
        Process text with all configured preprocessing steps.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return "neutral"
        
        # HTML entity decoding
        text = html.unescape(text)
        
        # Handle emojis
        if self.handle_emojis:
            text = self._process_emojis(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' URL ', text)
            text = self.email_pattern.sub(' EMAIL ', text)
        
        # Remove mentions and hashtags
        if self.remove_mentions:
            text = self.mention_pattern.sub(' ', text)
            
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub(' ', text)
        
        # Case normalization
        if self.lowercase:
            text = text.lower()
        
        # Normalize repeated characters (keep some emphasis)
        text = self.repeated_chars_pattern.sub(r'\1\1', text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self.special_chars_pattern.sub(' ', text)
        
        # Text normalization
        if self.normalize_text:
            # Replace multiple whitespace with single space
            text = self.whitespace_pattern.sub(' ', text)
            
            # Fix common contractions for sentiment analysis
            text = self._normalize_contractions(text)
        
        # Trim and final cleanup
        text = text.strip()
        
        # Handle max length
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        
        # Ensure we don't return empty text
        if not text:
            text = "neutral"
            
        return text
    
    def _process_emojis(self, text: str) -> str:
        """Convert emojis to sentiment-aware text descriptions."""
        for emoji, description in self.emoji_sentiment_map.items():
            text = text.replace(emoji, description)
        
        # Remove any remaining emojis that we don't have mappings for
        # This is a simple approach - could be enhanced with emoji library
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", 
            flags=re.UNICODE
        )
        
        text = emoji_pattern.sub(' ', text)
        return text
    
    def _normalize_contractions(self, text: str) -> str:
        """Expand contractions for better sentiment analysis."""
        contractions = {
            "n't": " not",
            "'re": " are", 
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "won't": "will not",
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "mustn't": "must not",
            "needn't": "need not",
            "daren't": "dare not",
            "mayn't": "may not",
            "oughtn't": "ought not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "doesn't": "does not",
            "don't": "do not",
            "didn't": "did not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
    
    def batch_process(self, texts: List[str]) -> List[str]:
        """Process multiple texts efficiently."""
        return [self.process(text) for text in texts]
    
    def get_stats(self, text: str) -> Dict[str, Union[int, float]]:
        """Get preprocessing statistics for a text."""
        original_length = len(text)
        processed_text = self.process(text)
        processed_length = len(processed_text)
        
        # Count different elements
        urls = len(self.url_pattern.findall(text))
        mentions = len(self.mention_pattern.findall(text))
        hashtags = len(self.hashtag_pattern.findall(text))
        
        return {
            "original_length": original_length,
            "processed_length": processed_length,
            "reduction_ratio": (original_length - processed_length) / original_length if original_length > 0 else 0,
            "urls_found": urls,
            "mentions_found": mentions,
            "hashtags_found": hashtags,
            "word_count": len(processed_text.split()),
            "char_count": processed_length
        }