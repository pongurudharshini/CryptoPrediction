"""
Text Preprocessor for Crypto Sentiment Analysis.
Cleans raw social media posts and news text for NLP models.
"""

import re
import string

class TextCleaner:
    """Preprocesses raw text for FinBERT, VADER, and RoBERTa."""
    
    # Common crypto slang normalization mapping
    CRYPTO_SLANG = {
        r'\bhodl\b': 'hold',
        r'\bbullish\b': 'optimistic positive price growth',
        r'\bbearish\b': 'pessimistic negative price drop',
        r'\bto the moon\b': 'extreme price increase',
        r'\brekt\b': 'severe monetary loss',
        r'\bdip\b': 'temporary price drop',
        r'\bfud\b': 'fear uncertainty doubt'
    }

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Cleans input string:
        - Removes URLs, HTML tags, and user mentions (@user)
        - Normalizes crypto jargon
        - Strips excessive whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Normalize crypto slang
        for pattern, replacement in cls.CRYPTO_SLANG.items():
            text = re.sub(pattern, replacement, text)

        # Remove extra white spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text