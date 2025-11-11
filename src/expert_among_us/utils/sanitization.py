"""Text sanitization for removing high-entropy patterns before embedding."""

import re
from typing import Optional


class TextSanitizer:
    """Remove high-entropy patterns from text before embedding.
    
    Replaces UUIDs, hex strings, API keys, and other high-entropy
    patterns to reduce noise in vector embeddings while preserving
    semantic content for search results and prompts.
    
    Two patterns:
    1. Hex/Numeric: 7+ hex digits with separators (UUIDs, lookup tables, coordinates)
    2. Base64: 32+ base64 characters (encryption keys, tokens)
    
    Sanitization occurs BEFORE embedding but AFTER SQLite storage,
    so search results show original content while embeddings are clean.
    """
    
    # Pattern 1: Hex/Numeric content (7+ hex digits with separators)
    HEX_NUMERIC_PATTERN = re.compile(
        r'[\({]?'                                    # Optional opening brace/paren
        r'(?:0[xXoObB]|&[hH])?'                     # Prefixes: 0x, 0o, 0b, &h
        r'(?:[0-9a-fA-F][\-,.:/ \t]*){7,}'          # 7+ hex digits with separators
        r'(?:[hHbBuUlLfFdD]+)?'                     # Suffixes: h, b, u, l, f, d
        r'[\)}]?',                                   # Optional closing brace/paren
        re.IGNORECASE
    )
    
    # Pattern 2: Base64/encryption content (32+ base64 chars)
    BASE64_PATTERN = re.compile(
        r'[A-Za-z0-9+/_-]{32,}={0,2}'
    )
    
    def __init__(
        self,
        enable_hex_numeric: bool = True,
        enable_base64: bool = True,
        removal_mode: str = "remove"  # "remove" or "replace"
    ):
        """Initialize sanitizer.
        
        Args:
            enable_hex_numeric: Toggle hex/numeric pattern sanitization
            enable_base64: Toggle base64 pattern sanitization  
            removal_mode: 
                - "remove": Delete patterns entirely
                - "replace": Replace with normalized tokens
        """
        self.enable_hex_numeric = enable_hex_numeric
        self.enable_base64 = enable_base64
        self.removal_mode = removal_mode
    
    def sanitize(self, text: str) -> str:
        """Sanitize high-entropy patterns in text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text with patterns removed or replaced
        """
        result = text
        
        if self.removal_mode == "remove":
            # Remove patterns entirely
            if self.enable_hex_numeric:
                result = self.HEX_NUMERIC_PATTERN.sub('', result)
            if self.enable_base64:
                result = self.BASE64_PATTERN.sub('', result)
        else:
            # Replace with normalized tokens
            if self.enable_hex_numeric:
                result = self.HEX_NUMERIC_PATTERN.sub('HEX_TOKEN', result)
            if self.enable_base64:
                result = self.BASE64_PATTERN.sub('BASE64_TOKEN', result)
        
        return result
    
    def get_patterns(self) -> dict[str, re.Pattern]:
        """Get the compiled regex patterns for inspection/testing.
        
        Returns:
            Dictionary mapping pattern names to compiled regex objects
        """
        return {
            'hex_numeric': self.HEX_NUMERIC_PATTERN,
            'base64': self.BASE64_PATTERN
        }