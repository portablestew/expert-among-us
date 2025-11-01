"""Encoding-aware symbol utilities for cross-platform console output."""

import sys

# Detect if we need ASCII fallbacks for Windows CP1252 encoding
USE_ASCII_FALLBACKS = sys.stdout.encoding and sys.stdout.encoding.lower() in ('cp1252', 'cp850', 'ascii')

# Symbol mappings: Unicode -> ASCII fallback
SYMBOLS = {
    'info': 'i' if USE_ASCII_FALLBACKS else 'ℹ',
    'warning': '!' if USE_ASCII_FALLBACKS else '⚠',
    'error': 'X' if USE_ASCII_FALLBACKS else '✗',
    'success': 'v' if USE_ASCII_FALLBACKS else '✓',
}