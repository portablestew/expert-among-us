"""Transparent compression/decompression for diff content."""

import zlib


class CompressionError(Exception):
    """Raised when compression/decompression fails."""
    pass


def compress_diff(diff: str) -> bytes:
    """Compress diff content using zlib.
    
    Args:
        diff: Uncompressed diff string
        
    Returns:
        Compressed bytes (includes zlib header)
        
    Raises:
        CompressionError: If compression fails
    """
    try:
        # Encode to UTF-8 first
        diff_bytes = diff.encode('utf-8')
        
        # Compress with level 6 (good balance of speed/size)
        compressed = zlib.compress(diff_bytes, level=6)
        
        return compressed
    except Exception as e:
        raise CompressionError(f"Failed to compress diff: {e}")


def decompress_diff(compressed: bytes) -> str:
    """Decompress diff content.
    
    Args:
        compressed: Compressed bytes (with zlib header)
        
    Returns:
        Decompressed diff string
        
    Raises:
        CompressionError: If decompression fails
    """
    try:
        # Decompress
        decompressed_bytes = zlib.decompress(compressed)
        
        # Decode from UTF-8
        diff = decompressed_bytes.decode('utf-8')
        
        return diff
    except zlib.error as e:
        raise CompressionError(f"Failed to decompress diff (corrupt data?): {e}")
    except UnicodeDecodeError as e:
        raise CompressionError(f"Failed to decode decompressed diff: {e}")