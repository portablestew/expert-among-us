"""Unit tests for diff compression utilities."""

import pytest

from expert_among_us.utils.compression import (
    CompressionError,
    compress_diff,
    decompress_diff,
)


def test_compress_decompress_roundtrip():
    """Test that compression and decompression are inverses."""
    original = "diff --git a/file.py b/file.py\n" * 1000
    compressed = compress_diff(original)
    decompressed = decompress_diff(compressed)
    assert decompressed == original


def test_compression_ratio():
    """Test that compression achieves good ratio on typical diffs."""
    diff = "- old line\n+ new line\n" * 1000
    compressed = compress_diff(diff)
    ratio = len(compressed) / len(diff.encode('utf-8'))
    assert ratio < 0.3  # Should compress to <30%


def test_decompress_corrupt_data():
    """Test error handling for corrupt compressed data."""
    with pytest.raises(CompressionError):
        decompress_diff(b"not valid zlib data")


def test_large_diff():
    """Test compression of large diffs (near 100KB limit)."""
    large_diff = "line " * 20000  # ~100KB
    compressed = compress_diff(large_diff)
    decompressed = decompress_diff(compressed)
    assert decompressed == large_diff
    assert len(compressed) < len(large_diff.encode('utf-8'))


def test_empty_diff():
    """Test compression of empty diff."""
    empty_diff = ""
    compressed = compress_diff(empty_diff)
    decompressed = decompress_diff(compressed)
    assert decompressed == empty_diff


def test_unicode_diff():
    """Test compression of diff with unicode characters."""
    unicode_diff = "diff --git a/file.py b/file.py\n" \
                   "- old line with émojis 🎉\n" \
                   "+ new line with émojis 🚀\n" * 100
    compressed = compress_diff(unicode_diff)
    decompressed = decompress_diff(compressed)
    assert decompressed == unicode_diff


def test_compress_returns_bytes():
    """Test that compress_diff returns bytes."""
    diff = "test diff content"
    compressed = compress_diff(diff)
    assert isinstance(compressed, bytes)


def test_decompress_accepts_bytes():
    """Test that decompress_diff requires bytes input."""
    diff = "test diff content"
    compressed = compress_diff(diff)
    # Should work with bytes
    decompressed = decompress_diff(compressed)
    assert isinstance(decompressed, str)
    assert decompressed == diff


def test_real_world_git_diff():
    """Test compression with a realistic git diff."""
    git_diff = """diff --git a/src/main.py b/src/main.py
index 1234567..abcdefg 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ class MyClass:
     def method(self):
-        old_code = "something"
+        new_code = "something else"
         return result
 
@@ -20,3 +20,4 @@ def another_function():
     pass
+    # Added comment
"""
    compressed = compress_diff(git_diff)
    decompressed = decompress_diff(compressed)
    assert decompressed == git_diff
    # Verify compression actually happened
    assert len(compressed) < len(git_diff.encode('utf-8'))


def test_compression_level():
    """Test that compression uses level 6 (implicitly through good ratio)."""
    # Create repetitive content that should compress well
    repetitive_diff = "same line repeated\n" * 1000
    compressed = compress_diff(repetitive_diff)
    
    # With level 6, we should get good compression on repetitive content
    ratio = len(compressed) / len(repetitive_diff.encode('utf-8'))
    assert ratio < 0.1  # Should compress very well