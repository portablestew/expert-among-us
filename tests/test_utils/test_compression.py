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