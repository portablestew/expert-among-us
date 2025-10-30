"""Tests for truncation utilities."""

import pytest

from expert_among_us.utils.truncate import (
    extract_all_filenames_from_diff,
    extract_filename_from_diff,
    is_binary_file,
    truncate_diff_for_embedding,
    truncate_to_bytes,
    truncate_to_tokens,
)


def test_truncate_to_bytes_no_truncation():
    """Test truncating text that fits within byte limit."""
    text = "Hello, world!"
    truncated, was_truncated = truncate_to_bytes(text, max_bytes=100)
    
    assert truncated == text
    assert was_truncated is False


def test_truncate_to_bytes_with_truncation():
    """Test truncating text that exceeds byte limit."""
    text = "a" * 200
    truncated, was_truncated = truncate_to_bytes(text, max_bytes=100)
    
    assert len(truncated.encode("utf-8")) <= 100
    assert was_truncated is True


def test_truncate_to_bytes_utf8_safety():
    """Test that truncation respects UTF-8 boundaries."""
    # Use multibyte characters
    text = "Hello 你好世界 " * 10
    truncated, was_truncated = truncate_to_bytes(text, max_bytes=50)
    
    # Should decode without errors
    assert isinstance(truncated, str)
    # Should be truncated
    assert was_truncated is True


def test_truncate_to_bytes_empty():
    """Test truncating empty string."""
    truncated, was_truncated = truncate_to_bytes("", max_bytes=100)
    
    assert truncated == ""
    assert was_truncated is False


def test_truncate_to_tokens_no_truncation():
    """Test token-based truncation without exceeding limit."""
    text = "Hello, world!"
    truncated, was_truncated = truncate_to_tokens(text, max_tokens=100)
    
    assert truncated == text
    assert was_truncated is False


def test_truncate_to_tokens_with_truncation():
    """Test token-based truncation when exceeding limit."""
    text = "word " * 1000  # ~1000 tokens at 4 chars/token
    truncated, was_truncated = truncate_to_tokens(text, max_tokens=100)
    
    assert len(truncated) < len(text)
    assert was_truncated is True


def test_is_binary_file_text():
    """Test detecting text files."""
    text_content = b"Hello, world!\nThis is text.\n"
    assert is_binary_file(text_content) is False


def test_is_binary_file_binary():
    """Test detecting binary files (with null bytes)."""
    binary_content = b"Hello\x00world"
    assert is_binary_file(binary_content) is True


def test_is_binary_file_mostly_nontext():
    """Test detecting binary files with high ratio of non-text bytes."""
    binary_content = bytes(range(256))
    assert is_binary_file(binary_content) is True


def test_is_binary_file_empty():
    """Test empty content."""
    assert is_binary_file(b"") is False


def test_extract_filename_from_diff_git_format():
    """Test extracting filename from git diff format."""
    diff_line = "diff --git a/src/test.py b/src/test.py"
    filename = extract_filename_from_diff(diff_line)
    
    assert filename == "src/test.py"


def test_extract_filename_from_diff_unified_plus():
    """Test extracting filename from unified diff +++ format."""
    diff_line = "+++ b/src/test.py"
    filename = extract_filename_from_diff(diff_line)
    
    assert filename == "src/test.py"


def test_extract_filename_from_diff_unified_minus():
    """Test extracting filename from unified diff --- format."""
    diff_line = "--- a/src/test.py"
    filename = extract_filename_from_diff(diff_line)
    
    assert filename == "src/test.py"


def test_extract_filename_from_diff_no_match():
    """Test that non-diff lines return None."""
    diff_line = "some random text"
    filename = extract_filename_from_diff(diff_line)
    
    assert filename is None


def test_extract_all_filenames_from_diff():
    """Test extracting all filenames from a complete diff."""
    diff = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
diff --git a/dir/file2.py b/dir/file2.py
--- a/dir/file2.py
+++ b/dir/file2.py
@@ -1,3 +1,3 @@
"""
    
    filenames = extract_all_filenames_from_diff(diff)
    
    assert len(filenames) == 2
    assert "file1.py" in filenames
    assert "dir/file2.py" in filenames


def test_truncate_diff_for_embedding_no_truncation():
    """Test diff truncation when within limits."""
    diff = "small diff content"
    truncated, was_truncated = truncate_diff_for_embedding(diff)
    
    assert truncated == diff
    assert was_truncated is False


def test_truncate_diff_for_embedding_with_truncation():
    """Test diff truncation when exceeding limits."""
    diff = "a" * 40000  # Exceeds 30KB default
    truncated, was_truncated = truncate_diff_for_embedding(diff)
    
    assert len(truncated.encode("utf-8")) < len(diff.encode("utf-8"))
    assert was_truncated is True
    assert "[TRUNCATED" in truncated


def test_truncate_diff_for_embedding_empty():
    """Test truncating empty diff."""
    truncated, was_truncated = truncate_diff_for_embedding("")
    
    assert truncated == ""
    assert was_truncated is False


def test_filter_binary_from_diff_empty():
    """Test filtering empty diff."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    filtered, binary_files = filter_binary_from_diff("")
    
    assert filtered == ""
    assert binary_files == []


def test_filter_binary_from_diff_text_only():
    """Test filtering diff with only text content (should pass through unchanged)."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Text content should pass through unchanged
    assert filtered == diff
    assert binary_files == []


def test_filter_binary_from_diff_binary_only():
    """Test filtering diff with only binary content."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    # Create a diff with binary content (contains null bytes)
    diff = """diff --git a/image.png b/image.png
--- a/image.png
+++ b/image.png
Binary files differ
\x00\x01\x02\x03\x04\x05
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Binary content should be replaced with placeholder
    assert "[binary file: image.png]" in filtered
    assert "image.png" in binary_files
    assert len(binary_files) == 1
    # Original binary content should not be present
    assert "\x00" not in filtered


def test_filter_binary_from_diff_mixed():
    """Test filtering mixed diff with both text and binary files."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    diff = """diff --git a/text.py b/text.py
--- a/text.py
+++ b/text.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
diff --git a/binary.png b/binary.png
--- a/binary.png
+++ b/binary.png
Binary files differ
\x00\x01\x02\x03\x04\x05
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Text file should be preserved
    assert "text.py" in filtered
    assert 'print("Hello, World!")' in filtered
    
    # Binary file should be replaced with placeholder
    assert "[binary file: binary.png]" in filtered
    assert "binary.png" in binary_files
    assert len(binary_files) == 1
    
    # Binary content should not be in filtered diff
    assert "\x00" not in filtered


def test_filter_binary_from_diff_multiple_binary():
    """Test filtering diff with multiple binary files."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    diff = """diff --git a/image1.png b/image1.png
--- a/image1.png
+++ b/image1.png
Binary files differ
\x00\x01\x02\x03\x04\x05
diff --git a/image2.jpg b/image2.jpg
--- a/image2.jpg
+++ b/image2.jpg
Binary files differ
\x00\x06\x07\x08\x09\x0a
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Both binary files should be detected
    assert len(binary_files) == 2
    assert "image1.png" in binary_files
    assert "image2.jpg" in binary_files
    
    # Both should have placeholders
    assert "[binary file: image1.png]" in filtered
    assert "[binary file: image2.jpg]" in filtered
    
    # No binary content should remain
    assert "\x00" not in filtered


def test_filter_binary_from_diff_no_filename():
    """Test filtering diff with malformed section (no filename)."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    # Malformed diff without proper headers
    diff = """Some random content
without proper diff format
@@ -1,3 +1,3 @@
Some changes here
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Should handle gracefully - keep content as-is
    assert filtered == diff
    assert binary_files == []


def test_filter_binary_from_diff_high_nontext_ratio():
    """Test that files with high ratio of non-text bytes are detected as binary."""
    from expert_among_us.utils.truncate import filter_binary_from_diff
    
    # Create diff with high ratio of non-printable characters
    binary_content = "".join(chr(i) for i in range(256))
    diff = f"""diff --git a/data.bin b/data.bin
--- a/data.bin
+++ b/data.bin
@@ -1,3 +1,3 @@
{binary_content}
"""
    
    filtered, binary_files = filter_binary_from_diff(diff)
    
    # Should be detected as binary due to high non-text ratio
    assert "data.bin" in binary_files
    assert "[binary file: data.bin]" in filtered