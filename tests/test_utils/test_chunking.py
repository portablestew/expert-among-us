"""Tests for chunking utilities to ensure strict size limits."""

import pytest
from expert_among_us.utils.chunking import chunk_text_with_lines


class TestChunkTextWithLines:
    """Test chunk_text_with_lines with focus on size guarantees."""
    
    def test_normal_text_stays_within_limit(self):
        """Normal text should chunk without truncation."""
        text = "line1\nline2\nline3\n" * 100
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        for chunk_text, _, _ in chunks:
            assert len(chunk_text) <= 8192, f"Chunk exceeded limit: {len(chunk_text)}"
    
    def test_single_long_line_gets_truncated(self):
        """A single line exceeding chunk_size must be truncated."""
        # Create a 20KB single line (minified JS scenario)
        long_line = "x" * 20000
        chunks = chunk_text_with_lines(long_line, chunk_size=8192)
        
        assert len(chunks) == 1, "Should produce exactly one chunk"
        chunk_text, line_start, line_end = chunks[0]
        
        # CRITICAL: Chunk must not exceed limit
        assert len(chunk_text) <= 8192, f"Chunk size {len(chunk_text)} exceeded limit 8192"
        # Should have truncation marker
        assert "...[line truncated]" in chunk_text
        # Line numbers should be correct
        assert line_start == 1
        assert line_end == 1
    
    def test_multiple_long_lines_all_truncated(self):
        """Multiple long lines should each be truncated independently."""
        # Three 15KB lines
        text = ("a" * 15000 + "\n" + 
                "b" * 15000 + "\n" + 
                "c" * 15000)
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        # Should produce 3 chunks (one per line since each exceeds limit)
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        
        for chunk_text, _, _ in chunks:
            # CRITICAL: Every chunk must be within limit
            assert len(chunk_text) <= 8192, f"Chunk size {len(chunk_text)} exceeded limit"
            assert "...[line truncated]" in chunk_text
    
    def test_mixed_normal_and_long_lines(self):
        """Mix of normal and degenerate lines should all stay within limit."""
        lines = [
            "normal line 1",
            "x" * 10000,  # Degenerate
            "normal line 2",
            "normal line 3",
            "y" * 15000,  # Degenerate
            "normal line 4"
        ]
        text = "\n".join(lines)
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        for chunk_text, _, _ in chunks:
            # CRITICAL: All chunks must be within limit
            assert len(chunk_text) <= 8192, f"Chunk exceeded limit: {len(chunk_text)}"
    
    def test_edge_case_exactly_at_limit(self):
        """Line exactly at chunk_size should NOT be truncated (fits exactly)."""
        exact_line = "x" * 8192
        chunks = chunk_text_with_lines(exact_line, chunk_size=8192)
        
        assert len(chunks) == 1
        chunk_text, _, _ = chunks[0]
        # Should NOT be truncated - exactly at limit is OK
        assert len(chunk_text) == 8192
        assert "...[line truncated]" not in chunk_text
    
    def test_just_under_limit(self):
        """Line just under limit should not be truncated."""
        under_limit_line = "x" * 8000
        chunks = chunk_text_with_lines(under_limit_line, chunk_size=8192)
        
        assert len(chunks) == 1
        chunk_text, _, _ = chunks[0]
        # Should not be truncated
        assert "...[line truncated]" not in chunk_text
        assert len(chunk_text) == 8000
    
    def test_chunk_size_parameter_respected(self):
        """Custom chunk_size should be respected."""
        text = "x" * 10000
        chunks = chunk_text_with_lines(text, chunk_size=2048)
        
        for chunk_text, _, _ in chunks:
            assert len(chunk_text) <= 2048, f"Chunk exceeded custom limit: {len(chunk_text)}"
    
    def test_base64_scenario(self):
        """Simulate base64 encoded data (common degenerate case)."""
        # Base64 string with no newlines (200KB)
        base64_data = "iVBORw0KGgoAAAANSUhEUgAA" * 8000
        chunks = chunk_text_with_lines(base64_data, chunk_size=8192)
        
        assert len(chunks) == 1
        chunk_text, _, _ = chunks[0]
        assert len(chunk_text) <= 8192, f"Base64 chunk exceeded limit: {len(chunk_text)}"
        assert "...[line truncated]" in chunk_text
    
    def test_minified_javascript_scenario(self):
        """Simulate minified JavaScript (common degenerate case)."""
        # Single line minified JS > 8KB
        minified_js = "var a=1;function b(){return a+1;}" * 300
        chunks = chunk_text_with_lines(minified_js, chunk_size=8192)
        
        for chunk_text, _, _ in chunks:
            assert len(chunk_text) <= 8192, f"Minified JS chunk exceeded limit: {len(chunk_text)}"
    
    def test_empty_text(self):
        """Empty text should return empty list."""
        chunks = chunk_text_with_lines("", chunk_size=8192)
        assert chunks == []
    
    def test_line_numbers_correct_with_truncation(self):
        """Line numbers should remain accurate even with truncation."""
        lines = [
            "line 1",
            "x" * 10000,  # Line 2 - truncated
            "line 3",
            "line 4"
        ]
        text = "\n".join(lines)
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        # Should have 3 chunks: [line1], [line2-truncated], [line3,line4]
        assert len(chunks) == 3
        
        # Verify line numbers
        assert chunks[0][1] == 1 and chunks[0][2] == 1  # Line 1
        assert chunks[1][1] == 2 and chunks[1][2] == 2  # Line 2
        assert chunks[2][1] == 3 and chunks[2][2] == 4  # Lines 3-4
    
    def test_multiple_normal_lines_single_chunk(self):
        """Multiple small lines should combine into single chunk."""
        lines = ["line" + str(i) for i in range(100)]
        text = "\n".join(lines)
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        # Should fit in one chunk
        assert len(chunks) == 1
        chunk_text, line_start, line_end = chunks[0]
        assert len(chunk_text) <= 8192
        assert line_start == 1
        assert line_end == 100
    
    def test_chunk_boundary_splitting(self):
        """Lines should split at chunk boundaries correctly."""
        # Create text that will span multiple chunks
        line = "x" * 4000  # 4KB per line
        text = "\n".join([line] * 5)  # 5 lines = 20KB total
        chunks = chunk_text_with_lines(text, chunk_size=8192)
        
        # Should create 3 chunks: [line1,line2], [line3,line4], [line5]
        assert len(chunks) == 3
        
        for chunk_text, _, _ in chunks:
            assert len(chunk_text) <= 8192
    
    def test_truncation_marker_format(self):
        """Truncation marker should be consistent."""
        long_line = "a" * 20000
        chunks = chunk_text_with_lines(long_line, chunk_size=8192)
        
        chunk_text, _, _ = chunks[0]
        # Should end with truncation marker
        assert chunk_text.endswith("...[line truncated]")
        # Content before marker should be original text
        assert chunk_text.startswith("a" * 100)  # Check first 100 chars
    
    def test_newline_handling(self):
        """Newlines should be handled correctly in chunk size calculation."""
        # Lines with explicit newlines
        text = "line1\nline2\nline3"
        chunks = chunk_text_with_lines(text, chunk_size=100)
        
        # Should produce one chunk since total is small
        assert len(chunks) == 1
        chunk_text, _, _ = chunks[0]
        assert chunk_text == "line1\nline2\nline3"