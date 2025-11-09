"""Content truncation utilities for handling large diffs and embedding limits."""

import re
from typing import List, Optional


def truncate_to_bytes(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate text to maximum byte size.
    
    Safely truncates text to fit within the specified byte limit while
    respecting UTF-8 character boundaries.
    
    Args:
        text: Text to truncate
        max_bytes: Maximum size in bytes
        
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text:
        return text, False
    
    encoded = text.encode("utf-8")
    
    if len(encoded) <= max_bytes:
        return text, False
    
    # Truncate to max_bytes
    truncated_bytes = encoded[:max_bytes]
    
    # Decode, ignoring incomplete UTF-8 sequences at the end
    truncated_text = truncated_bytes.decode("utf-8", errors="ignore")
    
    return truncated_text, True


def truncate_to_tokens(text: str, max_tokens: int, chars_per_token: float = 4.0) -> tuple[str, bool]:
    """Estimate token count and truncate if necessary.
    
    Uses a rough estimation of characters per token to truncate text
    before sending to embedding models. This is an approximation since
    actual tokenization is model-specific.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum token count
        chars_per_token: Estimated characters per token (default 4.0 for English)
        
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text:
        return text, False
    
    # Estimate token count
    estimated_tokens = len(text) / chars_per_token
    
    if estimated_tokens <= max_tokens:
        return text, False
    
    # Calculate character limit
    max_chars = int(max_tokens * chars_per_token)
    
    # Truncate
    truncated_text = text[:max_chars]
    
    return truncated_text, True


def is_binary_file(content: bytes, sample_size: int = 8192) -> bool:
    """Detect if content is binary.
    
    Checks for null bytes and other binary indicators in a sample of the content.
    
    Args:
        content: File content as bytes
        sample_size: Number of bytes to sample (default 8192)
        
    Returns:
        True if content appears to be binary
    """
    if not content:
        return False
    
    # Sample the content
    sample = content[:sample_size]
    
    # Check for null bytes (strong binary indicator)
    if b"\x00" in sample:
        return True
    
    # Check for high ratio of non-text bytes
    # Text files should have mostly printable ASCII and common whitespace
    text_chars = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13))
    text_ratio = text_chars / len(sample) if sample else 1.0
    
    # If less than 85% text characters, likely binary
    # This catches cases like PNG headers which have some ASCII bytes but
    # also contain non-text control characters (0x89, 0x1A, etc.)
    return text_ratio < 0.85


def extract_filename_from_diff(diff_line: str) -> Optional[str]:
    """Parse filename from a diff line.
    
    Extracts filenames from git diff headers like:
    - "diff --git a/path/to/file b/path/to/file"
    - "+++ b/path/to/file"
    - "--- a/path/to/file"
    
    Args:
        diff_line: A line from a git diff
        
    Returns:
        Filename if found, None otherwise
    """
    # Match diff --git a/file b/file
    git_diff_match = re.match(r"^diff --git a/(.*?) b/", diff_line)
    if git_diff_match:
        return git_diff_match.group(1)
    
    # Match +++ b/file or --- a/file
    unified_diff_match = re.match(r"^[\+\-]{3} [ab]/(.*?)$", diff_line)
    if unified_diff_match:
        return unified_diff_match.group(1)
    
    return None


def extract_all_filenames_from_diff(diff: str) -> List[str]:
    """Extract all filenames from a complete diff.
    
    Parses a git diff to extract all modified file paths.
    
    Args:
        diff: Complete git diff content
        
    Returns:
        List of unique filenames found in the diff
    """
    filenames = set()
    
    for line in diff.split("\n"):
        filename = extract_filename_from_diff(line)
        if filename:
            filenames.add(filename)
    
    return sorted(filenames)


def truncate_diff_for_embedding(
    diff: str,
    max_bytes: int = 100000,
    max_tokens: int = 8192,
) -> tuple[str, bool]:
    """Truncate diff content for embedding generation.
    
    Applies both byte and token limits to ensure diff content fits
    within embedding model constraints (Titan token limit, ~100KB).
    
    Args:
        diff: Diff content to truncate
        max_bytes: Maximum bytes (default 100000 for ~25K tokens)
        max_tokens: Maximum estimated tokens (default 8000)
        
    Returns:
        Tuple of (truncated_diff, was_truncated)
    """
    if not diff:
        return diff, False
    
    # Apply byte limit first
    truncated, truncated_bytes = truncate_to_bytes(diff, max_bytes)
    
    # Then apply token limit
    truncated, truncated_tokens = truncate_to_tokens(truncated, max_tokens)
    
    was_truncated = truncated_bytes or truncated_tokens
    
    if was_truncated:
        truncated += "\n\n[TRUNCATED - content exceeded embedding limits]"
    
    return truncated, was_truncated


def filter_binary_from_diff(diff: str) -> tuple[str, list[str]]:
    """Filter binary file content from a git diff.
    
    Parses a git diff and replaces binary file sections with placeholders,
    preserving only the filename. This ensures binary content doesn't pollute
    stored changelists.
    
    Args:
        diff: Complete git diff content
        
    Returns:
        Tuple of (filtered_diff, list_of_binary_filenames)
        - filtered_diff: Diff with binary sections replaced by placeholders
        - list_of_binary_filenames: List of binary files detected
    """
    if not diff:
        return diff, []
    
    # Split diff into file sections based on "diff --git" headers
    sections = []
    current_section = []
    binary_files = []
    
    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            # Save previous section if it exists
            if current_section:
                sections.append("\n".join(current_section))
            # Start new section
            current_section = [line]
        else:
            current_section.append(line)
    
    # Don't forget the last section
    if current_section:
        sections.append("\n".join(current_section))
    
    # Process each section
    filtered_sections = []
    for section in sections:
        # Extract filename from the section
        filename = None
        for line in section.split("\n"):
            filename = extract_filename_from_diff(line)
            if filename:
                break
        
        if not filename:
            # No filename found, keep section as-is
            filtered_sections.append(section)
            continue
        
        # Check if the section contains binary content
        # Convert section to bytes for binary detection
        section_bytes = section.encode("utf-8", errors="ignore")
        
        if is_binary_file(section_bytes):
            # Replace binary section with placeholder
            binary_files.append(filename)
            placeholder = f"diff --git a/{filename} b/{filename}\n[binary file: {filename}]"
            filtered_sections.append(placeholder)
        else:
            # Keep text file section as-is
            filtered_sections.append(section)
    
    # Reconstruct the diff
    filtered_diff = "\n".join(filtered_sections)
    
    return filtered_diff, binary_files