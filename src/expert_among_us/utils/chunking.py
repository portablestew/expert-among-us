"""Text chunking utilities for managing large embeddings."""


def chunk_text(text: str, chunk_size: int = 8192) -> list[str]:
    """Split text into fixed-size chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Chunk size in characters (default: 8192)
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def chunk_text_with_lines(text: str, chunk_size: int = 8192) -> list[tuple[str, int, int]]:
    """Split text into fixed-size chunks with line number tracking.
    
    Args:
        text: Text to chunk
        chunk_size: Chunk size in characters (default: 10KB)
        
    Returns:
        List of tuples: (chunk_text, line_start, line_end)
    """
    if not text:
        return []
    
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    line_start = 1
    line_num = 1
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        # If adding this line exceeds chunk size and we have content, start new chunk
        if current_size + line_size > chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((chunk_text, line_start, line_num - 1))
            current_chunk = []
            current_size = 0
            line_start = line_num
        
        current_chunk.append(line)
        current_size += line_size
        line_num += 1
    
    # Add final chunk if any
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append((chunk_text, line_start, line_num - 1))
    
    return chunks