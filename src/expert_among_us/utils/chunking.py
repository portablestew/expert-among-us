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