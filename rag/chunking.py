def fixed_size_chunking(text: str, chunk_size: int = 500):
    """
    Split text into fixed-size chunks
    """

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


##mechanism two

def overlapping_chunking(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split text into overlapping chunks using sliding window
    """

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks
