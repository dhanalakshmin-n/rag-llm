import faiss
import numpy as np

# Global variables to store index and texts
index = None
stored_chunks = []


def create_index(dim: int):
    """
    Create FAISS index with given dimension
    """
    global index
    index = faiss.IndexFlatL2(dim)


def add_embeddings(embeddings, chunks):
    """
    Add embeddings and corresponding text chunks to index
    """
    global stored_chunks

    if index is None:
        raise ValueError("Index not created")

    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)

    stored_chunks.extend(chunks)


def search(query_embedding, top_k=3):
    """
    Search similar chunks
    """
    if index is None:
        raise ValueError("Index not created")

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(stored_chunks):
            results.append(stored_chunks[idx])

    return results
