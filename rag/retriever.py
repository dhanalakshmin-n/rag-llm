from rag.embeddings import embed_query
from rag.vectorstore import search


def retrieve_relevant_chunks(query: str, top_k: int = 3):
    """
    Retrieve top relevant chunks for a query
    """

    # Convert query to embedding
    query_embedding = embed_query(query)

    # Search vector store
    results = search(query_embedding, top_k)

    return results
