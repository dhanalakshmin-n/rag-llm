from sentence_transformers import SentenceTransformer

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list):
    """
    Convert list of texts into embeddings
    """
    return model.encode(texts)


def embed_query(query: str):
    """
    Convert single query into embedding
    """
    return model.encode([query])[0]
