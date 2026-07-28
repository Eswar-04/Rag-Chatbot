# app/embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model (384 dimensions)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(chunks) -> np.ndarray:
    """Embed a list of LangChain document chunks."""
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings  # shape: (n_chunks, 384)

def embed_query(query: str) -> list[float]:
    """Embed a single query string and return as a plain list."""
    embedding = model.encode([query])[0]
    return embedding.tolist()
