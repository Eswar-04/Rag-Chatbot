# Import the SentenceTransformer class from HuggingFace's sentence-transformers
from sentence_transformers import SentenceTransformer

# Load the embedding model (MiniLM) from HuggingFace
# This model converts English sentences into 384-length vector embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  #  Small, fast, and accurate for document search

# Define a function to generate vector embeddings from document chunks
def embed_documents(chunks):
    # Extract only the text part from each LangChain Document chunk
    texts = [chunk.page_content for chunk in chunks]  #  List of strings like ["Drink water...", "Wake up early...", ...]

    # Use the embedding model to convert texts into numerical vectors
    embeddings = model.encode(texts)  #  Each text → [0.12, -0.45, ..., 0.91] (384 values per chunk)

    # Return the full list of embeddings (to store into Milvus later)
    return embeddings  #  Output: List of vectors (same order as input chunks)
