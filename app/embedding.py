from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model for generating embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_documents(chunks):
    # Extract only the text content from each chunk
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings for the list of text chunks
    embeddings = model.encode(texts)
    
    # Return the list of embeddings
    return embeddings
