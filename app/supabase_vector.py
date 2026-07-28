# app/supabase_vector.py
# Replaces milvus_client.py - uses Supabase pgvector for vector storage

from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_embeddings(texts: list[str], embeddings: list[list[float]]):
    """Insert text chunks and their embeddings into Supabase."""
    try:
        rows = [
            {"content": text, "embedding": embedding}
            for text, embedding in zip(texts, embeddings)
        ]
        result = supabase.table("document_vectors").insert(rows).execute()
        print(f"✅ Stored {len(rows)} vectors into Supabase pgvector")
        return result
    except Exception as e:
        print("❌ Error storing embeddings:", str(e))
        raise

def search_similar(query_embedding: list[float], top_k: int = 3) -> list[str]:
    """Find top_k most similar chunks using L2 distance via pgvector RPC."""
    try:
        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": top_k
            }
        ).execute()
        return [row["content"] for row in result.data]
    except Exception as e:
        print("❌ Error searching vectors:", str(e))
        raise

def clear_vectors():
    """Delete all stored vectors (useful before re-indexing)."""
    try:
        supabase.table("document_vectors").delete().neq("id", 0).execute()
        print("✅ Cleared all vectors from document_vectors table")
    except Exception as e:
        print("❌ Error clearing vectors:", str(e))
        raise
