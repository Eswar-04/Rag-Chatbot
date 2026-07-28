# app/store_vectors.py
from app.parser import load_and_split_pdfs
from app.embedding import embed_documents
from app.supabase_vector import store_embeddings, clear_vectors

def store_vectors():
    try:
        print("🔄 Clearing old vectors...")
        clear_vectors()

        print("🔄 Loading and splitting PDF...")
        chunks = load_and_split_pdfs()

        print(f"🔄 Embedding {len(chunks)} chunks...")
        embeddings = embed_documents(chunks)

        texts = [chunk.page_content for chunk in chunks]

        print("🔄 Storing vectors into Supabase pgvector...")
        store_embeddings(texts, embeddings.tolist())

        print(f"✅ Successfully stored {len(texts)} vectors into Supabase.")
    except Exception as e:
        print("❌ Error in store_vectors:", str(e))
        raise
