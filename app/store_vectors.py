from app.parser import load_and_split_pdfs
from app.embedding import embed_documents
from app.milvus_client import connect_milvus, create_collection

# Store embeddings into Milvus vector DB
def store_vectors():
    connect_milvus()  # // DB connect
    chunks = load_and_split_pdfs()  # // PDF la irundhu text chunks edukkrom
    embeddings = embed_documents(chunks)  # // chunk text → vector
    collection = create_collection()  # // Milvus table ready pannrom

    texts = [chunk.page_content for chunk in chunks]  # get all original texts
    entities = [embeddings, texts]  # store vectors + corresponding texts
    collection.insert(entities)  # // Store vectors into DB
    print(f"✅ Stored {len(embeddings)} vectors into Milvus.")  # confirm

    # ✅ Create IVF_FLAT index
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )
    print("✅ Created IVF_FLAT index")

    # ✅ Load collection to make it searchable
    collection.load()
    print("📥 Collection loaded for search")

# 🧠 Main trigger
if __name__ == "__main__":
    store_vectors()
