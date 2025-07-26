from app.parser import load_and_split_pdfs
from app.embedding import embed_documents
from app.milvus_client import connect_milvus, create_collection

def store_vectors():
    try:
        connect_milvus()
        chunks = load_and_split_pdfs()
        embeddings = embed_documents(chunks)
        collection = create_collection()
        texts = [chunk.page_content for chunk in chunks]
        entities = [embeddings, texts]
        collection.insert(entities)
        print(f" Stored {len(embeddings)} vectors into Milvus.")

        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
        )
        collection.load()
        print(" Collection ready for search.")
    except Exception as e:
        print(" Error in store_vectors:", str(e))
