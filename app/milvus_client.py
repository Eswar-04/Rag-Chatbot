from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

def connect_milvus():
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print(" Connected to Milvus")
    except Exception as e:
        print(" Failed to connect to Milvus:", str(e))

def create_collection(collection_name="rag_collection", dim=384):
    if utility.has_collection(collection_name):
        print(" Collection already exists.")
        return Collection(name=collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="RAG Embedding Collection")
    collection = Collection(name=collection_name, schema=schema)
    print(" Collection created:", collection_name)
    return collection
