# Milvus SDK import – to connect, create, and manage vector collections
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# For reading environment variables from .env file
import os
from dotenv import load_dotenv

# Load all variables from .env into the app (like MILVUS_HOST, MILVUS_PORT)
load_dotenv()

# Read host and port for Milvus connection (from .env or use default)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # // IP or host where Milvus is running
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")      # // Port number for Milvus

# Function to connect to Milvus server
def connect_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)  # // Make connection to Milvus DB
    print("✅ Connected to Milvus")  # // Confirmation message

# Function to create a new collection (like a vector table) in Milvus
def create_collection(collection_name="rag_collection", dim=384):
    # ✅ New way to check if collection exists using utility
    if utility.has_collection(collection_name):
        print("⚠️ Collection already exists")  # // Inform that it already exists
        return Collection(name=collection_name)  # // Just return the existing one

    # Define what fields this collection will have
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # // Auto-generated ID field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),          # // Vector field (384-dim)
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)            # // Store original text
    ]

    # Define schema using above fields
    schema = CollectionSchema(fields, description="RAG Embedding Collection")  # // Define how collection looks

    # Create collection
    collection = Collection(name=collection_name, schema=schema)  # // Create it in Milvus
    print("✅ Collection created:", collection_name)  # // Confirm it's created
    return collection  # // Return collection object
