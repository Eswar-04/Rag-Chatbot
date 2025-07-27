from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Milvus host and port from environment variables (with default fallback)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Function to establish connection to the Milvus server
def connect_milvus():
    try:
        # Connect to Milvus using the provided host and port
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        print(" Connected to Milvus")
    except Exception as e:
        # Print error message if connection fails
        print(" Failed to connect to Milvus:", str(e))

# Function to create a new collection in Milvus (if it doesn't already exist)
def create_collection(collection_name="rag_collection", dim=384):
    # Check if the collection already exists
    if utility.has_collection(collection_name):
        print(" Collection already exists.")
        return Collection(name=collection_name)

    # Define the schema fields:
    # 1. 'id' as primary key (auto-generated)
    # 2. 'embedding' field to store vector embeddings
    # 3. 'text' field to store original text
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    # Create collection schema with field definitions
    schema = CollectionSchema(fields, description="RAG Embedding Collection")

    # Create the collection using the defined schema
    collection = Collection(name=collection_name, schema=schema)
    print(" Collection created:", collection_name)

    # Return the created collection object
    return collection
