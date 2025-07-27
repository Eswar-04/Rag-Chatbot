from app.parser import load_and_split_pdfs  # Loads and splits PDF into text chunks
from app.embedding import embed_documents    # Converts text chunks into embeddings
from app.milvus_client import connect_milvus, create_collection  # Milvus DB operations

# Function to store text embeddings into Milvus vector database
def store_vectors():
    try:
        # Connect to the Milvus server
        connect_milvus()

        # Load and split the PDF into smaller chunks
        chunks = load_and_split_pdfs()

        # Generate vector embeddings for each text chunk
        embeddings = embed_documents(chunks)

        # Create or retrieve the target collection in Milvus
        collection = create_collection()

        # Prepare the text content for storage
        texts = [chunk.page_content for chunk in chunks]

        # Create a list of entities (embeddings + text) to insert into Milvus
        entities = [embeddings, texts]

        # Insert data into the Milvus collection
        collection.insert(entities)
        print(f" Stored {len(embeddings)} vectors into Milvus.")

        # Create an index on the 'embedding' field for efficient vector search
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "L2",         # Distance metric to use for similarity
                "index_type": "IVF_FLAT",    # Index type
                "params": {"nlist": 128}     # Number of clusters
            }
        )

        # Load the collection into memory for querying
        collection.load()
        print(" Collection ready for search.")

    except Exception as e:
        # Handle and print any errors during the process
        print(" Error in store_vectors:", str(e))
