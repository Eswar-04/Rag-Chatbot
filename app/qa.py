from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from app.milvus_client import connect_milvus
from fastapi import HTTPException

#  Load the sentence transformer model to convert text → vector
model = SentenceTransformer("all-MiniLM-L6-v2")

#  Function to get best matching answer for the given question
def ask_question(question, collection_name="rag_collection"):
    try:
        #  Connect to Milvus database
        connect_milvus()

        #  Convert user's question into embedding vector
        question_embedding = model.encode([question])

        #  Load the Milvus collection (rag_collection)
        collection = Collection(name=collection_name)
        collection.load()

        #  Search for the closest match to the question vector
        results = collection.search(
            data=question_embedding,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=1
        )

        #  Get the top match result from search
        top_match = results[0][0]
        print("Similarity Score:", top_match.distance)

        #  Return the match score to the frontend
        return f" Best match score: {top_match.distance:.4f}"

    except Exception as e:
        #  Print and raise error if anything goes wrong
        print(" Error in qa.py:", str(e))
        raise HTTPException(status_code=500, detail=f"QA internal error: {e}")
