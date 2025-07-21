from sentence_transformers import SentenceTransformer  #  Load model to convert text to vector
from pymilvus import Collection  #  Used to load and search vector collection
from app.milvus_client import connect_milvus  #  Milvus DB connection function
from fastapi import HTTPException  #  To handle backend errors

#  Load sentence transformer model for embedding text
model = SentenceTransformer("all-MiniLM-L6-v2")

#  Main function to process user question and fetch best answer
def ask_question(question, collection_name="rag_collection"):
    try:
        #  Connect to Milvus DB
        connect_milvus()

        #  Convert user’s question into vector form
        question_embedding = model.encode([question])

        #  Load the vector collection (table) from Milvus
        collection = Collection(name=collection_name)
        collection.load()

        #  Perform vector similarity search to find best match
        results = collection.search(
            data=question_embedding,  #  query vector
            anns_field="embedding",  #  field that contains vectors
            param={"metric_type": "L2", "params": {"nprobe": 10}},  #  search config
            limit=1,  #  return only top 1 result
            output_fields=["text"],  #  fetch text content of best match
        )

        #  Get the best matched result from Milvus
        top_match = results[0][0]
        print("Similarity Score:", top_match.distance)

        #  Return both the matching text and score to frontend
        return {
            "answer": top_match.entity.get("text"),
            "score": top_match.distance
        }

    except Exception as e:
        # ✅ If any error occurs, raise it with HTTP error message
        print(" Error in qa.py:", str(e))
        raise HTTPException(status_code=500, detail=f"QA internal error: {e}")
