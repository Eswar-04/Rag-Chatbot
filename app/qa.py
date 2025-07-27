# app/qa.py

from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from app.milvus_client import connect_milvus
from fastapi import HTTPException
from transformers import pipeline
from wordsegment import load, segment
from app.supabase_client import store_qa
import re

# Load vocabulary for the wordsegment library
load()

# Initialize the embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FLAN-T5 model for generating answers
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Function to split long glued words into proper English words
def split_glued_text(glued):
    return " ".join(segment(glued))

# Function to clean and enhance the generated text
def clean_text(text):
    text = text.strip()

    # Fix long glued words using wordsegment
    words = text.split()
    processed = []
    for word in words:
        # Apply split only to long words that are not already spaced
        if len(word) > 10 and not re.search(r'\s', word):
            processed.append(split_glued_text(word))
        else:
            processed.append(word)
    text = " ".join(processed)

    # Remove repeated short phrases (1-3 words repeating)
    text = re.sub(r'(\b\w+\b(?: \b\w+\b){0,2}) \1+', r'\1', text)

    # Fix pattern like "system s" → "systems"
    text = re.sub(r'\b([a-zA-Z]+) s\b', r'\1s', text)

    # Capitalize the first letter if it's lowercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Fix punctuation spacing (e.g., remove extra spaces before commas or periods)
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize all spacing

    return text

# Function to answer a user question using vector search and LLM
def ask_question(question, collection_name="rag_collection"):
    try:
        # Connect to Milvus vector database
        connect_milvus()

        # Generate embedding for the input question
        question_embedding = embedder.encode([question])

        # Load the specified collection from Milvus
        collection = Collection(name=collection_name)
        collection.load()

        # Perform vector similarity search to find top matching text chunks
        results = collection.search(
            data=question_embedding,
            anns_field="embedding",  # Field in Milvus to compare embeddings
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3,  # Number of top matches to retrieve
            output_fields=["text"],  # We only need the text field in the result
        )

        # Extract top matching chunks from search results
        top_matches = results[0][:3]
        matched_chunks = " ".join([match.entity.get("text") for match in top_matches])
        score = top_matches[0].distance  # Distance score of best match

        # Construct a prompt for the QA model using the matched content
        prompt = (
            f"You are an AI assistant helping with textbook content. "
            f"Use the following content to answer clearly and informatively.\n\n"
            f"Context:\n{matched_chunks}\n\n"
            f"Question:\n{question}"
        )

        # Generate the answer using FLAN-T5 model
        raw_output = qa_model(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]

        # Clean and post-process the generated answer
        final_answer = clean_text(raw_output)

        # Append the vector match score to the answer
        final_answer_with_score = f"{final_answer}\n\nVector Best Score: {score:.6f}"

        # Store the question and answer in Supabase for logging/auditing
        store_qa(question, "text", final_answer_with_score, score)

        # Return the final answer and score
        return {"answer": final_answer_with_score, "score": score}

    except Exception as e:
        # Log and raise an HTTP exception if anything goes wrong
        print(" Error in ask_question:", str(e))
        raise HTTPException(status_code=500, detail=f"QA internal error: {e}")
