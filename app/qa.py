# app/qa.py
from app.embedding import embed_query
from app.supabase_vector import search_similar
from app.supabase_client import store_qa
from transformers import pipeline
from wordsegment import load, segment
from fastapi import HTTPException
import re

# Load vocabulary for wordsegment
load()

qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def split_glued_text(glued):
    return " ".join(segment(glued))

def clean_text(text):
    text = text.strip()

    # Wordsegment fix for long glued words
    words = text.split()
    processed = []
    for word in words:
        if len(word) > 10 and not re.search(r'\s', word):
            processed.append(split_glued_text(word))
        else:
            processed.append(word)
    text = " ".join(processed)

    # Remove repeated phrases
    text = re.sub(r'(\b\w+\b(?: \b\w+\b){0,2}) \1+', r'\1', text)

    # Fix common split pattern like "system s" → "systems"
    text = re.sub(r'\b([a-zA-Z]+) s\b', r'\1s', text)

    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Punctuation spacing
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text

def ask_question(question: str):
    try:
        # 1. Embed the query
        query_embedding = embed_query(question)

        # 2. Search similar chunks from Supabase pgvector
        matched_chunks = search_similar(query_embedding, top_k=3)

        if not matched_chunks:
            return {"answer": "No relevant content found in the document.", "score": None}

        context = " ".join(matched_chunks)

        # 3. Build prompt for FLAN-T5
        prompt = (
            f"You are an AI assistant helping with textbook content. "
            f"Use the following content to answer clearly and informatively.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

        # 4. Generate answer
        raw_output = qa_model(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
        final_answer = clean_text(raw_output)

        # 5. Save to Supabase qa_history
        store_qa(question, "text", final_answer, score=None)

        return {"answer": final_answer}

    except Exception as e:
        print("❌ Error in ask_question:", str(e))
        raise HTTPException(status_code=500, detail=f"QA internal error: {e}")
