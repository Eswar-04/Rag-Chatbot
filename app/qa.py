#  app/qa.py
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from app.milvus_client import connect_milvus
from fastapi import HTTPException
from transformers import pipeline
from wordsegment import load, segment
from app.supabase_client import store_qa
import re

# Load vocabulary for wordsegment
load()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def split_glued_text(glued):
    return " ".join(segment(glued))

def clean_text(text):
    text = text.strip()

    #  Wordsegment fix for long glued words
    words = text.split()
    processed = []
    for word in words:
        if len(word) > 10 and not re.search(r'\s', word):
            processed.append(split_glued_text(word))
        else:
            processed.append(word)
    text = " ".join(processed)

    # 🧹 Remove repeated phrases
    text = re.sub(r'(\b\w+\b(?: \b\w+\b){0,2}) \1+', r'\1', text)

    #  Removed space remover regex that caused bugs
    #  Disabled camelCase splitting for accuracy

    # Fix common split pattern like "system s" → "systems"
    text = re.sub(r'\b([a-zA-Z]+) s\b', r'\1s', text)

    #  Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    #  Punctuation spacing
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)

    return text

def ask_question(question, collection_name="rag_collection"):
    try:
        connect_milvus()
        question_embedding = embedder.encode([question])
        collection = Collection(name=collection_name)
        collection.load()

        results = collection.search(
            data=question_embedding,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3,
            output_fields=["text"],
        )

        top_matches = results[0][:3]
        matched_chunks = " ".join([match.entity.get("text") for match in top_matches])
        score = top_matches[0].distance

        prompt = (
            f"You are an AI assistant helping with textbook content. "
            f"Use the following content to answer clearly and informatively.\n\n"
            f"Context:\n{matched_chunks}\n\n"
            f"Question:\n{question}"
        )

        raw_output = qa_model(prompt, max_new_tokens=250, do_sample=False)[0]["generated_text"]
        final_answer = clean_text(raw_output)

        # ✅ Save to Supabase
        store_qa(question, "text", final_answer, score)

        return {"answer": final_answer, "score": score}

    except Exception as e:
        print("❌ Error in ask_question:", str(e))
        raise HTTPException(status_code=500, detail=f"QA internal error: {e}")
