#  app/supabase_client.py
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_qa(question, mode, answer, score=None):
    try:
        data = {
            "question": question,
            "mode": mode,
            "answer": answer,
            "score": score
        }
        result = supabase.table("qa_history").insert(data).execute()
        print(" Supabase response:", result)
    except Exception as e:
        print(" Error saving to Supabase:", str(e))

