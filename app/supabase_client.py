# app/supabase_client.py

from supabase import create_client  # Supabase Python client
import os
from dotenv import load_dotenv  # To load environment variables from .env file

# Load environment variables
load_dotenv()

# Get Supabase URL and Key from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to store a question-answer pair into the Supabase table
def store_qa(question, mode, answer, score=None):
    try:
        # Prepare data dictionary to be inserted
        data = {
            "question": question,  # User's question
            "mode": mode,          # Mode of QA (e.g., 'text' or 'image')
            "answer": answer,      # Generated answer
            "score": score         # Optional: similarity score from vector search
        }

        # Insert data into the 'qa_history' table
        result = supabase.table("qa_history").insert(data).execute()

        # Print Supabase's response (useful for debugging/logging)
        print(" Supabase response:", result)

    except Exception as e:
        # Handle and print errors if insertion fails
        print(" Error saving to Supabase:", str(e))
