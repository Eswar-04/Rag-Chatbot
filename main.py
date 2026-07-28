from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qa import ask_question
from app.store_vectors import store_vectors
from app.image_qa import image_question_answering
import os

app = FastAPI()

# FRONTEND_URL: set this in Render dashboard to your Streamlit service URL
# e.g. https://rag-chatbot-ui.onrender.com
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8501",   # local Streamlit default port
]

if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str
    mode: str

@app.post("/ask")
async def ask(question: Question):
    try:
        pdf_path = "data/Artificial Intelligence.pdf"
        if question.mode == "text":
            return ask_question(question.query)
        elif question.mode == "image":
            return {"answer": image_question_answering(question.query, pdf_path)}
        else:
            return {"answer": "Invalid question mode"}
    except Exception as e:
        return {"answer": f"Server error: {e}"}

@app.get("/init")
async def init_store():
    try:
        store_vectors()
        return {"status": "Stored vectors"}
    except Exception as e:
        return {"error": f"Vector storing failed: {e}"}
