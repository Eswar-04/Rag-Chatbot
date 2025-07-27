from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qa import ask_question  # Text-based QA handler
from app.store_vectors import store_vectors  # Function to load and store vectors into Milvus
from app.image_qa import image_question_answering  # Image-based QA handler

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define request schema using Pydantic
class Question(BaseModel):
    query: str  # User's question
    mode: str   # Type of question: "text" or "image"

# Route to handle question-answering requests
@app.post("/ask")
async def ask(question: Question):
    try:
        pdf_path = "data/Artificial Intelligence.pdf"  # Path to the PDF used for answering

        # If the mode is 'text', use vector + LLM-based QA
        if question.mode == "text":
            return ask_question(question.query)

        # If the mode is 'image', use image captioning + QA
        elif question.mode == "image":
            return {"answer": image_question_answering(question.query, pdf_path)}

        # Invalid mode
        else:
            return {"answer": "Invalid question mode"}

    except Exception as e:
        # Catch and return any error during processing
        return {"answer": f"Server error: {e}"}

# Route to initialize and store document vectors in Milvus
@app.get("/init")
async def init_store():
    try:
        store_vectors()  # Process PDF and store embeddings
        return {"status": "Stored vectors"}
    except Exception as e:
        # Return error message if vector storing fails
        return {"error": f"Vector storing failed: {e}"}
