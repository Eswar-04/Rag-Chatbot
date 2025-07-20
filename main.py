from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qa import ask_question
from app.store_vectors import store_vectors

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # you can restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QueryRequest(BaseModel):
    query: str

# Main chatbot endpoint
@app.post("/ask")
async def get_answer(req: QueryRequest):
    response = ask_question(req.query)
    return {"answer": response}

# Optional: trigger vector storing
@app.get("/store")
def store():
    store_vectors()
    return {"status": "Vectors stored"}


#  Model for incoming POST request (question from user)
class Question(BaseModel):
    query: str  # frontend will send JSON like { "query": "your question" }

#  POST /ask → accepts question from frontend and returns answer
@app.post("/ask")
async def ask(question: Question):
    answer = ask_question(question.query)  # use qa.py logic to search & get answer
    return {"answer": answer}  # send back to frontend as JSON

#  GET /init → Load all vectors from PDFs into Milvus DB
@app.get("/init")
async def init_store():
    store_vectors()  # runs the vector storing logic
    return {"status": " Stored vectors into Milvus"}  # confirmation

#  CLI (Command Line Interface) support → optional manual running
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "store":
        # Example: python main.py store
        store_vectors()
    else:
        # If no args, show this message
        print(" To store data manually, run: python main.py store")
