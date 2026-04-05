from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import query_rag

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG API running 🚀"}


@app.post("/ask")
def ask(req: QueryRequest):
    try:
        response = query_rag(req.question)
        return {
            "question": req.question,
            "answer": response
        }
    except Exception as e:
        return {
            "error": str(e)
        }