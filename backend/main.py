# backend/main.py
"""
FastAPI entry point for NotePilot RAG backend.

Responsibilities:
- Accept user questions
- Call RAG pipeline
- Return answer + sources
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.rag import answer_question

app = FastAPI(title="NotePilot RAG API")


# ----------------------------
# Request / Response schemas
# ----------------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Main RAG endpoint
# ----------------------------
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = answer_question(question)

    return result
