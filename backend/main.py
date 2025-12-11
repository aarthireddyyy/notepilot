# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .rag import PARAGRAPHS, retrieve_most_similar

app = FastAPI(title="NotePilot - Part 2 (Minimal RAG API)")

# --- Pydantic models for request and response ---
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    retrieved_context: str
    similarity: Optional[float] = None

# --- existing health route (Part 0) ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- new POST endpoint for demo RAG ---
@app.post("/demo/ask", response_model=AskResponse)
async def demo_ask(payload: AskRequest):
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # retrieve the most similar paragraph (top 1)
    results = retrieve_most_similar(PARAGRAPHS, q, top_k=1)
    if not results:
        # fallback (shouldn't happen with hardcoded paragraphs)
        raise HTTPException(status_code=500, detail="No paragraphs available for retrieval")

    idx, sim, paragraph = results[0]

    # optional: simple "answer" composed from the retrieved context (fake generation)
    # For now keep it deterministic and simple (do not call any LLM)
    short_answer = paragraph.split(".")[0].strip() + "."

    response = AskResponse(
        answer=short_answer,
        retrieved_context=paragraph,
        similarity=round(sim, 4)
    )
    return response
