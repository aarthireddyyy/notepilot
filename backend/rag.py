# backend/rag.py
"""
RAG logic for NotePilot.

Responsibilities:
- Take a user question
- Retrieve relevant chunks from Chroma (VectorStore)
- Build a guarded prompt
- Call local LLM (Ollama)
- Return answer + sources
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, List
import requests

from backend.vector_store import VectorStore


# ----------------------------
# Prompt template (GUARDRAILS)
# ----------------------------
PROMPT_TEMPLATE = """
You are a helpful study assistant.

Answer the question ONLY using the context provided below.
If the answer is not present in the context, say:
"I'm not sure based on your materials."

Context:
{context}

Question:
{question}

Answer:
"""


# ----------------------------
# Call Ollama (Qwen)
# ----------------------------
def call_llm(prompt: str) -> str:
    """
    Calls local Ollama server to generate text.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]


# ----------------------------
# Main RAG function
# ----------------------------
def answer_question(question: str, top_k: int = 3) -> Dict:
    """
    Full RAG pipeline:
    - retrieve
    - build context
    - prompt LLM
    - return answer + sources
    """
    
    # 1. Initialize vector store
    vs = VectorStore(
        collection_name="notepilot_chunks",
        persist_directory="data/chroma_db"
    )
    logger.info(f"Received question: {question}")

    # 2. Retrieve relevant chunks
    results = vs.search(question, top_k=top_k)

    if not results:
        return {
            "answer": "I'm not sure based on your materials.",
            "sources": []
        }
    logger.info(f"Retrieved {len(results)} chunks from vector store")

    # 3. Build context from retrieved chunks
    context_chunks: List[str] = []
    sources: List[str] = []

    MAX_DISTANCE = 1.2  # heuristic threshold

    for r in results:
        dist = r.get("distance")

    # Skip weak matches
        if dist is not None and dist > MAX_DISTANCE:
            logger.info(f"Skipping chunk due to high distance: {dist}")
            continue

        context_chunks.append(r["document"])

        meta = r.get("metadata", {})
        if "doc_name" in meta:
            sources.append(meta["doc_name"])

    if not context_chunks:
        logger.info("No relevant context after filtering")
        return {
            "answer": "I'm not sure based on your materials.",
            "sources": []
        }
   
       
    context = "\n\n".join(context_chunks)

    # 4. Build guarded prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    logger.info("Calling LLM with grounded prompt")

    # 5. Call LLM
    answer = call_llm(prompt)

    return {
        "answer": answer.strip(),
        "sources": sorted(set(sources))
    }
    logger.info("LLM response generated successfully")
from backend.config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    TOP_K,
    MAX_DISTANCE
)
