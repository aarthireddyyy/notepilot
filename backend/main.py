# backend/main.py
from fastapi import FastAPI

app = FastAPI(title="NotePilot - Backend (Part 0)")

@app.get("/health")
async def health():
    """
    Health check endpoint for Part 0.
    Returns a simple JSON to confirm the server is running.
    """
    return {"status": "ok"}

