🧠 NotePilot — Your AI-Powered Study Assistant

Smart. Context-aware. Built for students who want superpowers.
NotePilot is an AI study assistant that reads your PDF notes, understands them, and gives accurate, non-hallucinated answers using a custom RAG pipeline.
It’s designed to be fast, minimal, and production-ready, built with technologies used in real AI engineering teams today.

🚀 Features
🔹 1. FastAPI Backend
Lightweight, async API server
Cleanly structured endpoints
Production-ready design

🔹 2. LangChain + ChromaDB Integration
Vector store for persistent embeddings
Query-time filtering
Works offline once your notes are stored

🔹 3. RAG Pipeline That Actually Understands Your Note
PDF text extraction
Chunking + metadata
Continuous updates as notes evolve

🔹 4. Better Embeddings
High-quality open-source embedding models
Improved semantic search
Sharp and relevant answers

🔹 5. Smart Context Filtering (NO Hallucinations)
High-distance chunks are automatically skipped
If your notes don’t contain the answer → it says "Not found"
Ensures high accuracy + reliability

🔹 6. Clean and Minimal UI
Simple input box
Instant responses
Ideal for everyday studying

⚙️ Tech Stack

Backend
FastAPI
LangChain
ChromaDB
Python 3.10+

Frontend
React
Tailwind (optional)
AI Stack
Open-source embeddings
Custom RAG pipeline
Chunk filtering
PDF preprocessing

📘 How It Works (Architecture)
PDF Notes → Chunking → Embeddings → ChromaDB → Similarity Search → Smart Distance Filter → LLM Generates Final Answer

OUTPUT :
<img width="1872" height="968" alt="Screenshot 2025-12-16 230846" src="https://github.com/user-attachments/assets/de9378ae-6418-462e-b7ad-3ad06c3b75c2" />
