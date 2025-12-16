# backend/config.py

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

CHROMA_DIR = "data/chroma_db"
CHROMA_COLLECTION = "notepilot_chunks"

TOP_K = 3
MAX_DISTANCE = 1.2
