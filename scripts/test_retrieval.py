# scripts/test_retrieval.py
"""
Query Chroma and print top-k chunks.

This file also ensures the project root is on sys.path so backend imports work when run directly.
Usage:
    python scripts/test_retrieval.py "How to prevent overfitting?"
"""

import sys
import os
# === BEGIN: ensure project root is on sys.path ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# === END: sys.path fix ===

from backend.vector_store import VectorStore

def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Enter a question: ").strip()
    if not q:
        print("No question provided.")
        return

    vs = VectorStore(collection_name="notepilot_chunks", persist_directory="data/chroma_db")
    results = vs.search(q, top_k=3)
    if not results:
        print("No results found. Did you ingest documents to Chroma?")
        return

    for i, r in enumerate(results):
        print(f"\n--- Rank {i+1} | id: {r['id']} | distance: {r['distance']:.4f} ---")
        meta = r.get("metadata", {})
        print(f"source: {meta.get('doc_name')} | chunk: {meta.get('chunk_index')}")
        print(r['document'][:600] + ("..." if len(r['document']) > 600 else ""))

if __name__ == "__main__":
    main()
