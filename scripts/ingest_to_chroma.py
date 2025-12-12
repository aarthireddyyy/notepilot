# scripts/ingest_to_chroma.py
"""
Run ingestion (scripts/ingest_docs.py), then add chunks to Chroma using backend.vector_store.VectorStore.

Usage:
    python scripts/ingest_to_chroma.py
"""
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from pathlib import Path
import importlib.util


# import ingest_folder from scripts/ingest_docs.py dynamically
INGEST_PATH = Path("scripts/ingest_docs.py").resolve()
spec = importlib.util.spec_from_file_location("ingest_scripts", str(INGEST_PATH))
ingest_mod = importlib.util.module_from_spec(spec)
sys.modules["ingest_scripts"] = ingest_mod
spec.loader.exec_module(ingest_mod)  # type: ignore

from ingest_scripts import ingest_folder  # type: ignore

# import VectorStore
from backend.vector_store import VectorStore

def main():
    # ingest chunks from data/raw
    chunks = ingest_folder(Path("data/raw"), target_tokens=300, overlap_tokens=50)
    if not chunks:
        print("No chunks found to ingest. Please add files to data/raw/ and re-run.")
        return

    # initialize vector store (this will create data/chroma_db directory)
    vs = VectorStore(collection_name="notepilot_chunks", persist_directory="data/chroma_db")

    print(f"Adding {len(chunks)} chunks to Chroma (collection: {vs.collection_name}) ...")
    vs.add_documents(chunks)
    print("Done. Chroma DB is persisted at data/chroma_db (if supported).")

if __name__ == "__main__":
    main()
