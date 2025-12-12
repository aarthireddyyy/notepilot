# backend/vector_store.py
"""
Simple VectorStore wrapper using Chroma + sentence-transformers.

Usage:
    from backend.vector_store import VectorStore
    vs = VectorStore(collection_name="notepilot_chunks", persist_directory="data/chroma_db")
    vs.add_documents(chunks)            # chunks: list of dicts with keys: 'text','doc_name','chunk_index',...
    results = vs.search("what is overfitting?", top_k=3)
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(
        self,
        collection_name: str = "notepilot_chunks",
        persist_directory: str = "data/chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).resolve())
        self.model_name = embedding_model_name

        # initialize embedder (SentenceTransformers)
        self.embedder = SentenceTransformer(self.model_name)

        # Start a persistent Chroma client (stores DB to disk)
        # Using PersistentClient so the DB is saved across runs
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except Exception:
            # fallback to simple in-memory client if PersistentClient not available
            self.client = chromadb.Client()

        # get or create collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return list of float vectors (python lists)."""
        if not texts:
            return []
        embs = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # convert numpy arrays to plain lists for Chroma
        return [emb.tolist() for emb in embs]

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        chunks: list of dicts, each must contain at least:
            - 'text' (str)
            - may contain metadata like 'doc_name', 'chunk_index', 'source_path'
        This method computes embeddings and adds docs to Chroma with metadata + ids.
        """
        if not chunks:
            return

        ids = []
        docs = []
        metadatas = []
        for c in chunks:
            doc_name = c.get("doc_name", "unknown")
            idx = c.get("chunk_index", 0)
            _id = f"{doc_name}::{idx}"
            ids.append(_id)
            docs.append(c["text"])
            # metadata: include doc_name, chunk_index, source_path if available
            metadata = {
                "doc_name": doc_name,
                "chunk_index": idx
            }
            if "source_path" in c:
                metadata["source_path"] = c["source_path"]
            metadatas.append(metadata)

        # compute embeddings
        embeddings = self._embed_texts(docs)

        # add to collection; if ids already exist Chroma will raise — you can choose to delete first
        try:
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            # try upsert logic: delete existing ids then add
            try:
                # remove any existing ids (best-effort)
                for _id in ids:
                    try:
                        self.collection.delete(ids=[_id])
                    except Exception:
                        pass
                self.collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to add documents to Chroma: {e2}") from e2

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query Chroma with a text query. Returns list of dicts:
          { 'id', 'document', 'metadata', 'distance', 'embedding' }
        Note: Some Chroma versions don't support returning 'ids' directly via include,
        so we reconstruct an id from metadata if available.
        """
        if not query:
            return []

        # compute query embedding
        q_emb = self._embed_texts([query])[0]

        # query the collection
        # include allowed items only (ids is not allowed in some chroma versions)
        resp = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        results = []
        if resp:
            # Chroma returns lists-of-lists (batch results). We take the first batch element.
            docs = resp.get("documents", [[]])[0]
            metas = resp.get("metadatas", [[]])[0]
            dists = resp.get("distances", [[]])[0]
            embs = resp.get("embeddings", [[]])[0]

            # loop safety: iterate over docs length
            for i in range(len(docs)):
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                dist = float(dists[i]) if (i < len(dists) and dists[i] is not None) else None
                emb = embs[i] if i < len(embs) else None

                # reconstruct id from metadata if possible (we used doc_name::chunk_index when adding)
                _id = None
                if isinstance(meta, dict):
                    doc_name = meta.get("doc_name")
                    chunk_index = meta.get("chunk_index")
                    if doc_name is not None and chunk_index is not None:
                        _id = f"{doc_name}::{chunk_index}"

                results.append({
                    "id": _id,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                    "embedding": emb
                })
        return results
