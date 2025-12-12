# scripts/ingest_docs.py
"""
Ingest docs from data/raw/ and chunk them into ~N token chunks.
Usage:
  # from project root (notepilot/)
  python scripts/ingest_docs.py

You can also pass optional args:
  python scripts/ingest_docs.py --raw_dir data/raw --target_tokens 300 --overlap 50

This script prints a short summary (number of docs, number of chunks) and
shows a few example chunks.
"""

import os
import re
import argparse
from typing import List, Dict, Any
from pathlib import Path

# PDF loader
try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise ImportError("PyPDF2 is required. Install with `pip install PyPDF2`") from e

# ----------------------
# 1) Loaders
# ----------------------
def load_txt_file(path: Path) -> str:
    """Load .txt or .md file as plain text (utf-8)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf_file(path: Path) -> str:
    """Extract text from a PDF using PyPDF2 PdfReader."""
    text_parts: List[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = ""
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)

def load_document(path: Path) -> str:
    """Dispatch loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return load_txt_file(path)
    elif suffix == ".pdf":
        return load_pdf_file(path)
    else:
        # unsupported file types are ignored
        return ""

# ----------------------
# 2) Simple chunking
# ----------------------
_SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[\.\!\?])\s+')

def approximate_tokens_to_words(tokens: int) -> int:
    """
    Rough conversion: 1 token ≈ 0.75 words (heuristic).
    So words_target = tokens * 0.75
    This is a rough approximation good enough for demo/chunking.
    """
    return max(50, int(tokens * 0.75))

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex (keeps punctuation)."""
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_REGEX.split(text)
    # ensure sentences are trimmed
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(
    text: str,
    doc_name: str,
    target_tokens: int = 300,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Split text into chunks approximately target_tokens in length (approx by words).
    Returns list of dicts with metadata: {doc_name, chunk_index, text}.
    overlap_tokens: approximate token overlap between consecutive chunks (helps retrieval).
    """
    if not text.strip():
        return []

    target_words = approximate_tokens_to_words(target_tokens)
    overlap_words = approximate_tokens_to_words(overlap_tokens)

    sentences = split_into_sentences(text)
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[str] = []
    current_words = 0
    chunk_index = 0

    def flush_chunk():
        nonlocal chunk_index, current_chunk, current_words
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append({
                "doc_name": doc_name,
                "chunk_index": chunk_index,
                "text": chunk_text
            })
            chunk_index += 1
            # prepare next chunk by keeping overlap sentences (as words) if possible
            if overlap_words > 0:
                # keep last sentences until overlap_words reached (approx)
                kept: List[str] = []
                kept_words = 0
                for sent in reversed(current_chunk):
                    wcount = len(sent.split())
                    if kept_words + wcount <= overlap_words or not kept:
                        kept.insert(0, sent)
                        kept_words += wcount
                    else:
                        break
                current_chunk = kept
                current_words = kept_words
            else:
                current_chunk = []
                current_words = 0

    for sent in sentences:
        wcount = len(sent.split())
        # if a single sentence is huge (> target_words), we still add it (avoid infinite loop)
        if current_words + wcount <= target_words or not current_chunk:
            current_chunk.append(sent)
            current_words += wcount
        else:
            # flush current chunk and start new with this sentence
            flush_chunk()
            current_chunk.append(sent)
            current_words = wcount

        # optional: if exactly at target, flush early
        if current_words >= target_words:
            flush_chunk()

    # flush any remaining
    if current_chunk:
        flush_chunk()

    return chunks

# ----------------------
# 3) Ingest script main
# ----------------------
def ingest_folder(raw_dir: Path, target_tokens: int = 300, overlap_tokens: int = 50):
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    # collect files
    supported_ext = {".pdf", ".txt", ".md"}
    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in supported_ext and p.is_file()]

    all_chunks: List[Dict[str, Any]] = []
    for file in files:
        try:
            text = load_document(file)
        except Exception as e:
            print(f"Warning: failed to load {file.name}: {e}")
            continue
        if not text.strip():
            print(f"Skipping empty or unreadable file: {file.name}")
            continue
        chunks = chunk_text(text, doc_name=file.name, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        print(f"Loaded '{file.name}' → {len(chunks)} chunks")
        for c in chunks:
            # add a source path for traceability
            c["source_path"] = str(file)
        all_chunks.extend(chunks)

    # summary
    print("\n--- INGEST SUMMARY ---")
    print(f"Docs found: {len(files)}")
    print(f"Total chunks produced: {len(all_chunks)}")
    if all_chunks:
        print("\nExample chunk 0 metadata:")
        example = all_chunks[0]
        print({k: example[k] for k in ("doc_name", "chunk_index", "source_path")})
        print("\nExample chunk 0 text (first 400 chars):\n")
        print(example["text"][:400] + ("..." if len(example["text"]) > 400 else ""))

    return all_chunks

# ----------------------
# CLI
# ----------------------
def main():
    p = argparse.ArgumentParser(description="Ingest docs from data/raw/ and chunk them.")
    p.add_argument("--raw_dir", type=str, default="data/raw", help="Folder with raw docs")
    p.add_argument("--target_tokens", type=int, default=300, help="Approx target tokens per chunk")
    p.add_argument("--overlap", type=int, default=50, help="Approx overlap tokens between chunks")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    chunks = ingest_folder(raw_dir, target_tokens=args.target_tokens, overlap_tokens=args.overlap)
    print("\nDone.")

if __name__ == "__main__":
    main()
