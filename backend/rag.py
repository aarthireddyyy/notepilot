# backend/rag.py
# Simple in-memory retrieval logic (very small TF-based vectors).
# Meant to be imported by FastAPI endpoint.

import math
import string
from typing import List, Tuple

# Hardcoded paragraphs (study notes)
PARAGRAPHS: List[str] = [
    "Gradient descent is an optimization method that changes parameters gradually to minimize a loss function. "
    "It uses the loss gradient to decide which direction to move parameters.",

    "Overfitting happens when a model learns the training data too well, including noise, which hurts performance on new data. "
    "Regularization and more data can help reduce overfitting.",

    "A confusion matrix shows true vs predicted labels and helps compute metrics like accuracy, precision, and recall. "
    "It is useful for classification evaluation.",

    "Batch normalization normalizes layer inputs during training which can make training faster and more stable. "
    "It reduces internal covariate shift and often improves results."
]

# --- simple text processing / TF vectorizer ---
def tokenize(text: str) -> List[str]:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()

def build_vocab(documents: List[str]) -> dict:
    vocab = {}
    for doc in documents:
        for token in tokenize(doc):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def text_to_tf_vector(text: str, vocab: dict) -> List[float]:
    vec = [0.0] * len(vocab)
    for token in tokenize(text):
        if token in vocab:
            vec[vocab[token]] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        # if dimensions mismatch, return 0 similarity
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# --- retrieval function used by the API ---
def retrieve_most_similar(paragraphs: List[str], question: str, top_k: int = 1) -> List[Tuple[int, float, str]]:
    """
    Returns list of tuples: (index, similarity_score, paragraph) for top_k results.
    """
    vocab = build_vocab(paragraphs)
    para_vecs = [text_to_tf_vector(p, vocab) for p in paragraphs]
    q_vec = text_to_tf_vector(question, vocab)
    sims = [cosine_similarity(q_vec, pv) for pv in para_vecs]
    ranked = sorted(range(len(paragraphs)), key=lambda i: sims[i], reverse=True)
    return [(i, sims[i], paragraphs[i]) for i in ranked[:top_k]]
