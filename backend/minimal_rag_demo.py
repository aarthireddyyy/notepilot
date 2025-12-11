# backend/minimal_rag_demo.py
"""
Minimal in-memory RAG demo (no dependencies).
Usage:
    python backend/minimal_rag_demo.py "your question here"
Or run without args and it will ask for a question interactively.
"""

import sys
import math
import string

# ---------- 1) Hardcoded mini notes (3-5 short paragraphs) ----------
PARAGRAPHS = [
    "Gradient descent is an optimization method that changes parameters gradually to minimize a loss function. "
    "It uses the loss gradient to decide which direction to move parameters.",

    "Overfitting happens when a model learns the training data too well, including noise, which hurts performance on new data. "
    "Regularization and more data can help reduce overfitting.",

    "A confusion matrix shows true vs predicted labels and helps compute metrics like accuracy, precision, and recall. "
    "It is useful for classification evaluation.",

    "Batch normalization normalizes layer inputs during training which can make training faster and more stable. "
    "It reduces internal covariate shift and often improves results."
]

# ---------- 2) Simple text -> vector conversion (bag-of-words / term frequency) ----------
def tokenize(text):
    # lowercase, remove punctuation, split on whitespace
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()

def build_vocab(documents):
    vocab = {}
    for doc in documents:
        for token in tokenize(doc):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab  # token -> index

def text_to_tf_vector(text, vocab):
    vec = [0.0] * len(vocab)
    for token in tokenize(text):
        if token in vocab:
            vec[vocab[token]] += 1.0
    # optional: normalize to unit vector length to make cosine similarity stable
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

# ---------- 3) Cosine similarity ----------
def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be same length")
    # dot product (vectors are already normalized, but we'll compute general formula)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ---------- 4) Retrieval function ----------
def retrieve_most_similar(paragraphs, question, top_k=1):
    # Build vocab from paragraphs (not including the question so retrieval is based on doc space)
    vocab = build_vocab(paragraphs)
    # create vectors for paragraphs
    para_vecs = [text_to_tf_vector(p, vocab) for p in paragraphs]
    # create vector for question (only tokens inside vocab will count)
    q_vec = text_to_tf_vector(question, vocab)
    # compute similarities
    sims = [cosine_similarity(q_vec, pv) for pv in para_vecs]
    # get top_k indices sorted by similarity desc
    ranked = sorted(range(len(paragraphs)), key=lambda i: sims[i], reverse=True)
    return [(i, sims[i], paragraphs[i]) for i in ranked[:top_k]]

# ---------- 5) Main: accept CLI argument or ask input ----------
def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter a question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    print("\nQuestion:", question)
    retrieved = retrieve_most_similar(PARAGRAPHS, question, top_k=1)
    idx, score, paragraph = retrieved[0]
    print(f"\nRetrieved paragraph (index {idx}, similarity {score:.3f}):\n{paragraph}\n")
    # Optional fake generation step using the retrieved paragraph
    print("Answer (using this paragraph):")
    print(f"Based on the retrieved note, here's a short answer: {paragraph.split('.')[0]}.")
    print("\n--- End of demo ---\n")

if __name__ == "__main__":
    main()
