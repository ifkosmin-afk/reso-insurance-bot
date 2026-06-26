"""
Build BM25 search index from RESO knowledge base text files.
Splits documents into chunks and saves the BM25 index + metadata.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi

# ── Config ──────────────────────────────────────────────────────────────────
# Use directory of this script so it works both locally and on Railway (/app)
KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH    = os.path.join(KNOWLEDGE_DIR, "reso_bm25.pkl")
CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 150   # overlap characters
# ────────────────────────────────────────────────────────────────────────────


def load_files():
    files = sorted(f for f in os.listdir(KNOWLEDGE_DIR) if f.startswith("База_") and f.endswith(".txt"))
    docs = []
    for fname in files:
        path = os.path.join(KNOWLEDGE_DIR, fname)
        with open(path, "r", encoding="utf-8") as fh:
            docs.append((fname, fh.read()))
    print(f"Loaded {len(docs)} files")
    return docs


def split_by_document(text):
    """Split text into sections by ДОКУМЕНТ: headers."""
    parts = re.split(r"={6,}\n## ДОКУМЕНТ: (.+?)\n={6,}", text)
    sections = []
    if len(parts) < 3:
        sections.append(("unknown", text))
    else:
        for i in range(1, len(parts), 2):
            doc_name = parts[i].strip()
            doc_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((doc_name, doc_text))
    return sections


def chunk_text(text, doc_name, file_name):
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "doc_name": doc_name,
            "file_name": file_name,
        })
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def tokenize(text):
    """Simple Russian-friendly tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r"[а-яёa-z0-9]+", text)
    return tokens


def main():
    print("=== Building RESO BM25 index ===")
    docs = load_files()

    all_chunks = []
    for file_name, text in docs:
        sections = split_by_document(text)
        for doc_name, section_text in sections:
            chunks = chunk_text(section_text, doc_name, file_name)
            all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    tokenized = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": all_chunks, "tokenized": tokenized}, fh)
    print(f"BM25 index saved to {INDEX_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
