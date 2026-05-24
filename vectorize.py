#!/usr/bin/env python3
"""
Build the FAISS vector index once, then reuse from app.py.

Run:
  python vectorize.py

This script reads Alexnet.pdf, cleans and chunks it, embeds with BGE-small,
creates a FAISS index, and saves it to .faiss_index/.
"""
import sys
import re
from pathlib import Path

# Shared config from app.py
from app import PDF_PATH, INDEX_DIR, EMBED_MODEL

# Local research directory containing PDFs to index
BASE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = BASE_DIR / "research"

# Imports needed only for vectorization
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


def load_and_chunk(pdf_path: Path):
    """Load and chunk all PDFs from the local research folder.

    If the research directory does not exist, fall back to the original
    single-PDF behaviour using pdf_path.
    """
    pages = []

    if RESEARCH_DIR.is_dir():
        pdf_files = sorted(RESEARCH_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"[ERROR] No PDF files found in {RESEARCH_DIR}", file=sys.stderr)
            sys.exit(1)
        for path in pdf_files:
            loader = PyPDFLoader(str(path))
            pages.extend(loader.load())
    else:
        if not pdf_path.exists():
            print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

    # Clean PDF text: de-hyphenate across line breaks and normalize whitespace
    for d in pages:
        txt = d.page_content
        txt = re.sub(r"-\s*\n\s*", "", txt)      # join hyphenated breaks
        txt = re.sub(r"\s*\n\s*", " ", txt)      # flatten newlines
        txt = re.sub(r"\s+", " ", txt).strip()     # collapse spaces
        d.page_content = txt

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_and_save_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    return vs


def main():
    print("[VECTORIZER] Loading and chunking PDF...")
    chunks = load_and_chunk(PDF_PATH)
    print(f"[VECTORIZER] Chunks: {len(chunks)}")

    print(f"[VECTORIZER] Building FAISS index at {INDEX_DIR} ...")
    build_and_save_index(chunks)
    print("[VECTORIZER] Done. You can now run: python app.py")


if __name__ == "__main__":
    main()
