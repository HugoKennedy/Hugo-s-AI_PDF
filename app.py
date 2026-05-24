#!/usr/bin/env python3
"""
Minimal PDF-only RAG chat (no CLI args).
- Looks for Alexnet.pdf in the same directory.
- Extracts + chunks with LangChain.
- Indexes with FAISS using BGE-small embeddings.
- Answers questions via a very small Qwen Instruct model.

Run:
  python app.py

Dependencies (install at least these):
  pip install -U "langchain>=0.2" "langchain-community>=0.2" langchain-core \
      langchain-huggingface transformers torch sentence-transformers faiss-cpu pypdf
"""
import os
import sys
from pathlib import Path
import threading


# Prompt + parsing
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except Exception:  # pragma: no cover
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.schema.output_parser import StrOutputParser  # type: ignore
    from langchain.schema.runnable import RunnablePassthrough  # type: ignore

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, pipeline
import re
from sentence_transformers import CrossEncoder
from flask import Flask, jsonify, render_template, request, send_from_directory

# Configuration (no CLI)
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Alexnet.pdf"
INDEX_DIR = BASE_DIR / ".faiss_index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if INDEX_DIR.is_dir() and any(INDEX_DIR.iterdir()):
        try:
            return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(str(INDEX_DIR), embeddings)
    print("[ERROR] Vector index not found. Run: python vectorize.py", file=sys.stderr)
    sys.exit(1)


def create_llm():
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    gen_pipe = pipeline(
        task="text-generation",
        model=LLM_MODEL,
        tokenizer=tok,
        device=device,
        dtype=dtype,  # torch_dtype is deprecated in transformers; use dtype
        max_new_tokens=24,  # very short answers
        do_sample=False,    # deterministic
        repetition_penalty=1.05,
        num_beams=1,
        no_repeat_ngram_size=3,
        return_full_text=False,  # do not echo the prompt
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def create_qa_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    qa = pipeline(
        task="question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        device=device,
        dtype=dtype,
    )
    return qa


def build_retriever(vectorstore: FAISS):
    # Use pure similarity for precise fact lookup; reranker will handle ordering
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 30},
    )


def create_reranker():
    # Lightweight cross-encoder for passage reranking
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Generic utilities to choose non-generic spans from QA outputs
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","to","in","on","for","of","by","with","as","at","from","that","this","these","those","it","its","is","are","was","were","be","been","being","we","you","they","he","she","i","my","our","your","their","his","her","them","me","us","do","does","did","done","can","could","may","might","should","would","will","shall","have","has","had","not","no","yes","one","two","three","four","five","six","seven","eight","nine","ten"
}

def _tokens(s: str):
    toks = re.findall(r"[A-Za-z0-9]+", s.lower())
    return [t for t in toks if t not in STOPWORDS]


def _token_set(s: str):
    return set(_tokens(s))


def make_prompt():
    return PromptTemplate.from_template(
        """
        Answer the question using ONLY the provided Context.
        Your entire answer MUST be copied verbatim as a single contiguous substring from the Context.
        If the answer is not in the Context, reply exactly: I don't know.
        Keep it to a short phrase or sentence (≤15 words).

        Context:
        {context}

        Question: {question}
        Answer:
        """.strip()
    )


# -----------------------------
# Web UI (Flask) embedding - does NOT change the RAG pipeline logic
# -----------------------------
_FLASK_APP: Flask | None = None
_pipeline_lock = threading.Lock()
_pipeline_ready = False

# Global pipeline objects (initialized once)
_vectorstore = None
_llm = None
_qa_pipe = None
_retriever = None
_reranker = None
_prompt = None


def init_pipeline():
    global _pipeline_ready, _vectorstore, _llm, _qa_pipe, _retriever, _reranker, _prompt
    if _pipeline_ready:
        return
    print("[INFO] Loading FAISS index...")
    _vectorstore = load_vectorstore()

    print("[INFO] Loading LLM pipeline (this may download the model on first run)...")
    _llm = create_llm()
    _qa_pipe = create_qa_pipeline()

    _retriever = build_retriever(_vectorstore)
    _reranker = create_reranker()
    _prompt = make_prompt()

    _pipeline_ready = True


def answer_with_pipeline(q: str) -> str:
    # This mirrors the CLI flow exactly.
    docs = _retriever.invoke(q)
    if not docs:
        return "I don't know"

    pairs = [(q, d.page_content) for d in docs]
    with _pipeline_lock:
        scores = _reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:4]]
    ctx = "\n\n".join(d.page_content for d in top_docs)

    with _pipeline_lock:
        qa_out = _qa_pipe(question=q, context=ctx, top_k=5)
    cands = qa_out if isinstance(qa_out, list) else [qa_out]

    qset = _token_set(q)
    def is_informative(ans: str) -> bool:
        aset = _token_set(ans)
        return len(aset) > 0 and not aset.issubset(qset)

    best = None
    for c in sorted(cands, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        a = (c.get("answer", "") or "").strip()
        if a and is_informative(a):
            best = c
            break
    if best is None and cands:
        best = max(cands, key=lambda x: float(x.get("score", 0.0)))

    ans = (best.get("answer", "") if best else "").strip()
    score = float(best.get("score", 0.0)) if best else 0.0

    if not ans or score < 0.25 or not is_informative(ans):
        txt = _prompt.format(context=ctx, question=q)
        with _pipeline_lock:
            gen = _llm.invoke(txt)
        gen = (gen or "").strip().splitlines()[0].strip()
        if gen and gen.lower() != "i don't know" and gen in ctx and is_informative(gen):
            ans = gen

    if not ans:
        ans = "I don't know"
    return ans


def get_flask_app() -> Flask:
    global _FLASK_APP
    if _FLASK_APP is None:
        _FLASK_APP = Flask(__name__, static_folder="static", template_folder="templates")

        @_FLASK_APP.get("/")
        def index():
            return render_template("index.html")

        @_FLASK_APP.post("/ask")
        def ask():
            data = request.get_json(force=True, silent=True) or {}
            q = (data.get("question") or "").strip()
            if not q:
                return jsonify({"ok": False, "error": "Empty question"}), 400
            try:
                ans = answer_with_pipeline(q)
                return jsonify({"ok": True, "answer": ans})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500

        @_FLASK_APP.get("/logo.png")
        def logo_png():
            return send_from_directory(BASE_DIR, "logo.png")

    return _FLASK_APP


# -----------------------------
# Original CLI entry (kept intact)
# -----------------------------

def main():
    print("[INFO] Loading FAISS index...")
    vectorstore = load_vectorstore()

    print("[INFO] Loading LLM pipeline (this may download the model on first run)...")
    llm = create_llm()
    qa_pipe = create_qa_pipeline()

    retriever = build_retriever(vectorstore)
    reranker = create_reranker()
    prompt = make_prompt()

    print("\nRAG chat ready. Type your question (Ctrl+C to exit):")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            # Retrieve candidate docs and rerank by cross-encoder
            docs = retriever.invoke(q)
            if not docs:
                print("\nI don't know\n")
                continue
            pairs = [(q, d.page_content) for d in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            top_docs = [d for d, _ in ranked[:4]]
            ctx = "\n\n".join(d.page_content for d in top_docs)

            # Extractive QA over the merged context (take top-k spans)
            qa_out = qa_pipe(question=q, context=ctx, top_k=5)
            cands = qa_out if isinstance(qa_out, list) else [qa_out]

            qset = _token_set(q)
            def is_informative(ans: str) -> bool:
                aset = _token_set(ans)
                return len(aset) > 0 and not aset.issubset(qset)

            # Choose highest-score informative span; otherwise best overall
            best = None
            for c in sorted(cands, key=lambda x: float(x.get("score", 0.0)), reverse=True):
                a = (c.get("answer", "") or "").strip()
                if a and is_informative(a):
                    best = c
                    break
            if best is None and cands:
                best = max(cands, key=lambda x: float(x.get("score", 0.0)))

            ans = (best.get("answer", "") if best else "").strip()
            score = float(best.get("score", 0.0)) if best else 0.0

            # If low confidence or generic, try constrained generation but only accept substrings that add tokens
            if not ans or score < 0.25 or not is_informative(ans):
                txt = prompt.format(context=ctx, question=q)
                gen = llm.invoke(txt).strip().splitlines()[0].strip()
                if gen and gen.lower() != "i don't know" and gen in ctx and is_informative(gen):
                    ans = gen

            if not ans:
                ans = "I don't know"

            print("\n" + ans + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye")


if __name__ == "__main__":
    # Launch the web UI by default. To use the legacy CLI, run: python app.py --cli
    if "--cli" in sys.argv:
        main()
    else:
        init_pipeline()
        app = get_flask_app()
        print("[INFO] Web UI available at http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)
