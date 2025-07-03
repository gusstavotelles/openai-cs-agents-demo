"""
RAG store: ingere PDFs, gera embeddings (OpenAI) e realiza busca com FAISS.

Artefatos persistidos:
    data/rag_index.faiss  – índice FAISS binário
    data/rag_chunks.jsonl – JSONL com {"chunk": ..., "source": ...}

Uso:
    from rag_store import add_pdf, query_rag
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss  # type: ignore
from pypdf import PdfReader  # mais leve que pdfplumber
import openai

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

INDEX_PATH = DATA_DIR / "rag_index.faiss"
META_PATH = DATA_DIR / "rag_chunks.jsonl"

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def _embed_texts(texts: List[str]) -> np.ndarray:
    """Chama a API de Embeddings da OpenAI e devolve ndarray (n, d) float32."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    res = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = np.array([d.embedding for d in res.data], dtype=np.float32)
    return arr


def _load_index() -> faiss.IndexFlatIP | None:
    if not INDEX_PATH.exists():
        return None
    return faiss.read_index(str(INDEX_PATH))


def _save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, str(INDEX_PATH))


def _load_meta() -> List[Dict[str, Any]]:
    if not META_PATH.exists():
        return []
    with META_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _append_meta(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with META_PATH.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────
def add_pdf(file_bytes: bytes, filename: str) -> None:
    """
    Ingere um PDF: extrai texto, divide em chunks, gera embeddings
    e adiciona ao índice.
    """
    reader = PdfReader(BytesIO(file_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError("PDF sem texto extraível")

    # Split simples: parágrafos em chunks <= ~700 chars
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < 700:
            buf += (" " if buf else "") + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # Embeddings
    embeds = _embed_texts(chunks)
    if embeds.shape[0] == 0:
        raise ValueError("Nenhum chunk embutido")

    # Índice FAISS (Produto Interno + normalização -> cosseno)
    index = _load_index()
    if index is None:
        dim = embeds.shape[1]
        index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeds)
    index.add(embeds)

    # Persistir
    _save_index(index)
    _append_meta([{"chunk": c, "source": filename} for c in chunks])


def query_rag(question: str, k: int = 4) -> List[Dict[str, Any]]:
    """Retorna top-k chunks com score e fonte."""
    index = _load_index()
    metas = _load_meta()
    if index is None or not metas:
        return []

    q_vec = _embed_texts([question])
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k)  # type: ignore
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if 0 <= idx < len(metas):
            m = metas[idx]
            results.append(
                {"chunk": m["chunk"], "source": m["source"], "score": float(score)}
            )
    return results
