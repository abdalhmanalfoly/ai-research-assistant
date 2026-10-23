"""TEST-ONLY synthetic fakes. Not part of the production path (production uses Ollama + PostgreSQL/pgvector).
All paper data below is synthetic and does not describe real publications."""
from __future__ import annotations

import hashlib
import math
import re

from app.services.types import PaperRef, ScoredChunk, SearchFilters

DIM = 64


def hash_vector(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self, fail=None):
        self.fail = fail

    async def embed_query(self, text):
        if self.fail:
            raise self.fail
        return hash_vector(text)

    async def embed_documents(self, texts):
        return [hash_vector(t) for t in texts]


class InMemoryStore:
    def __init__(self, rows=None, fail=None):
        self.rows: list[tuple[PaperRef, int | None, str | None, int, str]] = rows or []
        self.fail = fail

    def add(self, paper: PaperRef, page, text, section=None):
        self.rows.append((paper, page, section, len(self.rows), text))

    def search(self, vector, limit, filters: SearchFilters):
        if self.fail:
            raise self.fail
        out = []
        for i, (paper, page, section, idx, text) in enumerate(self.rows):
            if filters.paper_ids and paper.id not in filters.paper_ids:
                continue
            if filters.year_min and (paper.year or 0) < filters.year_min:
                continue
            if filters.year_max and (paper.year or 9999) > filters.year_max:
                continue
            score = sum(a * b for a, b in zip(vector, hash_vector(text)))
            out.append(ScoredChunk(f"c{i}", text, page, section, idx, score, paper))
        return sorted(out, key=lambda c: -c.score)[:limit]

    def count_indexed_chunks(self):
        if self.fail:
            raise self.fail
        return len(self.rows)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply=None, fail=None):
        self.reply, self.fail, self.calls = reply, fail, []

    async def chat(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise self.fail
        return self.reply if isinstance(self.reply, str) else self.reply(messages)


PAPER_A = PaperRef("p-a", "Synthetic Paper A on Retinal Imaging", ["A. Author"], "10.0000/synthetic-a", None, 2021)
PAPER_B = PaperRef("p-b", "Synthetic Paper B on Segmentation", ["B. Author"], None, "https://example.invalid/b", 2023)


# ---------------- embedding / reindex fakes (TEST-ONLY) ----------------
class FakeSTModel:
    """Stands in for sentence_transformers.SentenceTransformer. Deterministic, L2-normalised, 1024-dim by default."""

    def __init__(self, dim=1024, mode="ok"):
        self.dim, self.mode, self.calls, self.max_seq_length, self.evaluated = dim, mode, [], None, False
        self.half_called = False

    def eval(self):
        self.evaluated = True

    def half(self):
        self.half_called = True
        return self

    def get_sentence_embedding_dimension(self):
        return self.dim

    def vector(self, text):
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()) or ["_"]:
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def encode(self, texts, **kw):
        self.calls.append((list(texts), kw))
        if self.mode == "raise":
            raise RuntimeError("boom: secret paper text " + texts[0])
        out = [self.vector(t) for t in texts]
        if self.mode == "nan":
            out[0][0] = float("nan")
        elif self.mode == "inf":
            out[0][0] = float("inf")
        elif self.mode == "short":
            out = [v[:-1] for v in out]
        elif self.mode == "unnormalised":
            out = [[x * 3 for x in v] for v in out]
        elif self.mode == "drop":
            out = out[:-1]
        return out


class FakeChunkRepo:
    """In-memory ChunkRepo. rows: id -> {text, vec, model}. Pending = no vec or model != current."""

    def __init__(self, texts: dict[str, str], model="BAAI/bge-large-en-v1.5", done: set[str] = ()):
        self.model = model
        self.rows = {i: {"text": t, "vec": [0.0] if i in done else None, "model": model if i in done else "nomic-embed-text"}
                     for i, t in texts.items()}
        self.saves = 0

    def _pending(self, r):
        return r["vec"] is None or r["model"] != self.model

    def count_pending(self):
        return sum(self._pending(r) for r in self.rows.values())

    def fetch_pending(self, after_id, limit):
        ids = sorted(i for i, r in self.rows.items() if self._pending(r) and (after_id is None or i > after_id))
        return [(i, self.rows[i]["text"]) for i in ids[:limit]]

    def save(self, items):
        self.saves += 1
        for i, v in items:
            self.rows[i].update(vec=v, model=self.model)
