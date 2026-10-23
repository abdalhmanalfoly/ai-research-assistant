"""Local Hugging Face embedder for BAAI/bge-large-en-v1.5 (sentence-transformers).

Model-card policy (BAAI/bge-*-en-v1.5): queries get the instruction
"Represent this sentence for searching relevant passages: "; passages get NO instruction.
Both sides use the same L2-normalisation setting, and pgvector uses cosine distance.
All preprocessing lives here so routes/services never invent their own rules.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from typing import Any, Callable

from app.core.errors import (EmbeddingDimensionMismatch, EmbeddingUnavailable, InvalidEmbedding,
                             InvalidEmbeddingInput)

log = logging.getLogger(__name__)
NORM_TOLERANCE = 1e-2


def resolve_device(requested: str, cuda_available: bool) -> str:
    """auto -> cuda if available else cpu. An explicit CUDA request without CUDA FAILS (no silent fallback)."""
    requested = requested.strip().lower()
    if not re.fullmatch(r"auto|cpu|cuda(:\d+)?", requested):
        raise EmbeddingUnavailable("EMBEDDING_DEVICE must be auto, cpu, cuda or cuda:N.")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise EmbeddingUnavailable(
            "EMBEDDING_DEVICE requests CUDA but no CUDA device is available. Use EMBEDDING_DEVICE=cpu or auto, "
            "or install a CUDA-enabled PyTorch build.")
    return requested


def _default_loader(model_name: str, device: str, cache_dir: str | None, offline: bool,
                    dtype: str = "float32") -> Any:
    from sentence_transformers import SentenceTransformer  # heavy import, kept lazy
    model_kwargs = None
    if dtype == "float16" and device.startswith("cuda"):
        import torch
        # Load directly in fp16 so the fp32 weights never occupy VRAM (important on 4 GB cards).
        model_kwargs = {"torch_dtype": torch.float16}
    return SentenceTransformer(model_name, device=device, cache_folder=cache_dir, local_files_only=offline,
                               model_kwargs=model_kwargs)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class HuggingFaceEmbedder:
    def __init__(self, model_name: str, dimension: int, device: str = "auto", batch_size: int = 16,
                 max_seq_length: int = 512, normalize: bool = True,
                 query_instruction: str = "Represent this sentence for searching relevant passages: ",
                 document_prefix: str = "", cache_dir: str | None = None, offline: bool = False,
                 model_loader: Callable[..., Any] | None = None,
                 cuda_available: Callable[[], bool] | None = None, dtype: str = "float32"):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        dtype = dtype.strip().lower()
        if dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16")
        self.model, self.dimension, self.device_setting = model_name, dimension, device
        self.batch_size, self.max_seq_length, self.normalize = batch_size, max_seq_length, normalize
        self.dtype = dtype
        self.query_instruction, self.document_prefix = query_instruction, document_prefix
        self.cache_dir, self.offline = cache_dir, offline
        self._loader = model_loader or _default_loader
        self._cuda_available = cuda_available or _cuda_available
        self._st: Any = None
        self.device: str | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # one encode at a time; the model object is shared

    # ---------- loading / availability ----------
    def load(self) -> None:
        """Load once (thread-safe). Downloads from the Hub only if not cached (unless offline=True -> fails)."""
        if self._st is not None:
            return
        with self._load_lock:
            if self._st is not None:
                return
            device = resolve_device(self.device_setting, self._cuda_available())
            try:
                log.info("loading embedding model %s on %s", self.model, device)
                st = self._loader(self.model, device, self.cache_dir, self.offline, self.dtype)
                if hasattr(st, "max_seq_length"):
                    st.max_seq_length = self.max_seq_length
                if hasattr(st, "eval"):
                    st.eval()
                if self.dtype == "float16":
                    if device.startswith("cuda"):
                        try:
                            cast = st.half()
                            if cast is not None:
                                st = cast
                            log.info("using float16 embeddings on %s", device)
                        except Exception as e:
                            log.warning("could not switch embeddings to float16 on %s (%s); using float32",
                                        device, type(e).__name__)
                    else:
                        log.warning("EMBEDDING_DTYPE=float16 is only supported on CUDA; using float32 on %s", device)
            except EmbeddingUnavailable:
                raise
            except ImportError:
                raise EmbeddingUnavailable(
                    "sentence-transformers/torch are not installed. Run: pip install -r requirements.txt") from None
            except Exception as e:
                log.error("embedding model load failed: %s", type(e).__name__)
                hint = " (offline mode is on: the model must already be cached)" if self.offline else ""
                raise EmbeddingUnavailable(f"Could not load embedding model '{self.model}'{hint}.") from None
            actual = self._model_dimension(st)
            if actual is not None and actual != self.dimension:
                raise EmbeddingDimensionMismatch(
                    f"Model '{self.model}' outputs {actual} dimensions but EMBEDDING_DIM={self.dimension}.")
            self._st, self.device = st, device

    @staticmethod
    def _model_dimension(st: Any) -> int | None:
        fn = getattr(st, "get_sentence_embedding_dimension", None)
        try:
            return int(fn()) if fn else None
        except Exception:
            return None

    def availability(self) -> str:
        """Cheap, no model load, no network: loaded | cached | not_cached | unavailable."""
        if self._st is not None:
            return "loaded"
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return "unavailable"
        try:
            hit = try_to_load_from_cache(self.model, "config.json", cache_dir=self.cache_dir)
            return "cached" if isinstance(hit, str) else "not_cached"
        except Exception:
            return "unavailable"

    # ---------- validation ----------
    @staticmethod
    def _check_texts(texts: list[str]) -> None:
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise InvalidEmbeddingInput()

    def _validate(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise InvalidEmbedding("Embedding model returned an unexpected number of vectors.")
        for v in vectors:
            if len(v) != self.dimension:
                raise EmbeddingDimensionMismatch(
                    f"Embedding has {len(v)} dimensions; expected {self.dimension}.")
            if not all(isinstance(x, float) and math.isfinite(x) for x in v):
                raise InvalidEmbedding("Embedding contains NaN or infinite values.")
            if self.normalize and abs(math.sqrt(sum(x * x for x in v)) - 1.0) > NORM_TOLERANCE:
                raise InvalidEmbedding("Embedding is not L2-normalised although EMBEDDING_NORMALIZE=true.")
        return vectors

    # ---------- encoding ----------
    def _encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):  # bounded memory per call
            batch = texts[i:i + self.batch_size]
            try:
                with self._infer_lock:
                    raw = self._st.encode(batch, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                          convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                log.error("embedding inference failed: %s", type(e).__name__)  # never log the text
                raise EmbeddingUnavailable("Embedding inference failed.") from None
            rows = raw.tolist() if hasattr(raw, "tolist") else raw
            out.extend([[float(x) for x in row] for row in rows])
        return self._validate(out, len(texts))

    def with_query_instruction(self, text: str) -> str:
        """Prepend the query instruction exactly once."""
        if self.query_instruction and not text.startswith(self.query_instruction):
            return self.query_instruction + text
        return text

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise InvalidEmbeddingInput()
        if not texts:
            return []
        self._check_texts(texts)
        return self._encode([self.document_prefix + t for t in texts])  # passages: NO query instruction

    def embed_query_sync(self, text: str) -> list[float]:
        self._check_texts([text])
        return self._encode([self.with_query_instruction(text)])[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query_sync, text)
