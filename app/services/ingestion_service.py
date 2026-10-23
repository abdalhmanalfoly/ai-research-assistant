"""PDF -> pages -> chunks -> embeddings -> pgvector. Idempotent on the PDF's SHA-256."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete, select

from app.core.errors import (AppError, EmbeddingDimensionMismatch, InvalidUpload, NotFound, OCRRequired,
                              PDFExtractionError)
from app.db.models import Chunk, IngestionJob, Paper
from app.services.chunking import PageText, chunk_pages
from app.services.ocr import OCREngine
from app.services.types import Embedder

log = logging.getLogger(__name__)
MIN_CHARS_PER_PAGE = 25  # below this average the PDF is treated as scanned/image-only


def extract_pages(pdf: bytes, ocr: OCREngine | None = None, ocr_dpi: int = 200,
                  ocr_min_page_chars: int = 25, ocr_max_pages: int = 0) -> list[PageText]:
    """Extract text per page with PyMuPDF. page_number is the 1-based physical PDF page.

    Pages whose text layer is shorter than ocr_min_page_chars are rendered and passed to the
    OCR engine (when one is configured). OCR never spans pages, so page numbers stay exact.
    """
    try:
        import pymupdf as fitz  # PyMuPDF (the fitz alias is deprecated)
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            if doc.needs_pass:
                raise PDFExtractionError("The PDF is password protected.")
            pages: list[PageText] = []
            ocr_pages = 0
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if ocr is not None and len(text.strip()) < ocr_min_page_chars:
                    if ocr_max_pages and ocr_pages >= ocr_max_pages:
                        raise OCRRequired(
                            f"OCR is required for more than OCR_MAX_PAGES={ocr_max_pages} pages. "
                            "Raise OCR_MAX_PAGES (or split the document) and upload it again.")
                    pix = page.get_pixmap(dpi=ocr_dpi, colorspace=fitz.csRGB, alpha=False)
                    ocr_text = ocr.image_to_text(pix.tobytes("png"))
                    ocr_pages += 1
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                pages.append(PageText(i, text))
            if ocr_pages:
                log.info("OCR applied to %d of %d page(s)", ocr_pages, len(pages))
            return pages
    except AppError:
        raise
    except Exception:
        log.exception("PDF extraction failed")
        raise PDFExtractionError() from None


class IngestionService:
    def __init__(self, session_factory, embedder: Embedder, storage_dir: str, chunk_max_chars: int, embedding_dim: int,
                 ocr: OCREngine | None = None, ocr_dpi: int = 200, ocr_min_page_chars: int = 25,
                 ocr_max_pages: int = 0):
        self.sf, self.embedder, self.chunk_max_chars, self.dim = session_factory, embedder, chunk_max_chars, embedding_dim
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.ocr, self.ocr_dpi = ocr, ocr_dpi
        self.ocr_min_page_chars, self.ocr_max_pages = ocr_min_page_chars, ocr_max_pages

    def register(self, pdf: bytes, *, title: str, authors: list[str] | None = None, abstract: str | None = None,
                 year: int | None = None, doi: str | None = None, source_url: str | None = None) -> tuple[str, str | None]:
        """Returns (paper_id, job_id). job_id is None when the paper is already indexed/in progress."""
        if not pdf.startswith(b"%PDF-"):
            raise InvalidUpload("File is not a PDF.")
        digest = hashlib.sha256(pdf).hexdigest()
        with self.sf() as s:
            paper = s.scalar(select(Paper).where(Paper.content_hash == digest))
            if paper and paper.status in ("indexed", "pending", "processing"):
                return str(paper.id), None  # idempotent: no duplicate paper, no new job
            if not paper:
                if doi and s.scalar(select(Paper.id).where(Paper.doi == doi)):
                    raise InvalidUpload("A different paper with this DOI is already registered.")
                key = f"{digest}.pdf"
                (self.storage / key).write_bytes(pdf)
                paper = Paper(title=title, authors=authors or [], abstract=abstract, publication_year=year, doi=doi,
                              source_url=source_url, storage_key=key, content_hash=digest, status="pending")
                s.add(paper)
                s.flush()
            else:  # failed / needs_ocr -> allow retry
                paper.status, paper.error_message = "pending", None
            job = IngestionJob(paper_id=paper.id, status="pending")
            s.add(job)
            s.commit()
            return str(paper.id), str(job.id)

    async def run_job(self, job_id: str) -> None:
        jid = uuid.UUID(job_id)
        with self.sf() as s:
            job = s.get(IngestionJob, jid)
            if not job:
                raise NotFound("Job not found.")
            paper = s.get(Paper, job.paper_id)
            job.status, paper.status = "running", "processing"
            key, paper_id = paper.storage_key, paper.id
            s.commit()
        try:
            pdf = (self.storage / key).read_bytes()
            pages = await asyncio.to_thread(extract_pages, pdf, self.ocr, self.ocr_dpi,
                                            self.ocr_min_page_chars, self.ocr_max_pages)
            total = sum(len(p.text.strip()) for p in pages)
            if not pages or total < MIN_CHARS_PER_PAGE * len(pages):
                raise OCRRequired(
                    "The PDF has no usable text layer, and OCR found no readable text on its pages."
                    if self.ocr is not None else
                    "The PDF has no extractable text; OCR is required.")
            drafts = chunk_pages(pages, self.chunk_max_chars)
            if not drafts:
                raise OCRRequired()
            vectors = await self.embedder.embed_documents([d.text for d in drafts])
            if any(len(v) != self.dim for v in vectors):
                raise EmbeddingDimensionMismatch(
                    f"Embedding model returned vectors of a different size than EMBEDDING_DIM={self.dim}.")
            with self.sf() as s:
                s.execute(delete(Chunk).where(Chunk.paper_id == paper_id))  # re-ingest replaces old chunks
                s.add_all(Chunk(paper_id=paper_id, text=d.text, page_number=d.page_number, section=d.section,
                                chunk_index=d.chunk_index, embedding=v, embedding_model=self.embedder.model,
                                embedding_dim=len(v)) for d, v in zip(drafts, vectors))
                s.get(Paper, paper_id).status = "indexed"
                s.get(IngestionJob, jid).status = "succeeded"
                s.commit()
        except Exception as e:
            known = isinstance(e, AppError)
            if not known:
                log.exception("ingestion failed for job %s", job_id)
            msg = e.message if known else "Unexpected ingestion error."
            with self.sf() as s:
                s.get(IngestionJob, jid).status = "failed"
                s.get(IngestionJob, jid).error_message = msg
                s.get(Paper, paper_id).status = "needs_ocr" if isinstance(e, OCRRequired) else "failed"
                s.get(Paper, paper_id).error_message = msg
                s.commit()
