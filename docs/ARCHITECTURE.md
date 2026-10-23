# Architecture

**Status labels used in all docs:** ✅ Implemented & verified · 🟡 Implemented, not verified · ❌ Missing · 💡 Recommended

## 1. Audit summary (checked against the source in this repository)

| # | Question | Finding |
|---|---|---|
| 1 | What is implemented? | FastAPI app, 6 endpoints, chunking, evidence selection, prompt building, citation validation, RAG orchestration, Ollama chat client, local Hugging Face embedder (BGE), SQLAlchemy models, pgvector store, Alembic migrations (0001 schema, 0002 RLS), PDF ingestion with a local OCR fallback (RapidOCR/Tesseract), corpus import script. |
| 2 | Implemented but untested? | GPU embedding, the corpus import script, OCR quality on real scans, very large scanned documents (`OCR_MAX_PAGES` guard), and an automated DB/live-Ollama integration suite. The main API → ingestion → retrieval → chat path has been run live. |
| 3 | What is missing? | Authentication, CORS, rate limiting, Supabase Storage, chat history, `collections`/`conversations`/`messages` tables, DB/live-Ollama integration tests, paper delete/update endpoints, automatic recovery of jobs stuck in `processing`. |
| 4 | Which database does it expect? | PostgreSQL ≥ 13 with the `vector` extension (`DATABASE_URL`). Nothing Supabase-specific. |
| 5 | PostgreSQL + pgvector integrated? | ✅ Yes (`app/db/models.py`, `app/db/repositories/pgvector_store.py`, migration `0001_initial.py` with an HNSW cosine index). Verified against Supabase Postgres: migrations, ingestion, and cosine retrieval all ran. |
| 6 | SQLAlchemy / SQL / Supabase client? | **SQLAlchemy 2.0 + psycopg 3.** No raw SQL except in migrations and `health.py`. `grep -ri supabase` over the code finds nothing: no Supabase client library is used. |
| 7 | Where are PDFs stored? | **Local disk**: `STORAGE_DIR` (default `./data/storage`), file name `<sha256>.pdf`, referenced by `papers.storage_key` (`app/services/ingestion_service.py`). Not Supabase Storage. |
| 8 | How are chunks/embeddings persisted and retrieved? | `chunks.embedding` is `vector(1024)` (nullable; migration 0003), `embedding_model`/`embedding_dim` record the producer. NULL = pending re-index. Retrieval uses `Chunk.embedding.cosine_distance(query_vec)`, filtered to `papers.status='indexed'` and `chunks.embedding_model = EMBEDDING_MODEL`; score = `1 - distance`. |
| 9 | Existing endpoints? | See `API_REFERENCE.md`. All under `/api`, **none authenticated**. |
| 10 | Switch to Supabase without a rewrite? | **Yes, for the database**: it's a connection-string change plus two small changes made during this task (migration `0002_enable_rls.py`, `DB_DISABLE_PREPARED_STATEMENTS`/pool settings). **Storage and Auth would need new code.** |

## 2. Recommended architecture

```
Frontend / other app ──HTTPS──▶ FastAPI backend (this repo) ──SQLAlchemy/psycopg──▶ Supabase Postgres (+pgvector)
                                   │  └──HTTP──▶ Ollama (qwen2.5:3b, generation only)  [your machine/server]
                                   └──in-process──▶ sentence-transformers (BAAI/bge-large-en-v1.5, CPU/GPU, local cache)
                                   └──in-process──▶ RapidOCR / Tesseract (local; scanned pages only)
                                   └──(optional, 💡 future)──▶ Supabase Storage (private bucket) / Supabase Auth (JWT verify)
```

* **Primary approach: SQLAlchemy → Supabase PostgreSQL** ("Approach A"). Reasons: the code is already SQLAlchemy; pgvector similarity search, transactions (replace-all-chunks on re-ingest), and Alembic migrations all need a real Postgres connection; the Supabase client talks to PostgREST (HTTP) and is a poor fit for those.
* The Supabase Python client is *not* added. Add it later only if you implement Supabase Storage (💡), and even then plain HTTPS calls would work.
* **`@supabase/server` / `npm install @supabase/server` is NOT needed.** It is a JavaScript/TypeScript server library; this backend is Python. Do not add Node.js to this project. If you later build a JS frontend or edge function, that is where it would belong.
* Ollama runs outside Supabase (Supabase cannot host it). Backend and Ollama must be reachable from each other (`OLLAMA_BASE_URL`).

## 3. Lifecycles

### Initial ingestion (runs once per paper)
1. 🟡 `scripts/import_corpus.py` (or `POST /api/papers/upload`) calls `IngestionService.register()`: checks `%PDF-` magic bytes, computes SHA-256, copies the PDF to `STORAGE_DIR/<hash>.pdf`, inserts `papers` (`status=pending`) and `ingestion_jobs` (`pending`). Same hash already indexed/pending/processing → returns the existing paper, **no new job** (idempotent).
2. 🟡 `run_job()` sets job `running`, paper `processing`; PyMuPDF extracts text per physical page (1-based). Pages with < `OCR_MIN_PAGE_CHARS` characters are rendered at `OCR_DPI` and passed to the local OCR engine.
3. If OCR is disabled/unavailable or finds nothing and average text is < 25 chars/page → paper `needs_ocr`, job `failed`. `OCR_MAX_PAGES` bounds very large scanned documents.
4. `chunk_pages()` normalises text and splits into ≤ `CHUNK_MAX_CHARS` chunks that never cross a page, tagging a heuristic `section`.
5. Embeddings from `BAAI/bge-large-en-v1.5` via sentence-transformers (**no instruction on passages**), batches of `EMBEDDING_BATCH_SIZE`; every vector is validated (1024 floats, finite, L2-normalised) or the job fails.
6. Old chunks of that paper are deleted and new ones inserted in one transaction; paper → `indexed`, job → `succeeded`. Errors → `failed` + `error_message`.

### Every question (`POST /api/chat`)
1. Validate request (✅ validation logic is covered by pure-Python tests; the HTTP layer is 🟡).
2. Embed the question with BGE (query instruction `Represent this sentence for searching relevant passages: ` applied once), validate dimension.
3. pgvector cosine search (only chunks with `embedding_model = EMBEDDING_MODEL` and a non-NULL embedding): top `top_k × 3` candidates (filters: `paper_ids`, `year_min`, `year_max`).
4. Drop chunks below `RETRIEVAL_THRESHOLD`, remove near-duplicates, fit into the evidence budget → ids `E1…En`.
5. No evidence → `insufficient_evidence: true` **without calling the LLM**.
6. Send evidence + question to Qwen via Ollama (`/api/chat`, JSON mode). No retry on timeout.
7. Parse JSON; strip any `[E#]` not supplied; resolve valid ids to paper title/authors/DOI/URL/page from the database rows; return.

✅ = logic verified by unit tests with fakes; the full path was also exercised live against Supabase Postgres + Ollama.

## 4. PDF binary vs. searchable text
* **PDF binary** = the original file, kept for download/re-processing/re-indexing. Belongs in file storage (local disk now; Supabase Storage 💡).
* **Text chunks + vectors** = derived data used for search. Belongs in Postgres/pgvector (`chunks`).
* Losing the PDF does not break search, but you cannot re-index (e.g. after changing embedding model) without it.

## 5. Embedding subsystem (added later)
See `EMBEDDINGS.md`. Embeddings are produced **in-process** by `HuggingFaceEmbedder` (`app/services/providers/embeddings.py`); Ollama is used only for Qwen. Migration `0003_bge_1024.py` moves the schema to `vector(1024)` non-destructively; `scripts/reindex_embeddings.py` fills vectors for existing chunks.

## 6. Web frontend
`frontend/` is plain HTML/CSS/JavaScript (ES modules, no build step, no npm). FastAPI serves it at `/` (mounted after the API routers so `/api/*` and `/docs` always win). It only calls the HTTP API; see `frontend/README.md`.

## 7. Source map
`app/main.py` wiring · `app/api/routes/*` endpoints · `app/services/{chunking,evidence_service,citation_service,chat_service,retrieval_service,ingestion_service,prompts}.py` · `app/services/providers/{ollama,embeddings}.py` · `app/db/{models,session}.py`, `app/db/repositories/pgvector_store.py` · `migrations/versions/` · `scripts/import_corpus.py` · `tests/`.
