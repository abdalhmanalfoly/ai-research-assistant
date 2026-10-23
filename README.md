# AI Research Assistant — Backend

FastAPI + PostgreSQL/pgvector + Ollama (`qwen2.5:3b`, generation only) + local Hugging Face embeddings (`BAAI/bge-large-en-v1.5`, 1024 dims, via sentence-transformers). Answers questions about an indexed corpus of research PDFs, with application-validated citations (paper title, DOI/URL, real PDF page number). Scanned PDFs are handled by a local OCR fallback (RapidOCR); no page images leave the machine.

## Verification status (read first)

| Area | Status |
|---|---|
| Chunking, evidence selection, prompt building, citation validation, RAG orchestration | **Unit-tested, passing** (stdlib `unittest`, synthetic fixtures, fake embedder/LLM/in-memory store) |
| Embedder wrapper (fake model: dims, NaN/inf, order, query instruction, device, dtype, errors), resumable re-index logic, API/Ollama clients | **Unit-tested, passing** (73 passed, 6 integration skipped — see Tests) |
| Real BGE model download/inference, CPU and CUDA fp16 | **Live-tested** (`BAAI/bge-large-en-v1.5`, 1024 dims; CPU first load ~79 s; CUDA on RTX 2050 4 GB: first load ~26 s, batch 4×1300 chars 0.64 s, 0.68 GB allocated) |
| Migration 0003, pgvector search, SQL re-index | **Live-tested against Supabase Postgres** |
| FastAPI routes, error envelope | **Live-tested** via the web UI and `curl` |
| PostgreSQL / pgvector store, models, Alembic migration, ingestion pipeline, PyMuPDF extraction | **Live-tested end-to-end** (multiple PDFs indexed; jobs reached `succeeded`) |
| Live Qwen 2.5 3B via Ollama | **Live-tested** (100% GPU at 4096 context; cited answers ~6–10 s) |
| Qwen + BGE on one 4 GB GPU | **Live-tested**: Qwen 2.2 GB + BGE fp16 0.74 GB ≈ 3.1/4 GB; 73-chunk re-index and cited chat succeeded |
| Real research papers | **Live-tested**: two papers uploaded and indexed; cited answers verified on one |
| OCR fallback (`rapidocr-onnxruntime`) | **Live-tested** on a synthetic scanned page (~27 s/page on CPU incl. model load); real scans not evaluated |

Supabase is supported as a plain PostgreSQL host via `DATABASE_URL`; no Supabase client library, Auth, or Storage is used yet (see docs). Migrations: `0001` schema, `0002` RLS, `0003` vector(1024) for BGE.

Treat anything not marked as passing/live-tested as a first draft that still needs its first real run.

## Documentation
| Doc | Contents |
|---|---|
| [frontend/README.md](frontend/README.md) | Web UI (HTML/CSS/vanilla JS): run, structure, tests, limits |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Audit of what exists, architecture, data flow |
| [docs/SUPABASE_INTEGRATION.md](docs/SUPABASE_INTEGRATION.md) | Supabase database setup, connection strings, schema, storage plan |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Verified endpoints, errors, curl examples |
| [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) | Windows step-by-step setup and troubleshooting |
| [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) | BGE embeddings, query instruction, GPU/CPU, re-indexing, verification |
| [docs/PDF_INGESTION_AND_RAG.md](docs/PDF_INGESTION_AND_RAG.md) | Corpus import, chunking, retrieval, citations |
| [docs/SECURITY.md](docs/SECURITY.md) | Secrets, RLS, auth design, production checklist |

## Web UI
Start the backend (`uvicorn app.main:app --reload`) and open **http://localhost:8000/**. Pages: Research Chat, Library, Upload, Jobs, System Status, Settings. No Node/npm is needed to run it. See `frontend/README.md`. Tested so far: 28 Node unit tests with mocked fetch (passing); DOM tests in `frontend/tests/browser.html` and all real-backend behaviour are **not yet run**.

## Setup

```bash
# 1. Ollama (https://ollama.com/download), then:
ollama pull qwen2.5:3b
python -m scripts.download_model   # after pip install: downloads BAAI/bge-large-en-v1.5 (~1.3 GB) once
ollama list                      # both should appear

# 2. Database
docker compose up -d db          # pgvector/pgvector:pg16

# 3. Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 4. Schema
alembic upgrade head

# 5. Run
uvicorn app.main:app --reload    # docs: http://localhost:8000/docs
curl localhost:8000/api/health   # reports ollama up/unreachable, model available/missing, db status
```

## Importing the corpus

```bash
python -m scripts.import_corpus --dir data/pdfs                 # titles default to file names
python -m scripts.import_corpus --manifest corpus.json          # real metadata (see script docstring)
```
Re-running is safe: PDFs are keyed by SHA-256, so already-indexed files are skipped. Scanned pages are OCR'd automatically (RapidOCR, local; see `OCR_*` settings); a paper is marked `needs_ocr` only when OCR still finds no readable text. Upload one-off PDFs with `POST /api/papers/upload` and poll `GET /api/jobs/{id}`.

## Chat

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"question":"What limitations do the authors report?","paper_ids":[],"top_k":5}'
```
Response shape: `answer, citations[{id, paper_id, title, authors, page_number|null, section, doi, source_url, relevance_score}], retrieved_chunks, model, insufficient_evidence, warnings`.
`warnings` may contain `invalid_citations_removed`, `no_valid_citations`, `evidence_truncated_to_fit_context`.

## How citations work
Retrieved chunks get ids `E1..En` assigned by the app; the model must cite them as `[E1]`. Unknown ids are stripped; valid ids are resolved from the database. Unknown pages return `null`. **A valid id only proves that chunk was supplied to the model, not that it supports the sentence**; claim-to-evidence alignment needs separate evaluation.

## Embedding model changes
Embeddings are BGE (1024-dim) computed locally; see `docs/EMBEDDINGS.md`. Upgrading from the earlier nomic version: **back up**, `alembic upgrade head` (migration 0003 keeps old vectors in `embedding_legacy`), then `python -m scripts.reindex_embeddings`. Switching to another model later needs a new migration (the vector size is fixed in the schema), `EMBEDDING_DIM`, and the model's own query-instruction rules.

## Calibrating `RETRIEVAL_THRESHOLD`
0.45 is an uncalibrated placeholder (originally for nomic-embed-text; NOT validated for BGE). Collect ~30 questions with known relevant papers plus ~10 off-topic ones; log top scores for each; pick a threshold separating them; revisit when the corpus or embedding model changes.

## Tests
```bash
python -m unittest discover -s tests/unit -t .     # works with stdlib only; framework tests skip if deps missing
pytest                                             # after pip install -r requirements.txt
```
`tests/fixtures/fakes.py` is test-only synthetic data. `tests/integration/test_ocr_live.py` is opt-in (`RUN_OCR_TESTS=1`); DB and live-Ollama integration tests still need to be written.

## Known limitations
- OCR is CPU-bound (~5-30 s/page depending on DPI and page density); low-quality or handwritten scans may still fail.
- On a 4 GB GPU, Qwen (~2.2 GB at 4096 context) and BGE fp16 share VRAM; for bulk ingestion use `EMBEDDING_BATCH_SIZE=4` or lower, or `ollama stop qwen2.5:3b` first.
- No conversation history; no authentication; no rate limiting.
- Ingestion runs in-process via FastAPI BackgroundTasks (lost on restart; stuck `processing` rows are not auto-recovered).
- Section detection is heuristic (common heading names only).
- Evidence budget uses ~3 chars/token, an approximation.
- Qwen 2.5 3B may ignore the JSON/citation format; the service handles this (parse failure -> 502, missing citations -> warning) but quality needs evaluation on real papers.
- No retrieval-quality or claim-alignment evaluation harness yet.

## Integrating into the team repo
1. Copy `app/`, `migrations/`, `scripts/`, `alembic.ini`, `requirements.txt`, `.env.example`.
2. Merge dependencies; set `DATABASE_URL` to a Postgres with the `vector` extension.
3. Mount via `from app.main import create_app` or include the routers under `/api`.
4. `alembic upgrade head`, import the corpus, hit `/api/health` until `status: ok`.
