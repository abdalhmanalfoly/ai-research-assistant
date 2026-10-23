# API Reference (verified against the route source code)

Base URL (local): `http://localhost:8000` · all endpoints are under `/api` · interactive docs: `/docs` (Swagger) · JSON only (except upload = multipart).
**Authentication: none (❌ not implemented).** Anyone who can reach the server can call every endpoint, including upload. Do not expose it publicly before reading `SECURITY.md`.
Status: all routes 🟡 (written; the route tests exist but were never run).

## Endpoints

| Method | Path | Purpose | DB operations |
|---|---|---|---|
| GET | `/api/health` | Ollama, model and database status | `SELECT 1`; check `pg_extension` for `vector` |
| POST | `/api/chat` | Ask a question (RAG) | vector search over `chunks` ⨝ `papers` (indexed only); `count(chunks)` when nothing is found |
| GET | `/api/papers` | List papers | `SELECT count`, paged `SELECT papers` |
| GET | `/api/papers/{paper_id}` | One paper's metadata | `SELECT papers` by PK |
| POST | `/api/papers/upload` | Upload a PDF → background ingestion | insert `papers` + `ingestion_jobs`; later insert `chunks` |
| GET | `/api/jobs` | List jobs (newest first; added for the web UI) | `SELECT count`, paged join `ingestion_jobs` ⨝ `papers` |
| GET | `/api/jobs/{job_id}` | Ingestion job status | `SELECT ingestion_jobs` ⨝ `papers` |

### GET /api/health → 200 always
```json
{"status":"ok|degraded","api":"up","max_pdf_mb":50,"ollama":"up|unreachable|error","chat_model":"available|missing|unknown",
 "embedding_model":"loaded|cached|not_cached|unavailable","embedding_provider":"huggingface","embedding_dimension":1024,
 "embedding_device":"auto|cpu|cuda","embedding_dtype":"float32|float16","ocr":"rapidocr|tesseract|unavailable|disabled","ocr_language":"eng","ocr_dpi":200,
 "schema_dimension":1024,"chunks_pending_reindex":0,"database":"up|down","pgvector":"installed|missing|unknown",
 "chat_model_name":"qwen2.5:3b","embedding_model_name":"BAAI/bge-large-en-v1.5"}
```
`status` is `ok` only if everything is up/available/installed and `schema_dimension` equals `embedding_dimension`. `embedding_model` reports whether the local model is `loaded` in memory, `cached` on disk, `not_cached` (will download on first use) or `unavailable` (sentence-transformers/huggingface-hub missing); it **never loads the model**. `embedding_device` shows the selected device once the model is loaded; `embedding_dtype` is the configured precision (`float16` is applied only on CUDA). `chat_model` is checked in Ollama; Ollama is no longer involved in embeddings. `ocr` is informational (it does **not** affect `status`): the configured backend, `unavailable` when OCR is enabled but no backend can be found, or `disabled` when `OCR_ENABLED=false`. `schema_dimension`/`chunks_pending_reindex` are `null` when the tables do not exist yet. Changed in the BGE update: `embedding_model` values (was `available|missing|unknown`) and new fields; all previous keys are still present. No secrets or exception details are exposed.

### POST /api/chat
Request: `question` (string, trimmed, 1–2000 chars, required); `paper_ids` (list of UUID strings, ≤100, optional); `top_k` (1–20, optional, default `RETRIEVAL_TOP_K`); `year_min`, `year_max` (1000–3000, optional, `year_min ≤ year_max`).
Response 200:
```json
{"answer":"...","citations":[{"id":"E1","paper_id":"<uuid>","title":"<stored title>","authors":["..."],
  "page_number":7,"section":"Results","doi":null,"source_url":null,"relevance_score":0.6123,"excerpt":"first ≤600 chars of the retrieved passage or null"}],
 "retrieved_chunks":3,"model":"qwen2.5:3b","insufficient_evidence":false,"warnings":[]}
```
`page_number` is `null` when unknown. `warnings` may contain `invalid_citations_removed`, `no_valid_citations`, `evidence_truncated_to_fit_context`. `insufficient_evidence: true` with `citations: []` is returned (HTTP 200) when the corpus is empty, nothing passes the threshold, or the model says the evidence is insufficient. Filtering by a paper that is not `indexed` simply yields insufficient evidence (the `paper_processing` error exists in code but is never raised).

### GET /api/papers
Query: `limit` (1–100, default 20), `offset` (≥0), `year`, `status` (`pending|processing|indexed|failed|needs_ocr`), `q` (case-insensitive title substring). Response: `{"items":[Paper],"total":n,"limit":20,"offset":0}`; `Paper` = `id,title,authors,abstract,publication_year,doi,source_url,status,error_message,created_at,updated_at`.

### GET /api/papers/{paper_id}
Response `Paper`. 404 `not_found`; 422 if not a UUID.

### POST /api/papers/upload (multipart/form-data) → 202
Fields: `file` (required, `.pdf`, ≤ `MAX_PDF_MB`, must start with `%PDF-`), optional `title` (defaults to file name), `authors` (comma-separated), `year`, `doi`, `source_url`.
Response: `{"paper_id":"<uuid>","job_id":"<uuid>|null","already_registered":false}`. `already_registered: true` (and `job_id: null`) means the identical file is already indexed/pending/processing. A previously `failed`/`needs_ocr` file gets a new job. Errors: 400 `invalid_upload` (not PDF, too big, DOI belongs to another paper).
Note: the size check reads the file into memory (up to the limit + 1 byte) before rejecting.

### GET /api/jobs (new)
Query: `limit` (1–100, default 20), `offset`, `status` (`pending|running|succeeded|failed`), `paper_id` (UUID). Response `{"items":[Job],"total","limit","offset"}`. 🟡 unverified (FastAPI not installable in the build sandbox).

### GET /api/jobs/{job_id}
`{"id","paper_id","paper_title","status":"pending|running|succeeded|failed","error_message","created_at","updated_at"}`; 404 `not_found`.

## Error format
```json
{"error":{"code":"llm_unavailable","message":"The language model service (Ollama) is unreachable."}}
```
| HTTP | code | Meaning |
|---|---|---|
| 422 | `validation_error` (+`details:[{field,message}]`) | Bad payload; the submitted body is not echoed |
| 400 | `invalid_upload` | Bad upload |
| 404 | `not_found` | Unknown paper/job |
| 409 | `paper_processing` | Defined, not currently raised |
| 422 | `pdf_extraction_failed`, `ocr_required` | Recorded on the job/paper, not returned by upload (ingestion is async). `ocr_required` = no usable text even after OCR, or more scanned pages than `OCR_MAX_PAGES` |
| 502 | `invalid_llm_response` | Model output not usable JSON |
| 503 | `llm_unavailable`, `model_not_installed`, `embedding_unavailable` (model cannot load/run, CUDA missing, packages missing), `ocr_unavailable` (configured backend cannot start/import), `retrieval_failed`, `database_error` | Dependency down/missing |
| 504 | `llm_timeout` | Generation exceeded `LLM_TIMEOUT_SECONDS` (not retried) |
| 500 | `embedding_dimension_mismatch`, `invalid_embedding` (NaN/inf/not normalised), `internal_error` | Config mismatch / bug; stack traces are logged server-side only |

## curl examples (Windows: use `curl.exe`, not the PowerShell alias `curl`)
```powershell
curl.exe http://localhost:8000/api/health
curl.exe "http://localhost:8000/api/papers?limit=5&status=indexed"
curl.exe http://localhost:8000/api/papers/PASTE-PAPER-UUID
curl.exe -X POST http://localhost:8000/api/papers/upload -F "file=@C:\path\paper.pdf;type=application/pdf" -F "title=My paper" -F "year=2022"
curl.exe http://localhost:8000/api/jobs/PASTE-JOB-UUID
curl.exe -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"What limitations do the authors report?\",\"top_k\":5}"
```
PowerShell alternative: `Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/chat -ContentType 'application/json' -Body '{"question":"..."}'`.

## Calling from a frontend or another service
* Put the backend base URL in the caller's config (e.g. `VITE_API_BASE_URL=https://api.example.com`); never put database or Supabase secret keys in the frontend.
* **Same-origin by default:** the backend serves the web UI from `frontend/` at `/` (`SERVE_FRONTEND=true`), so no CORS is needed. **CORS is opt-in:** set `CORS_ORIGINS=http://localhost:5500` (comma-separated explicit origins; `*` is rejected) when the frontend is served from another origin. Only GET/POST and the `Content-Type` header are allowed; there are no credentials/cookies.
* Treat `422` as a user-input problem (show `details`), `503/504` as retryable-later, and `insufficient_evidence: true` as a valid answer, not an error.

## ❌ Not Implemented Yet (proposed design)
| Endpoint | Purpose |
|---|---|
| `DELETE /api/papers/{id}` | Remove paper + chunks (cascade) + stored PDF |
| `GET /api/papers/{id}/file` | Download/view the stored PDF (the web UI has no PDF link without it) |
| `POST /api/jobs/{id}/retry`, `.../cancel` | Job actions (the web UI shows none because they do not exist) |
| `POST /api/papers/{id}/reindex` | Re-embed one paper (today: CLI `scripts.reindex_embeddings`, whole corpus only) |
| `GET /api/papers/{id}/chunks` (admin) | Inspect extracted chunks/pages |
| `POST /api/conversations`, `GET /api/conversations/{id}` | Chat history (needs new tables) |
| Auth on all of the above | See `SECURITY.md` |
