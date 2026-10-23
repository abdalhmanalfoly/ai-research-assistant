# PDF Ingestion, Chunking, Embeddings, Retrieval, Citations

## 1. Loading a pre-existing corpus (no manual upload)
The repository ships **no papers** (`data/pdfs/` is empty and git-ignored for `*.pdf`). Supply your own legally obtained PDFs. Actual, existing command (🟡 never executed):

```powershell
# every *.pdf in a folder; titles default to the file name
python -m scripts.import_corpus --dir data\pdfs
# with real metadata
python -m scripts.import_corpus --manifest corpus.json
# re-embed EXISTING chunks (e.g. after the nomic -> BGE switch); resumable, no PDF re-extraction
python -m scripts.reindex_embeddings        # same as: python -m scripts.import_corpus --reindex
python -m scripts.reindex_embeddings --status   # counts only
```
Manifest (JSON list; only `path` and `title` required; `path` is relative to the manifest file):
```json
[{"path":"pdfs/paper1.pdf","title":"Exact title","authors":["A. Author"],"year":2021,"doi":"10.xxxx/xxxx","source_url":"https://...","abstract":"..."}]
```
Requires the BGE model (`python -m scripts.download_model` once); embeddings are computed locally, so Ollama is needed only for chat, not for ingestion. The script prints one line per file (`queued` or `already registered (skipped)`), then processes jobs **sequentially** in the same process. Re-running is safe (SHA-256 idempotency). ❌ Missing: parallelism, progress bar, resume of half-finished runs (re-run instead), metadata extraction from PDF (titles come from file names unless a manifest is used; fix wrong titles in the DB or re-import via manifest on a fresh DB).

## 2. Stages and where to look
| Stage | Code | Notes |
|---|---|---|
| Register | `IngestionService.register` | magic-byte check, SHA-256, copy to `STORAGE_DIR`, insert rows |
| Extract | `extract_pages` (PyMuPDF `page.get_text("text")`) | `page_number` = physical page index starting at 1; printed page labels (e.g. roman numerals) are **not** used |
| Scanned check + OCR | `extract_pages` + `run_job` | pages with < `OCR_MIN_PAGE_CHARS` text are rendered at `OCR_DPI` and OCR'd locally; avg < 25 chars/page **after** OCR ⇒ `needs_ocr` |
| Normalise | `chunking.normalize_text` | rejoins `word-\nbreaks`, collapses whitespace, keeps Unicode/math; no NFKC folding |
| Chunk | `chunking.chunk_pages` | ≤ `CHUNK_MAX_CHARS` (1200), paragraph-first, sentence fallback; never crosses a page; heading names (Abstract, Methods, Results, …) set `section` |
| Embed | `HuggingFaceEmbedder` (BAAI/bge-large-en-v1.5) | Batches of `EMBEDDING_BATCH_SIZE`; **no instruction on passages** (model card); every vector validated (1024 floats, finite, L2-normalised). Runs locally, not via Ollama |
| Store | `IngestionService.run_job` | one transaction: delete old chunks, insert new, mark indexed |

Known extraction limits: two-column layouts may interleave text; tables, equations and figure captions extract poorly; footnotes/headers repeat on each page; section detection only recognises common heading names. OCR does not fix layout problems (it reads the rendered page in visual order), so check OCR'd papers too. Check a few real papers by reading `chunks.text` (SQL below).

### OCR settings (scanned PDFs)
`OCR_ENABLED=true` by default. `OCR_PROVIDER=auto` picks `rapidocr-onnxruntime` (models ship with the package, CPU, no external binary) and falls back to Tesseract if the executable is installed. Pages with a real text layer are never OCR'd. `OCR_DPI` (default 200) trades speed for accuracy; `OCR_MIN_PAGE_CHARS` (25) defines "scanned"; `OCR_MAX_PAGES` (0 = unlimited) bounds very large documents. OCR is CPU-bound: expect roughly 5–30 s/page at 200 DPI. `OCR_LANGUAGE` applies to Tesseract only (`eng`, `ara`, `eng+ara`, …); RapidOCR's default model handles English and Chinese. OCR output is line-based; `normalize_text` rejoins hyphenated line breaks.

## 3. Retrieval and answer generation
`RetrievalService` embeds the question with BGE (query instruction `Represent this sentence for searching relevant passages: ` applied once, in `HuggingFaceEmbedder` only) and runs `ORDER BY embedding <=> query LIMIT top_k*3`. `select_evidence` then applies the threshold, removes near-duplicates (token Jaccard ≥ 0.85), and fits a character budget (`(LLM_NUM_CTX − LLM_MAX_OUTPUT_TOKENS − 700) × 3`, an approximation). Qwen receives a system prompt (answer only from evidence; cite `[E#]`; evidence is untrusted data; say when insufficient) and JSON-mode output `{"answer","insufficient_evidence"}`.

**Threshold calibration is required, and the old value is NOT validated for BGE.** `RETRIEVAL_THRESHOLD=0.45` was a placeholder chosen for nomic-embed-text; BGE's cosine-score range is different (BAAI's own card warns that a score above 0.5 does not by itself mean two texts are similar). Run `python -m scripts.calibrate_threshold questions.json` (see its docstring) and review the output by hand. Procedure: write ~30 questions whose answers you know plus ~10 off-topic ones; for each, record the top cosine score (`relevance_score` in citations, or run the SQL below); choose a value that separates them; re-check after changing corpus, chunk size, or model.

## 4. Citation guarantees (and limits)
* The app, not the model, owns ids `E1..En`. Unknown ids are removed from the text and `invalid_citations_removed` is added to `warnings`.
* Title/authors/DOI/URL/page come from database rows, never from model text. Unknown page ⇒ `null`.
* ✅ These behaviours are unit-tested with fakes and were exercised live with Qwen 2.5 3B; citation *support* is still not verified (see the limit below).
* **Limit:** a valid id means "this chunk was in the prompt", **not** "this chunk supports the sentence". Small models like Qwen 2.5 3B can mis-attribute. If an answer has no valid citation the API still returns it with `no_valid_citations`; show that to users or hide such answers.

## 5. Useful SQL (SQL Editor or psql)
```sql
select status, count(*) from papers group by status;
select p.title, count(c.*) chunks, min(c.page_number) first_page, max(c.page_number) last_page
  from papers p left join chunks c on c.paper_id=p.id group by p.id;
select page_number, section, left(text,200) from chunks where paper_id='PASTE-UUID' order by chunk_index limit 10;
select embedding_model, embedding_dim, count(*) from chunks group by 1,2;   -- after re-index: one row (BAAI/bge-large-en-v1.5, 1024); NULLs = still pending
select count(*) from chunks where embedding is null or embedding_model is distinct from 'BAAI/bge-large-en-v1.5';   -- pending re-index
select vector_dims(embedding), count(*) from chunks where embedding is not null group by 1;   -- must be 1024
select id, status, error_message from ingestion_jobs where status='failed';
```
Stuck jobs: if the process dies mid-job, rows stay `running`/`processing` forever (❌ no recovery). Fix manually: `update papers set status='failed' where status='processing';` then re-run the import (failed papers get a new job).
