# Embeddings (BAAI/bge-large-en-v1.5)

Status labels: ✅ verified (ran) · 🟡 written, not run · ❌ missing. The wrapper logic is tested with a fake model; the real BGE model, the database and CUDA float16 inference were verified on this Windows machine.

## 1. What embeddings are and why this project needs them
An embedding is a list of numbers (here 1024) that represents the *meaning* of a text. Texts with similar meaning get vectors that point in similar directions. The assistant embeds every paper chunk once (at indexing) and every question (at query time), then asks PostgreSQL/pgvector for the chunks whose vectors are closest to the question's vector. Those chunks become the evidence given to Qwen. Without embeddings, retrieval would be keyword matching.

## 2. Why BAAI/bge-large-en-v1.5, why 1024 dimensions
It is a strong open English retrieval embedder (MIT license, model card lists 1024 dimensions, 512-token maximum input, ~335M parameters, ~1.3 GB of fp32 weights). 1024 is simply the size of the model's output layer: every text becomes exactly 1024 floats, so the database column is `vector(1024)`. A different model (different size) cannot be stored in this column. Quality on *your* papers is untested; evaluate it.

## 3. Architecture (what changed)
| Piece | File |
|---|---|
| Embedder (load once, batching, validation, device, dtype, query instruction) | `app/services/providers/embeddings.py` → `HuggingFaceEmbedder` ✅ |
| Settings (`EMBEDDING_*`) | `app/core/config.py` ✅ validators unit-tested only if pydantic is installed |
| Wiring | `app/main.py` (`build_embedder`) |
| Schema | `app/db/models.py`, `migrations/versions/0003_bge_1024.py` 🟡 |
| Search (cosine, model filter, NULL filter) | `app/db/repositories/pgvector_store.py` 🟡 |
| Re-indexing | `app/services/reindex_service.py` ✅ logic, `app/db/repositories/chunk_repo.py` 🟡, `scripts/reindex_embeddings.py` 🟡 |
| Model download / smoke test | `scripts/download_model.py` ✅ |
| Threshold helper | `scripts/calibrate_threshold.py` 🟡 |
Ollama is now used **only** for Qwen (`app/services/providers/ollama.py`). The Ollama embedding code was removed because 768-dim nomic vectors cannot fit the 1024 column. Ingestion (`ingestion_service.py`) and chat (`retrieval_service.py`) call the same `embed_documents` / `embed_query` methods; no route does its own preprocessing.

## 4. Installing and caching the model
```powershell
pip install -r requirements.txt            # sentence-transformers, torch, huggingface-hub (ranges, not pinned)
python -m scripts.download_model           # downloads ~1.3 GB once to the Hugging Face cache, then smoke-tests
```
Default cache: `%USERPROFILE%\.cache\huggingface` (override with `HF_HOME` or `EMBEDDING_CACHE_DIR`). The API does **not** download at startup unless the model is missing at first use; to avoid a slow first request set `EMBEDDING_PRELOAD_ON_STARTUP=true` (loads from cache) or run the script above. After caching, run fully offline with `EMBEDDING_OFFLINE=true` (fails clearly if not cached). `GET /api/health` shows `embedding_model: loaded|cached|not_cached` without loading the model.
Python 3.11–3.13 recommended; PyTorch wheel availability for newer Python versions was not checked.

## 5. CPU and GPU
`EMBEDDING_DEVICE=auto` uses CUDA if PyTorch reports it, otherwise CPU. `cpu` forces CPU. `cuda` / `cuda:0` **fail with a clear error if CUDA is unavailable** (no silent fallback).

### This machine (verified)
- GPU: NVIDIA RTX 2050, **4 GB VRAM**, driver 577.05.
- Qwen: Ollama `qwen2.5:3b` with `LLM_NUM_CTX=4096` runs **100% GPU** (~2.2 GB). Ollama's 32k default context does **not** fit and splits CPU/GPU, so keep `LLM_NUM_CTX=4096` on this GPU.
- Embeddings: BGE-large runs on CUDA in **float16** (`EMBEDDING_DTYPE=float16`) with `EMBEDDING_BATCH_SIZE=4`.
- Measured: BGE first load ~26 s with Qwen resident (37 s cold); batch of 4 × 1300 chars 0.64 s; **0.68 GB allocated / 0.74 GB reserved**. During a cited chat, `nvidia-smi` showed **3143 MiB / 4096 MiB** (Qwen + BGE + CUDA context), 100% GPU. A 73-chunk paper re-indexed successfully while Qwen was resident.

The default PyPI PyTorch wheel on Windows is CPU-only. Install the CUDA build that matches the installed torch version (check `pip show torch`); for torch 2.14.1 this was:
```powershell
pip uninstall -y torch
pip install --no-cache-dir --no-deps torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
The cu126 wheel bundles the CUDA 12.6 runtime and works with the installed 577.05 driver; a separate CUDA toolkit is not required. The wheel is ~2.6 GB, so check free disk first (`pip cache purge` is safe for cached downloads).

Settings for this GPU:
```
EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_BATCH_SIZE=4
```
`float16` is used **only on CUDA**; on CPU it logs a warning and stays float32. The default loader passes `torch_dtype=torch.float16` to sentence-transformers, so the fp32 weights are never moved to the GPU (important on a 4 GB card). Moving to GPU does **not** require re-indexing: same model, same 1024 dimensions; fp16 only changes floating-point rounding, and existing vectors remain valid.

**4 GB VRAM is tight.** Qwen uses ~2.2 GB, leaving ~1.8 GB for BGE-large fp16 (weights ~0.7 GB plus activations). Keep batch 4. For a large PDF batch, stop Qwen first (`ollama stop qwen2.5:3b`) or accept that Ollama may reload it; if you hit CUDA OOM, use batch 2 or run that ingestion on CPU. During chat only one short query is embedded at a time, so BGE's working memory is small. One encode runs at a time per process (a lock protects the shared model). Inference runs in eval mode with no gradient tracking (sentence-transformers `encode`).

## 6. Query instruction (model-specific)
From the BAAI model card: queries are prefixed with `Represent this sentence for searching relevant passages: `; **passages get no instruction**. Implemented once, in `HuggingFaceEmbedder.with_query_instruction` (applied exactly once; text already starting with it is not prefixed again). Configurable via `EMBEDDING_QUERY_INSTRUCTION` / `EMBEDDING_DOCUMENT_PREFIX` (default empty). The card notes the instruction can be omitted with slight quality loss; keep it unless you evaluate otherwise. Unit tests cover once-only application and that documents never receive it ✅ (fake model).

## 7. Normalisation and metric
`EMBEDDING_NORMALIZE=true` L2-normalises **both** stored and query vectors, and the embedder rejects any vector whose norm is not ≈1. Search uses pgvector **cosine distance** (`<=>`, HNSW `vector_cosine_ops`); score = `1 − distance`. For unit vectors cosine and inner product rank identically, and cosine stays correct even if normalisation were switched off, so this combination is consistent. Do not change the normalisation setting without re-indexing.

## 8. Storage in Supabase PostgreSQL
`chunks.embedding vector(1024)`, plus `embedding_model` and `embedding_dim` per chunk. Search filters `embedding IS NOT NULL AND embedding_model = EMBEDDING_MODEL` so vectors from any other model are never compared. See `SUPABASE_INTEGRATION.md §4`.

## 9. Upgrading an existing database (nomic → BGE)
1. **Back up** (Supabase backups or `pg_dump`). Do not run migrations on a live DB without this.
2. `alembic upgrade head`. Migration 0003: if `chunks` is empty it creates the 1024 column; otherwise it **keeps** the old vectors in `embedding_legacy` (+ `embedding_legacy_model`), keeps all chunk ids/text/pages/sections/relations, and sets `embedding`, `embedding_model`, `embedding_dim` to NULL (= pending). Nothing is truncated, padded or reinterpreted. Downgrade restores the 768 layout (BGE vectors are dropped; they are regenerable).
3. `python -m scripts.reindex_embeddings --status` → shows pending count.
4. `python -m scripts.reindex_embeddings` → re-embeds pending chunks in batches (`--batch-size`, default 64; commit per batch). Safe to interrupt and re-run: progress is in the database; only pending chunks are processed. Failed chunks (e.g. empty text, invalid vector) are listed and stay pending; infrastructure errors (model cannot load) stop the run without marking anything. It never creates chunks or re-reads PDFs. (`python -m scripts.import_corpus --reindex` is an alias.)
5. Until re-indexing finishes, questions only see already re-embedded chunks (partial results, never wrong-model results).
6. After verifying retrieval, optionally reclaim space manually: `ALTER TABLE chunks DROP COLUMN embedding_legacy, DROP COLUMN embedding_legacy_model;` (no migration does this).
The repository ships no PDFs. A new, empty database needs your real PDFs: `python -m scripts.import_corpus --dir data\pdfs` (see `PDF_INGESTION_AND_RAG.md`).

## 10. Indexing new PDFs
`python -m scripts.import_corpus --dir data\pdfs` or `POST /api/papers/upload`. Per paper: extract → chunk → embed in batches → store in one transaction; same file again is skipped (SHA-256). Memory: one paper's text and vectors at a time (about 4 KB of vector data per chunk in the database; Python holds more temporarily). Corpora are processed paper by paper, never all at once.

## 11. Verifying
```powershell
curl.exe http://localhost:8000/api/health     # embedding_model, embedding_device, embedding_dtype, embedding_dimension, schema_dimension (both 1024), chunks_pending_reindex (0)
python -m scripts.reindex_embeddings --status
```
```sql
select vector_dims(embedding), embedding_model, embedding_dim, count(*) from chunks where embedding is not null group by 1,2,3;
select count(*) from chunks where embedding is null;
```
Tests: `python -m unittest discover -s tests -t .` (fake model; no downloads). Real-database tests: set `TEST_DATABASE_URL` to a throw-away database whose name contains `test` (they drop its tables) and re-run.

## 12. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `embedding_unavailable`: "not installed" | `pip install -r requirements.txt` |
| "Could not load embedding model" | no internet on first run, proxy, disk full, or `EMBEDDING_OFFLINE=true` without cache → `python -m scripts.download_model` |
| "requests CUDA but no CUDA device" | use `EMBEDDING_DEVICE=cpu/auto` or install CUDA PyTorch |
| Health still says `embedding_device: cpu` after installing CUDA torch | torch is still the CPU wheel → `python -c "import torch; print(torch.cuda.is_available())"`; reinstall the matching `+cuXXX` wheel |
| CUDA out of memory | Qwen (~2.2 GB) + BGE exceed 4 GB → `EMBEDDING_BATCH_SIZE=2`, run `ollama stop qwen2.5:3b` before bulk indexing, or use CPU for that run |
| Warning "float16 is only supported on CUDA" | fp16 was requested but the embedder is on CPU; set `EMBEDDING_DEVICE=cuda` or use float32 |
| Out of memory / very slow | lower `EMBEDDING_BATCH_SIZE`, `CHUNK_MAX_CHARS`; close other apps; use GPU |
| `embedding_dimension_mismatch` | `EMBEDDING_DIM` ≠ 1024, or migration 0003 not applied (`schema_dimension` in health) |
| `invalid_embedding` | NaN/inf or non-normalised output: check torch install / `EMBEDDING_NORMALIZE` |
| Every question → insufficient evidence | nothing re-indexed (`chunks_pending_reindex`), threshold too high for BGE's score range → calibrate |
| Poor answers though chunks exist | chunking/extraction quality, threshold, top_k; inspect `chunks.text`; BGE truncates input at 512 tokens, so keep chunks ≲ 1,200 characters |
| Slow first request | model loading; preload or run `download_model` |
| Search returns fewer rows than `top_k` | selective filters with HNSW; see `hnsw.ef_search` in pgvector docs |

## 13. Retrieval threshold
The old `RETRIEVAL_THRESHOLD=0.45` was a placeholder for nomic-embed-text and is **not validated for BGE**. Write `questions.json` (`[{"question": "...", "on_topic": true|false}]`, ≥ 20 on-topic from your real papers, ≥ 10 off-topic), run `python -m scripts.calibrate_threshold questions.json`, inspect the per-question top scores and choose a value; repeat when the corpus or model changes. Leave the threshold low enough that Qwen can still decide "insufficient evidence" itself rather than silently dropping borderline evidence.
