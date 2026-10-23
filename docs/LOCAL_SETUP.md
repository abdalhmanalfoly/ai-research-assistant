# Local Setup on Windows (PowerShell)

No command in this guide was executed against Supabase or Ollama by the author (no network/Windows machine was available). Steps are derived from the repository files; treat the first run as the real test.
Prerequisites: Windows 10/11, ~15 GB free disk with CUDA PyTorch (~2.6 GB wheel plus installed runtime; CPU-only PyTorch needs much less), Qwen ~2 GB, BGE ~1.3 GB, 16 GB RAM recommended, internet access.

## 1. Install tools
* Python 3.11 or newer from python.org (tick **Add python.exe to PATH**). Check: `python --version`
* Ollama for Windows from ollama.com/download. Check: `ollama --version`
* Optional: Docker Desktop (only for a local Postgres test database); Git.

## 2. Extract and open
Extract `ai-research-assistant.zip`, then:
```powershell
cd C:\path\to\ai-research-assistant
```

## 3. Virtual environment and dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (then re-open PowerShell)
python -m pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` uses version ranges that were never installed together by the author. If `pip` reports a conflict, note the message; after a successful install run `pip freeze > requirements.lock.txt` to pin it.

## 4. Configure `.env`
```powershell
Copy-Item .env.example .env
notepad .env
```
Set `DATABASE_URL` (see `SUPABASE_INTEGRATION.md §3`; start with the **session pooler** string from Supabase → Connect if you are unsure about IPv6). Leave the other defaults. Never share `.env`.

## 5. Supabase dashboard (once)
Enable the `vector` extension; reset/store the DB password; copy the connection string (details in `SUPABASE_INTEGRATION.md §2`).

## 6. Ollama models
```powershell
ollama serve            # skip if the Ollama tray app is already running
ollama pull qwen2.5:3b          # ~2 GB
# Embeddings no longer use Ollama (see step 6b)
ollama list
```
Sizes are approximate.

## 6b. Embedding model (BAAI/bge-large-en-v1.5, runs locally)
```powershell
python -m scripts.download_model      # first run downloads ~1.3 GB to the Hugging Face cache, then embeds a test text
python -m scripts.download_model --offline   # later: verifies it works without internet
```
CPU is fine (slower). On an NVIDIA GPU, install the CUDA PyTorch wheel, set `EMBEDDING_DEVICE=cuda` and `EMBEDDING_DTYPE=float16`; details and 4 GB-VRAM guidance: `EMBEDDINGS.md §5`.

## 7. Create the schema
```powershell
alembic upgrade head
alembic current         # expect 0003 (head)
```
Common failure: wrong URL prefix (must start with `postgresql+psycopg://`), unencoded password characters, or IPv6-only direct host on an IPv4 network (use the session pooler).

## 8. Start the API and check health
```powershell
uvicorn app.main:app --reload
# second PowerShell window:
curl.exe http://localhost:8000/api/health
```
Target: `"status":"ok"`. Open `http://localhost:8000/` for the web UI (Research Chat, Library, Upload, Jobs, System Status) or `http://localhost:8000/docs` for the raw API.

## 9. Import PDFs and verify
Put your own PDFs in `data\pdfs\` (the repo has none), then:
```powershell
python -m scripts.import_corpus --dir data\pdfs
# if you already had chunks from the old nomic model: python -m scripts.reindex_embeddings
curl.exe "http://localhost:8000/api/papers?status=indexed"
```
`failed` or `needs_ocr` papers show an `error_message`.

## 10. Ask a question and check citations
```powershell
curl.exe -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"What is the main contribution of the papers?\"}"
```
Verify: every `citations[].title` and `page_number` matches a real paper in your corpus (open the PDF at that page); ask an off-topic question and confirm `insufficient_evidence: true`; `warnings` should be empty. The first request is slow (model load).

## 11. Tests
```powershell
python -m unittest discover -s tests/unit -t .     # 69 unit tests pass with dependencies installed (~1 s)
python -m unittest discover -s tests -t .          # 75 tests, 6 skipped (DB / opt-in live tests)
pytest
```
| Kind | Where | Needs |
|---|---|---|
| Unit (pure logic, fakes) | `tests/unit/test_core.py` | nothing |
| OCR fallback (synthetic PDFs + fake engine) | `tests/unit/test_ocr.py` | PyMuPDF (in `requirements.txt`) |
| Mocked Ollama / API | `tests/unit/test_ollama_client.py`, `test_api.py` | `pip install -r requirements.txt` |
| Integration with DB / live Ollama | ❌ not written | Postgres, Ollama |
| Live OCR (opt-in) | `tests/integration/test_ocr_live.py` | `rapidocr-onnxruntime`; set `RUN_OCR_TESTS=1` |

## 12. Troubleshooting
| Symptom | Likely cause | Check / fix |
|---|---|---|
| `status: degraded`, `ollama: unreachable` | Ollama not running / wrong `OLLAMA_BASE_URL` | `ollama list`; `curl.exe http://localhost:11434/api/tags` |
| `chat_model: missing` or error `model_not_installed` | model not pulled | `ollama pull qwen2.5:3b` |
| `embedding_unavailable` | BGE cannot load: packages missing, no internet on first run, offline mode without cache, CUDA requested but absent | `pip install -r requirements.txt`; `python -m scripts.download_model`; set `EMBEDDING_DEVICE=cpu` |
| `database: down` | bad `DATABASE_URL`, firewall, IPv6-only host | test with `psql "<url>"`; try session pooler; check `sslmode=require` |
| `pgvector: missing` / `type "vector" does not exist` | extension not enabled | enable in Database → Extensions; re-run `alembic upgrade head` |
| `prepared statement ... does not exist` | transaction pooler (6543) with prepared statements | set `DB_DISABLE_PREPARED_STATEMENTS=true` or use session pooler |
| `password authentication failed` | wrong/old password, special chars not encoded | reset DB password; percent-encode |
| Pydantic error at startup | invalid setting (e.g. `LLM_NUM_CTX` too small) | read the message; compare to `.env.example` |
| `embedding_dimension_mismatch` | `EMBEDDING_DIM` ≠ model output or DB column | BGE-large = 1024; check `/api/health` `schema_dimension`; run `alembic upgrade head` |
| Paper `needs_ocr` | scanned PDF and OCR found no readable text, or OCR disabled/unavailable | check `/api/health` → `ocr`; `pip install -r requirements.txt`; try `OCR_DPI=300`; OCR externally (e.g. OCRmyPDF) as a last resort |
| OCR is very slow | CPU inference at high DPI / many scanned pages | lower `OCR_DPI` (150), split the document, or raise `OCR_MAX_PAGES`; expect roughly 5–30 s/page |
| `insufficient_evidence` for everything | threshold too high, nothing indexed, or filters too narrow | `select status,count(*) from papers group by 1;` lower `RETRIEVAL_THRESHOLD` after calibration |
| `llm_timeout` | CPU-only generation is slow | raise `LLM_TIMEOUT_SECONDS`, lower `LLM_NUM_CTX`/`top_k` |
| Duplicate import does nothing | same SHA-256 already registered | expected (idempotent) |
| Browser says CORS error | CORS not configured | see `SECURITY.md` |
| `ModuleNotFoundError: scripts` | run from the wrong folder | `cd` to the project root, run with `python -m` |
