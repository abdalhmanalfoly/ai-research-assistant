# Frontend (HTML + CSS + vanilla JavaScript)

A web UI for the AI Research Assistant backend. No framework, no bundler, no npm, no build step: ES modules loaded directly by the browser.

## Run it
**Option A (recommended): served by the backend, same origin, no CORS.**
```powershell
uvicorn app.main:app --reload
# open http://localhost:8000/
```
The backend mounts `frontend/` at `/` after all `/api/*` routes (disable with `SERVE_FRONTEND=false`).

**Option B: separate static server (development).**
```powershell
python -m http.server 5500 --directory frontend
# open http://localhost:5500/pages/chat.html
```
Then (1) in the backend `.env` set `CORS_ORIGINS=http://localhost:5500` and restart it, and (2) in the UI go to **Settings → Backend connection**, enter `http://localhost:8000`, Save. (The default API base is `''` = same origin; it is defined in `js/config.js`, and the per-browser override is stored in localStorage. Browsers cannot read `.env` files, so there is no frontend `.env`.)

## Pages and the endpoints they use
| Page | File | Backend endpoints |
|---|---|---|
| Research Chat | `pages/chat.html`, `js/pages/chat.js` | `POST /api/chat`, `GET /api/papers` (scope picker) |
| Research Library | `papers.html` | `GET /api/papers` (server-side `q`, `status`, `year`, paging), `GET /api/papers/{id}` |
| Upload Papers | `upload.html` | `GET /api/health` (upload size limit), `POST /api/papers/upload`, `GET /api/jobs/{id}`, `GET /api/papers/{id}` |
| Processing Jobs | `jobs.html` | `GET /api/jobs` (falls back to jobs started from this browser if the backend predates it), polls only while jobs are active |
| System Status | `status.html` | `GET /api/health` |
| Settings | `settings.html` | none (browser preferences; "Test connection" calls `/api/health`) |
Header health pill: `GET /api/health` every 30 s (the backend never loads the embedding model for this).

## Structure
```
frontend/
├── index.html                 redirect to pages/chat.html
├── pages/*.html               one thin HTML file per page (markup container + module script)
├── css/                       variables (tokens, light/dark) · base · layout · components · pages · responsive
├── js/
│   ├── config.js              API base, timings, limits, safe storage helpers
│   ├── theme-init.js          classic script: sets the theme before first paint
│   ├── api/                   client.js (fetch wrapper, ApiError) + chat/papers/jobs/health modules
│   ├── components/            sidebar (shell + header), modal, toast, citation, theme, icons
│   ├── pages/                 one module per page
│   └── utils/                 dom (safe element builder), format, validation, markdown (safe renderer)
└── tests/                     Node tests (*.test.js), browser.html + browser/run.js
```

## What it does and does not do
* Chat: each question is independent (the backend takes one question, no conversation id). History is kept only in the tab's `sessionStorage` and can be cleared. Filters (papers, years, top-k) are sent to the API. Answers render as safe Markdown; `[E1]`-style markers become buttons only for citations the backend actually returned. Citation cards show title, authors, page (or "page not available"), section, relevance, the retrieved passage (`excerpt`, an additive backend field), DOI/source links. "Insufficient evidence", missing citations, and backend warnings are shown explicitly. Stop button cancels waiting client-side only (the server may continue generating).
* Upload: validates `.pdf`, size (limit read from `/api/health`, default 50 MB) and optional metadata; uses `FormData` (no manual multipart header); indeterminate spinner (no fake progress); duplicate files reported; "Indexed" is shown only when the job succeeded **and** the paper reports `indexed`; automatic OCR is explained, and `needs_ocr` is shown only when OCR found no readable text.
* **Not available because the backend has no endpoint:** delete/reindex/retry/cancel, opening the stored PDF, server-side conversation history, authentication, sorting options.
* Security: no `innerHTML`; model and paper text is inserted as text nodes; only `http(s)` links; CSP meta tag (`script-src 'self'`). The backend still has **no authentication**; do not expose it publicly (see `docs/SECURITY.md`).

## Tests
```powershell
node --test "frontend/tests/*.test.js"      # needs Node 20+ only for TESTING; running the app needs no Node
```
* **Node tests (28, MOCKED fetch/storage; no browser, no backend):** API client (base URL, error envelope, network/timeout/abort, non-JSON, multipart headers), chat/papers/jobs/health normalisation incl. null fields, upload success/failure/duplicate, jobs fallback, health descriptions, PDF/question/metadata validation, Markdown/XSS parsing, formatting, theme resolution.
* **Browser tests (`frontend/tests/browser.html`, 7 checks):** DOM-level safe rendering, citation buttons, citation card, theme toggle, mobile drawer. Open `http://localhost:8000/tests/browser.html` and read the PASS/FAIL list. They were written but **not run** in the build environment (no browser available).
* No test here proves the real backend works end to end; that needs the manual checklist below with the backend, database and Ollama running.

## Manual smoke checklist (real backend)
1. `/pages/status.html` shows Ready (or tells you what is missing). 2. Upload a real PDF; watch the job reach Succeeded and the paper Indexed. 3. Library lists it; details open. 4. Ask a question about it; citations show the right title/page; ask something off-topic and expect "Insufficient evidence". 5. Stop the backend: pill turns Offline, chat shows a retryable error. 6. Resize below 800 px: drawer works. 7. Toggle dark mode.
