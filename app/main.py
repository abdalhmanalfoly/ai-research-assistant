import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pathlib import Path

from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import Container
from app.api.routes import chat, health, jobs, papers
from app.core.config import get_settings
from app.core.errors import AppError, DatabaseError
from app.core.logging import setup_logging
from app.db.repositories.pgvector_store import PgVectorStore
from app.db.session import get_session_factory
from app.services.chat_service import ChatService, RagConfig
from app.services.ingestion_service import IngestionService
from app.services.providers.embeddings import HuggingFaceEmbedder
from app.services.providers.ollama import OllamaClient
from app.services.ocr import build_ocr_engine
from app.services.retrieval_service import RetrievalService

log = logging.getLogger(__name__)


def build_embedder(s) -> HuggingFaceEmbedder:
    return HuggingFaceEmbedder(
        s.embedding_model, s.embedding_dim, s.embedding_device, s.embedding_batch_size, s.embedding_max_seq_length,
        s.embedding_normalize, s.embedding_query_instruction, s.embedding_document_prefix,
        s.embedding_cache_dir, s.embedding_offline, dtype=s.embedding_dtype)


def build_container() -> Container:
    s = get_settings()
    sf = get_session_factory()
    client = OllamaClient(s.ollama_base_url, s.llm_model, s.llm_timeout_seconds, s.llm_temperature,
                          s.llm_num_ctx, s.llm_max_output_tokens)  # generation only
    embedder = build_embedder(s)
    ocr = build_ocr_engine(s.ocr_provider, s.ocr_language) if s.ocr_enabled else None
    retrieval = RetrievalService(embedder, PgVectorStore(sf, s.embedding_model), s.embedding_dim)
    chat_service = ChatService(retrieval, client, RagConfig(s.retrieval_threshold, s.retrieval_top_k,
                                                            s.effective_evidence_budget_chars))
    ingestion = IngestionService(sf, embedder, s.storage_dir, s.chunk_max_chars, s.embedding_dim,
                                 ocr=ocr, ocr_dpi=s.ocr_dpi, ocr_min_page_chars=s.ocr_min_page_chars,
                                 ocr_max_pages=s.ocr_max_pages)
    return Container(settings=s, ollama=client, chat=chat_service, ingestion=ingestion,
                     session_factory=sf, embedder=embedder, ocr=ocr)


def _err(status: int, code: str, message: str, **extra):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()  # validates configuration at startup (fails fast on invalid values)
    setup_logging(s.log_level)
    if not hasattr(app.state, "container"):
        app.state.container = build_container()
    emb = getattr(app.state.container, "embedder", None)
    if s.embedding_preload_on_startup and emb is not None:
        await asyncio.to_thread(emb.load)  # intentional, explicit load; failures abort startup with a clear error
    yield
    await app.state.container.ollama.aclose()


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    st = get_settings()
    app = FastAPI(title="AI Research Assistant", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in st.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"])

    @app.exception_handler(AppError)
    async def _app_error(_: Request, e: AppError):
        return _err(e.status_code, e.code, e.message)

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(_: Request, e: SQLAlchemyError):
        log.error("database error: %s", type(e).__name__)
        return _err(DatabaseError.status_code, DatabaseError.code, DatabaseError.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [{"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]} for err in e.errors()]
        return _err(422, "validation_error", "Invalid request.", details=details)  # request body is not echoed

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")
        return _err(500, "internal_error", "An internal error occurred.")

    for r in (health.router, chat.router, papers.router, jobs.router):
        app.include_router(r, prefix="/api")
    if st.serve_frontend and FRONTEND_DIR.is_dir():  # registered LAST so /api/* and /docs always win
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
