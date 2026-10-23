import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import Container, get_container

router = APIRouter()


def _db_check(sf, model: str) -> dict:
    try:
        with sf() as s:
            s.execute(text("SELECT 1"))
            ext = s.scalar(text("SELECT 1 FROM pg_extension WHERE extname='vector'"))
    except Exception:
        return {"database": "down", "pgvector": "unknown", "schema_dimension": None, "chunks_pending_reindex": None}
    out = {"database": "up", "pgvector": "installed" if ext else "missing",
           "schema_dimension": None, "chunks_pending_reindex": None}
    try:  # these need the tables to exist; failures just report "unknown" (None)
        with sf() as s:
            typ = s.scalar(text("select format_type(atttypid, atttypmod) from pg_attribute "
                                "where attrelid = 'public.chunks'::regclass and attname = 'embedding'"))
            if typ and typ.startswith("vector(") and typ.endswith(")"):
                out["schema_dimension"] = int(typ[7:-1])
            out["chunks_pending_reindex"] = s.scalar(
                text("select count(*) from chunks where embedding is null or embedding_model is distinct from :m"),
                {"m": model})
    except Exception:
        pass
    return out


@router.get("/health")
async def health(c: Container = Depends(get_container)):
    s = c.settings
    ollama = await c.ollama.health()  # {"ollama","chat_model"}
    db = await asyncio.to_thread(_db_check, c.session_factory, s.embedding_model)
    emb_state = c.embedder.availability()  # loaded|cached|not_cached|unavailable; never loads the model
    schema_ok = db["schema_dimension"] in (None, s.embedding_dim)
    ok = (ollama["ollama"] == "up" and ollama["chat_model"] == "available" and emb_state in ("loaded", "cached")
          and db["database"] == "up" and db["pgvector"] == "installed" and schema_ok)
    ocr_state = ("disabled" if (not s.ocr_enabled or s.ocr_provider == "none")
                 else (getattr(c.ocr, "name", None) or "unavailable"))
    return {"status": "ok" if ok else "degraded", "api": "up", **ollama,
            "embedding_model": emb_state, "embedding_provider": s.embedding_provider,
            "embedding_dimension": s.embedding_dim,
            "embedding_device": c.embedder.device or s.embedding_device,
            "embedding_dtype": getattr(c.embedder, "dtype", s.embedding_dtype),
            "ocr": ocr_state, "ocr_language": s.ocr_language, "ocr_dpi": s.ocr_dpi, **db,
            "max_pdf_mb": s.max_pdf_mb, "chat_model_name": s.llm_model, "embedding_model_name": s.embedding_model}
