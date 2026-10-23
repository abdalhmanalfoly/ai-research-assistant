import re
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://research:research@localhost:5432/research"
    # Set true ONLY when DATABASE_URL points at a transaction-mode pooler (port 6543): that mode does not support prepared statements.
    db_disable_prepared_statements: bool = False
    db_pool_size: int = Field(5, ge=1, le=50)
    db_max_overflow: int = Field(5, ge=0, le=50)

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = Field(120, gt=0)
    llm_temperature: float = Field(0.1, ge=0, le=2)
    llm_num_ctx: int = Field(4096, ge=1024)
    llm_max_output_tokens: int = Field(700, ge=16)

    # --- Embeddings (local Hugging Face model; Ollama is NOT used for embeddings) ---
    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = Field(1024, gt=0)  # must equal the vector(N) column created by migration 0003
    embedding_device: str = "auto"  # auto | cpu | cuda | cuda:N
    embedding_dtype: str = "float32"  # float32 | float16 (float16 is used only on CUDA)
    embedding_batch_size: int = Field(16, ge=1, le=256)
    embedding_max_seq_length: int = Field(512, ge=8, le=512)  # BGE-large hard limit is 512 tokens
    embedding_normalize: bool = True  # L2-normalise both documents and queries (same policy for both)
    # Query-only instruction from the BAAI/bge-large-en-v1.5 model card. Passages get no instruction.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_document_prefix: str = ""
    embedding_cache_dir: str | None = None  # default: Hugging Face cache (HF_HOME)
    embedding_offline: bool = False  # True = never contact the Hub (model must already be cached)
    embedding_preload_on_startup: bool = False  # load the model at API start instead of on first use

    retrieval_top_k: int = Field(5, ge=1, le=20)
    retrieval_threshold: float = Field(0.45, ge=0, le=1)  # PLACEHOLDER: calibrate on your corpus
    evidence_budget_chars: int | None = None  # derived from LLM_NUM_CTX when unset
    chunk_max_chars: int = Field(1200, ge=200)

    cors_origins: str = ""  # comma-separated explicit origins; empty = no CORS (same-origin only)
    serve_frontend: bool = True  # serve ./frontend at / (never shadows /api)

    max_pdf_mb: int = Field(50, ge=1)
    storage_dir: str = "./data/storage"

    # --- OCR for scanned PDFs (local; page images never leave the machine) ---
    ocr_enabled: bool = True
    ocr_provider: str = "auto"  # auto | rapidocr | tesseract | none
    ocr_language: str = "eng"  # Tesseract language(s), e.g. eng, ara, eng+ara (RapidOCR uses its bundled model)
    ocr_dpi: int = Field(200, ge=100, le=400)  # rendering resolution for OCR; higher = slower/better
    ocr_min_page_chars: int = Field(25, ge=0, le=1000)  # below this, a page is treated as scanned
    ocr_max_pages: int = Field(0, ge=0)  # 0 = no limit; guards very large scanned documents

    @field_validator("ollama_base_url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("llm_provider")
    @classmethod
    def _provider(cls, v: str) -> str:
        if v != "ollama":
            raise ValueError("only the 'ollama' LLM provider is implemented")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _emb_provider(cls, v: str) -> str:
        if v != "huggingface":
            raise ValueError("only the 'huggingface' embedding provider is implemented")
        return v

    @field_validator("cors_origins")
    @classmethod
    def _cors(cls, v: str) -> str:
        if "*" in v:
            raise ValueError("CORS_ORIGINS must list explicit origins, not *")
        return v

    @field_validator("embedding_device")
    @classmethod
    def _device(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"auto|cpu|cuda(:\d+)?", v):
            raise ValueError("EMBEDDING_DEVICE must be auto, cpu, cuda or cuda:N")
        return v

    @field_validator("embedding_dtype")
    @classmethod
    def _dtype(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("float32", "float16"):
            raise ValueError("EMBEDDING_DTYPE must be float32 or float16")
        return v

    @field_validator("ocr_provider")
    @classmethod
    def _ocr_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("auto", "rapidocr", "tesseract", "none"):
            raise ValueError("OCR_PROVIDER must be auto, rapidocr, tesseract or none")
        return v

    @model_validator(mode="after")
    def _ctx(self):
        if self.embedding_model == "BAAI/bge-large-en-v1.5" and self.embedding_dim != 1024:
            raise ValueError("BAAI/bge-large-en-v1.5 produces 1024-dimensional vectors; set EMBEDDING_DIM=1024")
        if self.llm_num_ctx < self.llm_max_output_tokens + 1000:
            raise ValueError("LLM_NUM_CTX must exceed LLM_MAX_OUTPUT_TOKENS by at least 1000 (room for prompt + evidence)")
        return self

    @property
    def effective_evidence_budget_chars(self) -> int:
        if self.evidence_budget_chars:
            return self.evidence_budget_chars
        # ~3 chars/token is a conservative approximation for English scientific text; not exact.
        return int((self.llm_num_ctx - self.llm_max_output_tokens - 700) * 3)  # 700 tokens: system prompt + question


@lru_cache
def get_settings() -> Settings:
    return Settings()
