"""Application errors. Standard library only, so core logic can use them without web dependencies."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    message = "An internal error occurred."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class LLMUnavailable(AppError):
    status_code, code, message = 503, "llm_unavailable", "The language model service (Ollama) is unreachable."


class ModelNotInstalled(AppError):
    status_code, code, message = 503, "model_not_installed", "The configured model is not installed in Ollama."


class LLMTimeout(AppError):
    status_code, code, message = 504, "llm_timeout", "The language model did not respond in time."


class InvalidLLMResponse(AppError):
    status_code, code, message = 502, "invalid_llm_response", "The language model returned an unusable response."


class EmbeddingUnavailable(AppError):
    status_code, code, message = 503, "embedding_unavailable", "The embedding model is unavailable."


class EmbeddingDimensionMismatch(AppError):
    status_code, code, message = 500, "embedding_dimension_mismatch", "Embedding dimensions do not match the index."


class RetrievalError(AppError):
    status_code, code, message = 503, "retrieval_failed", "Evidence retrieval failed."


class DatabaseError(AppError):
    status_code, code, message = 503, "database_error", "A database error occurred."


class NotFound(AppError):
    status_code, code, message = 404, "not_found", "Resource not found."


class PaperProcessing(AppError):
    status_code, code, message = 409, "paper_processing", "The paper is still being processed."


class PDFExtractionError(AppError):
    status_code, code, message = 422, "pdf_extraction_failed", "The PDF could not be read."


class OCRRequired(AppError):
    status_code, code, message = 422, "ocr_required", "The PDF has no extractable text; OCR is required."


class OCRUnavailable(AppError):
    status_code, code, message = 503, "ocr_unavailable", "No OCR engine is available for scanned pages."


class InvalidUpload(AppError):
    status_code, code, message = 400, "invalid_upload", "Invalid upload."


class InvalidEmbeddingInput(AppError):
    status_code, code, message = 400, "invalid_embedding_input", "Text to embed is empty or invalid."


class InvalidEmbedding(AppError):
    status_code, code, message = 500, "invalid_embedding", "The embedding model produced an invalid vector."
