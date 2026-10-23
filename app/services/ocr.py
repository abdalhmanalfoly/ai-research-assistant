"""OCR fallback for scanned PDF pages.

Text-layer pages are always handled by PyMuPDF; OCR is only invoked for pages whose
extracted text is shorter than OCR_MIN_PAGE_CHARS. Backends are optional and imported
lazily so the API can start (and text PDFs can be ingested) without them:

  * rapidocr  -> rapidocr-onnxruntime (bundled models, CPU, no external binary) [default]
  * tesseract -> pytesseract + Pillow + the Tesseract executable
  * none      -> OCR disabled (scanned pages end up as `needs_ocr`)

OCR runs locally; page images never leave the machine.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import threading
from typing import Protocol, runtime_checkable

from app.core.errors import OCRUnavailable

log = logging.getLogger(__name__)


@runtime_checkable
class OCREngine(Protocol):
    name: str

    def image_to_text(self, image_bytes: bytes) -> str: ...


class RapidOCREngine:
    """rapidocr-onnxruntime. Models are bundled with the package; first use loads them (~1-2 s)."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()

    def _load(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    try:
                        from rapidocr_onnxruntime import RapidOCR
                    except ImportError:
                        raise OCRUnavailable(
                            "OCR backend 'rapidocr-onnxruntime' is not installed. Run: pip install -r requirements.txt"
                        ) from None
                    try:
                        self._engine = RapidOCR()
                        log.info("RapidOCR engine loaded")
                    except Exception:
                        log.exception("could not initialise RapidOCR")
                        raise OCRUnavailable("The RapidOCR engine could not be initialised.") from None
        return self._engine

    def image_to_text(self, image_bytes: bytes) -> str:
        engine = self._load()
        try:
            with self._lock:  # one ONNX session is shared between ingestion threads
                result, _ = engine(image_bytes)
        except Exception:
            log.exception("RapidOCR failed on a page image")
            return ""
        if not result:
            return ""
        lines = []
        for line in result:
            if len(line) > 1 and str(line[1]).strip():
                lines.append(str(line[1]).strip())
        return "\n".join(lines)


class TesseractEngine:
    """pytesseract backend. Requires the Tesseract executable on PATH (or PYTESSERACT_CMD)."""

    name = "tesseract"

    def __init__(self, language: str = "eng") -> None:
        self.language = language or "eng"

    def image_to_text(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise OCRUnavailable(
                "Tesseract OCR needs 'pytesseract' and 'Pillow'. Install them and the Tesseract executable."
            ) from None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return pytesseract.image_to_string(img, lang=self.language).strip()
        except pytesseract.TesseractNotFoundError:
            raise OCRUnavailable("The Tesseract executable was not found on PATH.") from None
        except pytesseract.TesseractError as e:
            raise OCRUnavailable(f"Tesseract could not process the page (check OCR_LANGUAGE): {e}") from None
        except Exception as e:
            # Unreadable image, unexpected OCR failure: treat this page as empty and keep going.
            log.error("tesseract OCR failed: %s", type(e).__name__)
            return ""


def _tesseract_available() -> bool:
    if importlib.util.find_spec("pytesseract") is None or importlib.util.find_spec("PIL") is None:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def build_ocr_engine(provider: str = "auto", language: str = "eng") -> OCREngine | None:
    """Return an OCR engine or None. 'auto' prefers the pip-only RapidOCR backend."""
    provider = (provider or "auto").strip().lower()
    if provider == "none":
        return None
    if provider == "rapidocr":
        return RapidOCREngine()
    if provider == "tesseract":
        return TesseractEngine(language)
    if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
        return RapidOCREngine()
    if _tesseract_available():
        return TesseractEngine(language)
    log.warning("OCR is enabled but no backend is available; scanned pages will be marked needs_ocr")
    return None
