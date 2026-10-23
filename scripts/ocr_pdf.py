"""Diagnose text extraction and OCR for one PDF without touching the database.

  python -m scripts.ocr_pdf "C:\\path\\to\\paper.pdf"
  python -m scripts.ocr_pdf "C:\\path\\to\\paper.pdf" --provider none   # text layer only
  python -m scripts.ocr_pdf "C:\\path\\to\\paper.pdf" --save-text out.txt

The first OCR call loads RapidOCR's bundled models, which can take ~20 seconds.
"""
import argparse
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.ingestion_service import extract_pages
from app.services.ocr import build_ocr_engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", help="path to a PDF file")
    ap.add_argument("--provider", default=None, help="auto | rapidocr | tesseract | none (default: .env)")
    ap.add_argument("--language", default=None, help="Tesseract language(s), e.g. eng or ara (default: .env)")
    ap.add_argument("--dpi", type=int, default=None, help="OCR render DPI (default: .env)")
    ap.add_argument("--min-chars", type=int, default=None, help="OCR pages below this many text chars (default: .env)")
    ap.add_argument("--max-pages", type=int, default=None, help="OCR at most N pages; 0 = unlimited (default: .env)")
    ap.add_argument("--show", type=int, default=200, help="characters of each page to print (default: 200)")
    ap.add_argument("--save-text", default=None, help="write the full extracted text to this file")
    a = ap.parse_args()

    path = Path(a.pdf).expanduser()
    if not path.is_file():
        print(f"file not found: {path}")
        return 2
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        print("not a PDF: missing %PDF- header")
        return 2

    st = get_settings()
    provider = a.provider or st.ocr_provider
    language = a.language or st.ocr_language
    dpi = a.dpi if a.dpi is not None else st.ocr_dpi
    min_chars = a.min_chars if a.min_chars is not None else st.ocr_min_page_chars
    max_pages = a.max_pages if a.max_pages is not None else st.ocr_max_pages

    ocr = None if provider == "none" else build_ocr_engine(provider, language)
    print(f"file: {path} ({len(data) / 1024 / 1024:.1f} MB)")
    print(f"provider: {getattr(ocr, 'name', 'none')}  dpi: {dpi}  min_chars: {min_chars}  max_pages: {max_pages}")
    if provider != "none" and ocr is None:
        print("warning: no OCR backend is available; only the PDF text layer will be used")

    started = time.time()
    try:
        pages = extract_pages(data, ocr, dpi, min_chars, max_pages)
    except Exception as e:
        print(f"extraction failed: {type(e).__name__}: {e}")
        return 1

    total = 0
    empty = 0
    for page in pages:
        text = page.text.strip()
        total += len(text)
        if not text:
            empty += 1
        preview = text[:max(0, a.show)].replace("\n", " ")
        print(f"  page {page.page_number}: {len(text)} chars | {preview!r}")
    print(f"pages: {len(pages)}  empty: {empty}  total chars: {total}  elapsed: {time.time() - started:.1f}s")

    if a.save_text:
        out = Path(a.save_text).expanduser()
        out.write_text("\n\n".join(f"--- page {p.page_number} ---\n{p.text}" for p in pages), encoding="utf-8")
        print(f"text written to: {out}")

    if total == 0:
        print("result: no text extracted; the scan quality, rotation or OCR language is the likely cause")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
