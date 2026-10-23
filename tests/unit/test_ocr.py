"""OCR fallback tests. Synthetic PDFs + a fake OCR engine; the real engine is tested in tests/integration."""
import unittest

import pymupdf

from app.core.errors import OCRRequired, OCRUnavailable
from app.services.ingestion_service import extract_pages
from app.services.ocr import OCREngine, build_ocr_engine

TEXT = "Attention Is All You Need. This page already has a searchable text layer."


def text_pdf(text: str = TEXT, *, image_only: bool = False, pages: int = 1) -> bytes:
    """Build a PDF with a real text layer, or with only an image of that text (simulated scan)."""
    src = pymupdf.open()
    for _ in range(pages):
        page = src.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=14)
    if not image_only:
        data = src.tobytes()
        src.close()
        return data
    pngs = [page.get_pixmap(dpi=150).tobytes("png") for page in src]
    src.close()
    out = pymupdf.open()
    for png in pngs:
        page = out.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=png)
    data = out.tobytes()
    out.close()
    return data


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, text: str = "OCR EXTRACTED TEXT", fail: Exception | None = None):
        self.text, self.fail, self.calls = text, fail, 0
        self.images: list[bytes] = []

    def image_to_text(self, image_bytes: bytes) -> str:
        self.calls += 1
        self.images.append(image_bytes)
        if self.fail:
            raise self.fail
        return self.text


class ExtractPagesTests(unittest.TestCase):
    def test_text_layer_skips_ocr(self):
        ocr = FakeOCR()
        pages = extract_pages(text_pdf(), ocr)
        self.assertEqual(ocr.calls, 0)
        self.assertEqual(pages[0].page_number, 1)
        self.assertIn("Attention Is All You Need", pages[0].text)

    def test_image_only_page_uses_ocr(self):
        ocr = FakeOCR("RECOVERED SCANNED TEXT")
        pages = extract_pages(text_pdf(image_only=True), ocr)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(pages[0].text, "RECOVERED SCANNED TEXT")
        self.assertTrue(ocr.images[0].startswith(b"\x89PNG"))  # rendered page, not the PDF

    def test_short_text_layer_is_ocrd(self):
        ocr = FakeOCR("FULL TEXT FROM OCR")
        pages = extract_pages(text_pdf("Hi"), ocr)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(pages[0].text, "FULL TEXT FROM OCR")

    def test_page_numbers_stay_exact(self):
        ocr = FakeOCR("SCANNED PAGE")
        pages = extract_pages(text_pdf(image_only=True, pages=2), ocr)
        self.assertEqual([p.page_number for p in pages], [1, 2])
        self.assertEqual(ocr.calls, 2)

    def test_ocr_disabled_keeps_empty_text(self):
        pages = extract_pages(text_pdf(image_only=True), None)
        self.assertEqual(pages[0].text.strip(), "")
        self.assertEqual(pages[0].page_number, 1)

    def test_ocr_max_pages_guard(self):
        with self.assertRaises(OCRRequired) as ctx:
            extract_pages(text_pdf(image_only=True, pages=2), FakeOCR(), ocr_max_pages=1)
        self.assertIn("OCR_MAX_PAGES=1", ctx.exception.message)

    def test_ocr_failure_is_an_app_error(self):
        with self.assertRaises(OCRUnavailable):
            extract_pages(text_pdf(image_only=True), FakeOCR(fail=OCRUnavailable()))

    def test_ocr_text_only_replaces_worse_text(self):
        ocr = FakeOCR("x")  # shorter than the existing text layer
        pages = extract_pages(text_pdf("Hi"), ocr)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(pages[0].text.strip(), "Hi")


class EngineSelectionTests(unittest.TestCase):
    def test_none_disables_ocr(self):
        self.assertIsNone(build_ocr_engine("none"))

    def test_unknown_provider_is_rejected_by_settings(self):
        from app.core.config import Settings

        with self.assertRaises(ValueError):
            Settings(ocr_provider="something-else")

    def test_protocol_is_satisfied_by_fake(self):
        self.assertIsInstance(FakeOCR(), OCREngine)


if __name__ == "__main__":
    unittest.main()
