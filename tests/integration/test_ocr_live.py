"""Live OCR smoke test (downloads nothing; the RapidOCR models ship with the package).

Skipped by default because it is slow on CPU. Run it explicitly:

    RUN_OCR_TESTS=1 python -m unittest tests.integration.test_ocr_live -v

On PowerShell:  $env:RUN_OCR_TESTS="1"; python -m unittest tests.integration.test_ocr_live -v
"""
import os
import unittest

import pymupdf

from app.services.ingestion_service import extract_pages
from app.services.ocr import RapidOCREngine

TEXT = "Attention Is All You Need"


def scanned_pdf(text: str) -> bytes:
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    page.insert_text((72, 72), text, fontsize=18)
    png = page.get_pixmap(dpi=200).tobytes("png")
    src.close()
    out = pymupdf.open()
    page = out.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=png)
    data = out.tobytes()
    out.close()
    return data


@unittest.skipUnless(os.getenv("RUN_OCR_TESTS") == "1", "set RUN_OCR_TESTS=1 to run the live OCR test")
class LiveRapidOCRTest(unittest.TestCase):
    def test_rapidocr_reads_a_scanned_page(self):
        pages = extract_pages(scanned_pdf(TEXT), RapidOCREngine(), ocr_dpi=200)
        self.assertIn("attention", pages[0].text.lower())
        self.assertEqual(pages[0].page_number, 1)


if __name__ == "__main__":
    unittest.main()
