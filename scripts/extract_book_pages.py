"""Dump the full-book PDF's OCR text layer to one JSON list, one entry per page.

Source: kunstformenderna00haec.pdf (downloaded by scripts/scrape_text_volume.py into
data/raw/text_book.pdf), a BHL-sourced scan of the complete bound Kunstformen der Natur volumes
(554 pages) including Haeckel's own explanatory text for each plate -- unlike the plates-only
archive.org item scripts/scrape_plates.py pulls from.

Output: data/raw/book_pages.json -- a JSON array of 554 strings, index == page number, used by
scripts/locate_plate_pages.py and scripts/extract_plate_text.py downstream.

Usage:
    python scripts/extract_book_pages.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / "data" / "raw" / "text_book.pdf"
OUT_JSON = REPO_ROOT / "data" / "raw" / "book_pages.json"


def main() -> int:
    if not PDF_PATH.exists():
        print(f"error: {PDF_PATH} not found — run scripts/scrape_text_volume.py first")
        return 1

    doc = pymupdf.open(PDF_PATH)
    pages = [doc[i].get_text() for i in range(doc.page_count)]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    print(f"Dumped {len(pages)} pages to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
