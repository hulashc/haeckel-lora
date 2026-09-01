"""Download Haeckel's own descriptive text volume for Kunstformen der Natur.

The plate captions (data/processed/plate_index.csv) only capture what's printed ON each plate:
one headline genus, the order, and a German common name. Most plates actually depict several
numbered specimens (e.g. Tafel 8 shows 4 distinct jellyfish, numbered 1-4) which aren't
individually named on the plate itself -- Haeckel named and described each one in the companion
explanatory text bound into the original volumes.

Note this is a DIFFERENT archive.org item than scripts/scrape_plates.py uses. The plates-only item
(KunstformenDerNaturErnstHaeckel) also has an "Additional Text PDF", but it turns out to just be a
low-quality OCR of the plate images themselves (same captions plate_index.csv already has, nothing
more) -- a dead end tried and discarded during this project's data pipeline. The actual
explanatory text lives in a separate, BHL-sourced item that scanned the complete bound volumes:
kunstformenderna00haec (554 pages, both "Sammlungen", OCR'd with real page-text quality).

Usage:
    python scripts/scrape_text_volume.py
"""

from __future__ import annotations

from pathlib import Path

import requests

ARCHIVE_ITEM = "kunstformenderna00haec"
PDF_NAME = "kunstformenderna00haec.pdf"
DOWNLOAD_URL = f"https://archive.org/download/{ARCHIVE_ITEM}/{PDF_NAME}"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "text_book.pdf"

CHUNK_SIZE = 1024 * 1024


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = OUT_PATH.stat().st_size if OUT_PATH.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416:
            print(f"Already fully downloaded: {OUT_PATH}")
            return 0
        resp.raise_for_status()
        total = resp.headers.get("Content-Length")
        total = int(total) + existing if total else None
        downloaded = existing
        with open(OUT_PATH, mode) as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e6:8.1f} / {total / 1e6:.1f} MB ({downloaded / total * 100:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:8.1f} MB", end="", flush=True)
    print()
    print(f"Saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
