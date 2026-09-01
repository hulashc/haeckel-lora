"""Slice the full-book page dump into one raw text block per plate.

Combines data/raw/book_pages.json (554 OCR'd pages) with data/raw/plate_page_map.json (each
plate's [start, end) page range) into data/raw/plate_raw_text.json: {plate: raw_text}. Each
block still contains OCR noise (Fraktur long-s misreads, the plate-image's own low-quality OCR
mixed in, occasional stray diagram-caption pages) -- cleanup happens downstream, in the parsing
pass that reads these blocks and produces data/processed/plate_details.json.

Usage:
    python scripts/extract_plate_text.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_JSON = REPO_ROOT / "data" / "raw" / "book_pages.json"
PAGE_MAP_JSON = REPO_ROOT / "data" / "raw" / "plate_page_map.json"
OUT_JSON = REPO_ROOT / "data" / "raw" / "plate_raw_text.json"


def main() -> int:
    with open(PAGES_JSON, encoding="utf-8") as f:
        pages = json.load(f)
    with open(PAGE_MAP_JSON, encoding="utf-8") as f:
        page_map = json.load(f)

    out = {}
    for plate, info in sorted(page_map.items()):
        block = "\n\n".join(pages[info["start"]:info["end"]])
        out[plate] = block

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    lengths = [len(v) for v in out.values()]
    print(f"Wrote {len(out)} plate text blocks to {OUT_JSON}")
    print(f"Length range: {min(lengths)}-{max(lengths)} chars, avg {sum(lengths)//len(lengths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
