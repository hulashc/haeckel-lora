"""Merge the per-batch structured-extraction JSON files into data/processed/plate_details.json.

The structuring step itself (data/raw/plate_explanations/*.txt -> per-plate taxonomy + figure
list) isn't a deterministic script -- it's an LLM read-and-transcribe pass over noisy OCR text,
done in parallel batches (see data/processed/README.md for the full methodology and its
revisions). This script just does the mechanical, deterministic part: take whatever batch/
override files exist in data/processed/plate_details_partial/ and combine them into one ordered,
validated plate_details.json, later files winning over earlier ones for the same plate number.

Usage:
    python scripts/merge_plate_details.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARTIAL_DIR = REPO_ROOT / "data" / "processed" / "plate_details_partial"
OUT_JSON = REPO_ROOT / "data" / "processed" / "plate_details.json"

# Order matters: later files override earlier ones for the same plate number. The base batches
# come first; anything that needed a correction after a page-boundary bug or a fabrication catch
# is layered on top, in the order those fixes were made.
BASE_BATCHES = [
    "001-010.json", "011-020.json", "021-030.json", "031-040.json", "041-050.json",
    "051-060.json", "061-070.json", "071-080.json", "081-090.json", "091-100.json",
]
OVERRIDES = [
    "corrected-13.json",  # 13 plates re-extracted after a page-boundary bug was fixed
    "090-fixed.json",     # plate 90's recovered continuation page (base batch had none)
    "085-manual.json",    # plate 85: source text confirmed absent from the scan; base batch
                           # had fabricated this one from background knowledge -- discarded
]


def main() -> int:
    plates: dict[str, dict] = {}
    for filename in BASE_BATCHES + OVERRIDES:
        path = PARTIAL_DIR / filename
        if not path.exists():
            print(f"warning: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                plates[entry["plate"]] = entry

    missing = [f"{i:03d}" for i in range(1, 101) if f"{i:03d}" not in plates]
    if missing:
        print(f"error: missing plates after merge: {missing}")
        return 1

    ordered = [plates[f"{i:03d}"] for i in range(1, 101)]
    total_figures = sum(len(e.get("figures", [])) for e in ordered)
    empty = [e["plate"] for e in ordered if not e.get("figures")]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(ordered)} plates, {total_figures} total figures.")
    if empty:
        print(f"Plates with no figures (see their 'taxonomy' note for why): {empty}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
