"""Check the 100 numbered plates for near-duplicates.

Each plate is a distinct book illustration, so this is mostly a sanity check rather than an
expected source of real duplicates (unlike a scraped photo corpus, there's no reprint/crop/
resize variation to catch here) -- but it's cheap to run and the project plan calls for it
before captioning, so it's worth actually doing rather than assuming.

Uses perceptual hashing (phash): any pair of plates with a Hamming distance below THRESHOLD
gets flagged for manual review. Tafel_000 (the cover page, not a content plate) is excluded.

Usage:
    python scripts/dedup_plates.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import imagehash
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATES_DIR = REPO_ROOT / "data" / "raw" / "plates"

THRESHOLD = 10  # Hamming distance; phash is 64 bits, <10 is a strong visual match


def main() -> int:
    plates = sorted(p for p in PLATES_DIR.glob("Tafel_*_300.jpg") if "_000_" not in p.name)
    if not plates:
        print(f"no plates found in {PLATES_DIR}")
        return 1

    hashes = {p: imagehash.phash(Image.open(p)) for p in plates}

    flagged = []
    for a, b in itertools.combinations(plates, 2):
        dist = hashes[a] - hashes[b]
        if dist < THRESHOLD:
            flagged.append((a.name, b.name, dist))

    print(f"Checked {len(plates)} plates ({len(list(itertools.combinations(plates, 2)))} pairs).")
    if flagged:
        print(f"{len(flagged)} near-duplicate pair(s) found (Hamming distance < {THRESHOLD}):")
        for a, b, dist in sorted(flagged, key=lambda x: x[2]):
            print(f"  {a} <-> {b}  (distance {dist})")
    else:
        print(f"No near-duplicates found (threshold: distance < {THRESHOLD}). All 100 plates are distinct.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
