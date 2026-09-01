"""Build the caption files for LoRA training from data/processed/plate_index.csv.

For each of the 100 numbered plates, writes an image + a same-named .txt caption file into
data/processed/dataset/ (the standard layout diffusers/kohya-style LoRA trainers expect: one
caption file per image, same basename). Caption format is a short trigger phrase plus tags,
per the project plan: a Haeckel-specific trigger token, then the plate's own printed
order/family name as the structural tag, then its genus, e.g.:

    haeckel_kunstformen, discomedusae, desmonema, natural history lithograph illustration

The order name is used as-is rather than collapsed into a handful of coarse buckets
(radiolaria/medusa/etc.) -- the plate index has the *exact* printed order for all 100 plates,
which is strictly more specific than a coarse bucket and lets the LoRA learn per-order
composition (e.g. Discomedusae's tangled tentacle mass vs. Cirripedia's segmented plates)
rather than lumping everything under a handful of guessed categories.

Usage:
    python scripts/build_captions.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATES_DIR = REPO_ROOT / "data" / "raw" / "plates"
INDEX_CSV = REPO_ROOT / "data" / "processed" / "plate_index.csv"
DATASET_DIR = REPO_ROOT / "data" / "processed" / "dataset"
CAPTIONS_JSONL = REPO_ROOT / "data" / "processed" / "captions.jsonl"

TRIGGER = "haeckel_kunstformen"


def build_caption(order: str, latin_name: str) -> str:
    return f"{TRIGGER}, {order.lower()}, {latin_name.lower()}, natural history lithograph illustration"


def main() -> int:
    if not INDEX_CSV.exists():
        print(f"error: {INDEX_CSV} not found — nothing to caption")
        return 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(INDEX_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    written = 0
    with open(CAPTIONS_JSONL, "w", encoding="utf-8") as jsonl:
        for row in rows:
            plate = row["plate"]
            src = PLATES_DIR / f"Tafel_{plate}_300.jpg"
            if not src.exists():
                print(f"warning: {src} missing, skipping plate {plate}")
                continue

            caption = build_caption(row["order"], row["latin_name"])

            dst_img = DATASET_DIR / src.name
            dst_txt = DATASET_DIR / f"{src.stem}.txt"
            shutil.copyfile(src, dst_img)
            dst_txt.write_text(caption, encoding="utf-8")

            jsonl.write(json.dumps({
                "plate": plate,
                "file": dst_img.name,
                "caption": caption,
                "latin_name": row["latin_name"],
                "order": row["order"],
                "german_name": row["german_name"],
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} image/caption pairs to {DATASET_DIR}")
    print(f"Wrote {CAPTIONS_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
