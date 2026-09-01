"""Download and extract the Kunstformen der Natur plates.

Source: the "Art Forms in Nature / Kunst-Formen der Natur" Internet Archive item
(https://archive.org/details/KunstformenDerNaturErnstHaeckel), which hosts 300 dpi
scans of all 100 plates sourced from BioLib.de. The individual plate JPEGs archive.org
auto-generates as "derivatives" are lossy, downscaled samples (a handful exist, not
all 100) — the real full-resolution set lives inside one zip on the same item:
KunstformenDerNatur-ErnstHaeckel-Original300DpiImages.zip (~650MB, 102 images: the
100 numbered plates plus a cover/title page). This script downloads that zip once
and extracts it into data/raw/plates/.

Usage:
    python scripts/scrape_plates.py
    python scripts/scrape_plates.py --skip-download   # zip already downloaded, just extract
    python scripts/scrape_plates.py --keep-zip         # don't delete the zip after extracting
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ARCHIVE_ITEM = "KunstformenDerNaturErnstHaeckel"
ZIP_NAME = "KunstformenDerNatur-ErnstHaeckel-Original300DpiImages.zip"
DOWNLOAD_URL = f"https://archive.org/download/{ARCHIVE_ITEM}/{ZIP_NAME}"
ITEM_URL = f"https://archive.org/details/{ARCHIVE_ITEM}"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "_download"
PLATES_DIR = RAW_DIR / "plates"

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def download_zip(zip_path: Path) -> None:
    """Stream the zip to disk, resuming a partial download if one exists."""
    headers = {}
    mode = "wb"
    existing = zip_path.stat().st_size if zip_path.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code == 416:
            # Already fully downloaded (server rejects a range past EOF).
            print(f"Zip already complete at {zip_path}")
            return
        resp.raise_for_status()

        total = resp.headers.get("Content-Length")
        total = int(total) + existing if total else None
        downloaded = existing

        with open(zip_path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:8.1f} MB", end="", flush=True)
    print()


def extract_zip(zip_path: Path) -> list[str]:
    PLATES_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            # Flatten any directory structure inside the zip; we only want the images.
            target = PLATES_DIR / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target.name)
    return extracted


def write_sources_md(extracted: list[str]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# data/raw/plates provenance",
        "",
        f"Downloaded: {stamp}",
        f"Source item: {ITEM_URL}",
        f"Source file: `{ZIP_NAME}` ({DOWNLOAD_URL})",
        "License: public domain (Ernst Haeckel, 1899-1904; scans by BioLib.de).",
        f"Plate count: {len(extracted)}",
        "",
        "This directory is not tracked in git (see .gitignore) — re-run",
        "`scripts/scrape_plates.py` to regenerate it.",
    ]
    (RAW_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="zip already downloaded, just extract")
    parser.add_argument("--keep-zip", action="store_true", help="don't delete the zip after extracting")
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / ZIP_NAME

    if not args.skip_download:
        print(f"Downloading {ZIP_NAME} from {DOWNLOAD_URL}")
        download_zip(zip_path)

    if not zip_path.exists():
        print(f"error: {zip_path} not found (need to download first)", file=sys.stderr)
        return 1

    print(f"Extracting {zip_path.name} -> {PLATES_DIR}")
    extracted = extract_zip(zip_path)
    print(f"Extracted {len(extracted)} files.")

    write_sources_md(extracted)
    print(f"Wrote {RAW_DIR / 'SOURCES.md'}")

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
