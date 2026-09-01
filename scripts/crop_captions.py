"""Crop each plate's printed caption bands into one small composite image.

Every Kunstformen der Natur plate carries its own caption, printed by Haeckel's
publisher: a top strip ("Haeckel, Kunstformen der Natur." / "Tafel N -- Genus.")
and a bottom strip (the order name plus its German common name, e.g.
"Discomedusae. -- Scheibenquallen."). No OCR engine is installed in this
environment, and the font is a stylized Victorian serif that OCR handles
poorly anyway -- so instead of transcribing programmatically, this script
crops just those two bands (not the whole ~2450x3600 plate) into one small
stacked composite per plate, small enough to read visually and cheap enough
to do for all 101 in bulk.

Output: data/raw/caption_crops/Tafel_NNN_300_caption.jpg
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATES_DIR = REPO_ROOT / "data" / "raw" / "plates"
OUT_DIR = REPO_ROOT / "data" / "raw" / "caption_crops"

TOP_BAND = (0.0, 0.045)     # fraction of image height: publisher/plate-number line
BOTTOM_BAND = (0.955, 1.0)  # fraction of image height: order / German name line
OUT_WIDTH = 1400            # downscale width for the composite (still legible)


def crop_band(im: Image.Image, frac_top: float, frac_bottom: float) -> Image.Image:
    w, h = im.size
    return im.crop((0, int(h * frac_top), w, int(h * frac_bottom)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plates = sorted(PLATES_DIR.glob("Tafel_*_300.jpg"))
    if not plates:
        print(f"no plates found in {PLATES_DIR}")
        return 1

    for path in plates:
        im = Image.open(path).convert("RGB")
        top = crop_band(im, *TOP_BAND)
        bottom = crop_band(im, *BOTTOM_BAND)

        scale = OUT_WIDTH / im.width
        top = top.resize((OUT_WIDTH, int(top.height * scale)))
        bottom = bottom.resize((OUT_WIDTH, int(bottom.height * scale)))

        gap = 12
        composite = Image.new("RGB", (OUT_WIDTH, top.height + bottom.height + gap), "white")
        composite.paste(top, (0, 0))
        composite.paste(bottom, (0, top.height + gap))

        out_path = OUT_DIR / f"{path.stem}_caption.jpg"
        composite.save(out_path, quality=90)

    print(f"Wrote {len(plates)} caption crops to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
