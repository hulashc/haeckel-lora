# data/processed/plate_index.csv

Per-plate metadata for all 100 numbered plates (Tafel 001-100; Tafel 000 is the book's cover
page, not a content plate, and is excluded here and from the training set).

Columns:
- `plate` — matches the `NNN` in `data/raw/plates/Tafel_NNN_300.jpg`
- `latin_name` — the genus/species printed top-right on the plate itself ("Tafel N — Name.")
- `order` — the taxonomic order/family Haeckel printed at the bottom of the plate
- `german_name` — the German common name printed alongside it (Fraktur in the original)

## Provenance

Transcribed directly off each plate's own printed caption — not from an external index — because
Wikimedia Commons only has a named category for 51 of the 100 plates, and even where it does, it's
not always reliable: Commons lists plate 44 as "Ammonoidea", but the plate itself clearly reads
"Ammonitida. — Ammonshörner." (verified visually). This file is the plate's own claim, cross-checked
against Commons wherever Commons had an entry (49 of 51 agreed exactly; the two disagreements were
Commons spelling variants except plate 44, which was a real wrong word on Commons' side).

Two crop bands per plate (top: publisher line + "Tafel N — Name."; bottom: order + German name)
were produced by `scripts/crop_captions.py` into `data/raw/caption_crops/`, then transcribed
visually in parallel batches of 10 — no OCR engine is installed in this environment, and the
stylized Victorian serif/Fraktur typefaces would likely fare worse under OCR than direct reading
anyway.
