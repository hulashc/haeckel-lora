"""Build the caption files for LoRA training from data/processed/plate_index.csv and
data/processed/plate_details.json.

For each of the 100 numbered plates, writes an image + a same-named .txt caption file into
data/processed/dataset/ (the standard layout diffusers/kohya-style LoRA trainers expect: one
caption file per image, same basename). Caption format is a trigger phrase, a background-color
tag, order/genus tags, one real descriptive clause about the specimen, and a suffix, e.g.:

    haeckel_kunstformen, black background, phaeodaria, circogonia, Shell 0.7mm diameter shaped
    like a regular icosahedron with 20 triangular faces and 12 corners each bearing a hollow
    spiny radial spine, its central face pierced by a six-toothed mouth opening., natural
    history lithograph illustration

The background-color tag ("black background" or "pale background") is measured directly from
each plate image -- median grayscale luminance of a central crop (12%-88% of width/height, to
exclude the cream page margin and printed caption text that surrounds every plate regardless of
its own background) -- not guessed or LLM-generated, per this project's zero-tolerance policy on
fabricated caption content. Across all 100 plates the measured medians cluster into two groups
with a wide, clean gap between them (roughly <=160 vs. >=180 out of 255), so a single threshold
at 170 reliably separates them; see `detect_background` below. Added because an eval run against
a real checkpoint (2026-09-02, `outputs/eval/step-3000/`) showed the model frequently generating
the wrong background color for a given plate's real subject, and the pre-existing captions never
mentioned background color at all -- there was no textual signal for the model to learn it from.

The order/genus tags alone (the original caption format, still used as the prefix here) are
rare Latin/German taxonomic words that SD1.5's frozen CLIP text encoder -- trained on ordinary
web image-caption pairs -- almost certainly represents weakly. With ~90 distinct taxonomic
orders spread across only 100 training images, the model can't learn order-conditioned
composition from image statistics alone; it has to lean on the caption bridging to CLIP's
existing knowledge, and bare tags don't give it much to bridge with. The descriptive clause is
pulled *verbatim* from data/processed/plate_details.json's own figure `note` text -- itself
sourced from Haeckel's real explanatory volume (see data/processed/README.md) -- specifically
because it uses ordinary descriptive English ("spiky", "triangular", "shell") that CLIP has a
real prior for, rather than composing new text (this project has a documented zero-tolerance
history with LLM-fabricated caption content; selecting/truncating real text avoids that risk
entirely rather than trying to catch it after the fact).

Per-plate figure selection: plate_details.json documents every individual specimen shown on a
plate (most plates show several), but plate_index.csv records only one genus per plate (whichever
was printed top-right). We pick whichever figure's species matches that genus; if none match
(plate 063 only, whose matching figure was lost to scan damage) we fall back to the plate's first
documented figure; if the plate has no documented figures at all (plate 085 only, per
data/processed/README.md's fabrication-incident writeup -- the real explanatory text for that
plate is confirmed absent from the scan) we fall back to the original tags-only caption.

Every caption is fit to SD1.5's real 77-token CLIP limit using the actual tokenizer (not a
character-count guess) so nothing important silently falls off the end at train time: try the
full note; if it doesn't fit, greedily pack as many clauses (sentence/semicolon-delimited) as fit
in original order, but a clause that *doesn't* fit is skipped rather than treated as a hard stop
-- its comma-segments get a shot at partial credit, and packing continues into later clauses
regardless, so leftover token budget doesn't go unused just because one clause happened to be
long. (An earlier version of this function stopped dead at the first clause that didn't fit,
even with 30+ tokens of budget left over and the plate's only structurally-descriptive clause
sitting unused right after it -- e.g. plate 026's Carmaris caption used only 47/77 tokens and
silently dropped its only mention of tentacles. Fixed 2026-09-03; see CLAUDE.md for the incident
and the before/after caption diffs across the dataset.) If even the first clause's first
comma-segment alone overflows the budget, falls back to word-level truncation. 22/100 plates'
full notes exceed 77 tokens, so this isn't a rare edge case; 16/100 plates' final captions change
under this fix (all gaining real content, none regressing).

Usage:
    python scripts/build_captions.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from pathlib import Path

from PIL import Image
from transformers import CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATES_DIR = REPO_ROOT / "data" / "raw" / "plates"
INDEX_CSV = REPO_ROOT / "data" / "processed" / "plate_index.csv"
PLATE_DETAILS_JSON = REPO_ROOT / "data" / "processed" / "plate_details.json"
DATASET_DIR = REPO_ROOT / "data" / "processed" / "dataset"
CAPTIONS_JSONL = REPO_ROOT / "data" / "processed" / "captions.jsonl"

TRIGGER = "haeckel_kunstformen"
SUFFIX = "natural history lithograph illustration"
MAX_TOKENS = 77

# Measured from all 100 plates (2026-09-02): central-crop median luminance clusters tightly
# below 160 (black backgrounds) or above 180 (pale/cream backgrounds), with a clean gap between
# -- no plate lands near this threshold, so it isn't a close call in practice.
BACKGROUND_LUMINANCE_THRESHOLD = 170
CENTRAL_CROP_FRACTION = 0.12  # exclude this fraction of width/height from each edge


def detect_background(image_path: Path) -> str:
    """Measures each plate's own illustration background (not the page margin around it) by
    taking the median grayscale luminance of a central crop, and classifies it as black or pale.
    Measured directly from pixels, never guessed -- see module docstring."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    x0, x1 = int(w * CENTRAL_CROP_FRACTION), int(w * (1 - CENTRAL_CROP_FRACTION))
    y0, y1 = int(h * CENTRAL_CROP_FRACTION), int(h * (1 - CENTRAL_CROP_FRACTION))
    median_luminance = statistics.median(img.crop((x0, y0, x1, y1)).getdata())
    return "black background" if median_luminance <= BACKGROUND_LUMINANCE_THRESHOLD else "pale background"

# Must match the model scripts/train_lora.py and scripts/eval_lora.py actually load -- the
# tokenizer's vocabulary/behavior is what the 77-token budget is measured against.
DEFAULT_PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def select_note(genus: str, figures: list[dict]) -> tuple[str | None, str | None]:
    """Returns (note, figure_number), or (None, None) if this plate has no documented figures."""
    if not figures:
        return None, None
    match = next(
        (f for f in figures if f.get("species", "").split()[0].lower() == genus.lower()),
        None,
    )
    chosen = match or figures[0]
    note = (chosen.get("note") or "").strip()
    return (note or None), chosen.get("number")


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, truncation=False).input_ids)


def _candidate(prefix: str, note_text: str | None) -> str:
    return f"{prefix}, {note_text}, {SUFFIX}" if note_text else f"{prefix}, {SUFFIX}"


def _greedy_chain(tokenizer, prefix: str, acc: str, units: list[str], joiner: str) -> str:
    """Greedily append `units` (already in original order) onto `acc`. A unit that fits within
    the token budget is kept; one that doesn't is skipped -- not a hard stop -- so a later,
    shorter unit (possibly carrying more visually-relevant content) still gets a chance. Only
    ever appends real, verbatim text in its original order; never fabricates or reorders."""
    for u in units:
        u = u.strip()
        if not u:
            continue
        # Strip acc's trailing separator punctuation before joining -- acc may already end in
        # ";" or "," from a previous clause/segment, and joining that straight into `joiner`
        # (itself ", " or " ") would otherwise produce doubled punctuation like "Giltsch;, view".
        acc_stripped = acc.rstrip(".,;").strip()
        trial = (acc_stripped + joiner + u) if acc_stripped else u
        trial_clean = trial.strip().rstrip(".,;").strip()
        if _token_len(tokenizer, _candidate(prefix, trial_clean)) <= MAX_TOKENS:
            acc = trial
    return acc


def fit_caption(tokenizer, prefix: str, note: str) -> str:
    """Fit `note` into the caption budget, always keeping real text in original order -- never
    generating new content. Packs as many clauses/comma-segments as fit, skipping (not stopping
    at) any single unit that doesn't, so leftover budget isn't wasted on an early truncation."""
    full = _candidate(prefix, note)
    if _token_len(tokenizer, full) <= MAX_TOKENS:
        return full

    clauses = [c.strip() for c in re.split(r"(?<=[.;])\s+", note) if c.strip()]
    acc = ""
    for clause in clauses:
        # Space-joined, unlike the comma-joiner below -- a clause already ending in "." or ";"
        # reads naturally with " " + the next clause ("shell; four spines...", "opening. Next
        # clause..."), so acc's own trailing punctuation is kept, not stripped, here.
        trial = (acc + " " + clause) if acc else clause
        trial_clean = trial.strip().rstrip(".,;").strip()
        if _token_len(tokenizer, _candidate(prefix, trial_clean)) <= MAX_TOKENS:
            acc = trial
        else:
            # Whole clause doesn't fit -- try its comma-segments for partial credit, then
            # continue on to the *next* clause regardless (this is the fix: the old version
            # stopped dead here even when later, shorter clauses would still have fit).
            acc = _greedy_chain(tokenizer, prefix, acc, clause.split(", "), ", ")

    acc_clean = acc.strip().rstrip(".,;").strip()
    if acc_clean:
        return _candidate(prefix, acc_clean)

    # Nothing at clause/comma granularity fit at all -- only happens if the very first clause's
    # first comma-segment alone overflows the budget. Same word-level fallback as before.
    words = clauses[0].split(", ")[0].split(" ")
    for n in range(len(words), 0, -1):
        cand = _candidate(prefix, " ".join(words[:n]).rstrip(",;"))
        if _token_len(tokenizer, cand) <= MAX_TOKENS:
            return cand

    # Nothing of the note fits at all -- drop it rather than exceed budget.
    return _candidate(prefix, None)


def build_caption(
    tokenizer, order: str, genus: str, background: str, figures: list[dict]
) -> tuple[str, str | None, int]:
    prefix = f"{TRIGGER}, {background}, {order.lower()}, {genus.lower()}"
    note, figure_number = select_note(genus, figures)
    caption = _candidate(prefix, None) if note is None else fit_caption(tokenizer, prefix, note)
    return caption, figure_number, _token_len(tokenizer, caption)


def main() -> int:
    if not INDEX_CSV.exists():
        print(f"error: {INDEX_CSV} not found — nothing to caption")
        return 1
    if not PLATE_DETAILS_JSON.exists():
        print(f"error: {PLATE_DETAILS_JSON} not found — run the figure-documentation pipeline first")
        return 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(PLATE_DETAILS_JSON, encoding="utf-8") as f:
        details_by_plate = {entry["plate"]: entry for entry in json.load(f)}

    print(f"loading tokenizer from {DEFAULT_PRETRAINED_MODEL} (for 77-token budget fitting)")
    tokenizer = CLIPTokenizer.from_pretrained(DEFAULT_PRETRAINED_MODEL, subfolder="tokenizer")

    written = 0
    no_figures_fallback = 0
    no_genus_match_fallback = 0
    truncated = 0

    with open(CAPTIONS_JSONL, "w", encoding="utf-8") as jsonl:
        for row in rows:
            plate = row["plate"]
            src = PLATES_DIR / f"Tafel_{plate}_300.jpg"
            if not src.exists():
                print(f"warning: {src} missing, skipping plate {plate}")
                continue

            figures = details_by_plate.get(plate, {}).get("figures", [])
            genus = row["latin_name"]
            background = detect_background(src)
            caption, figure_number, token_count = build_caption(
                tokenizer, row["order"], genus, background, figures
            )

            if not figures:
                no_figures_fallback += 1
            elif not any(f.get("species", "").split()[0].lower() == genus.lower() for f in figures):
                no_genus_match_fallback += 1

            note_full, _ = select_note(genus, figures)
            if note_full is not None:
                full_prefix = f"{TRIGGER}, {background}, {row['order'].lower()}, {genus.lower()}"
                full_len = _token_len(tokenizer, _candidate(full_prefix, note_full))
                if full_len > MAX_TOKENS:
                    truncated += 1

            dst_img = DATASET_DIR / src.name
            dst_txt = DATASET_DIR / f"{src.stem}.txt"
            shutil.copyfile(src, dst_img)
            dst_txt.write_text(caption, encoding="utf-8")

            jsonl.write(json.dumps({
                "plate": plate,
                "file": dst_img.name,
                "caption": caption,
                "latin_name": genus,
                "order": row["order"],
                "background": background,
                "german_name": row["german_name"],
                "figure_number": figure_number,
                "token_count": token_count,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} image/caption pairs to {DATASET_DIR}")
    print(f"Wrote {CAPTIONS_JSONL}")
    print(
        f"  {no_figures_fallback} plate(s) had no documented figures (tags-only caption)\n"
        f"  {no_genus_match_fallback} plate(s) had no figure matching the index genus (used figure 1)\n"
        f"  {truncated} plate(s)' full note exceeded {MAX_TOKENS} tokens and were truncated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
