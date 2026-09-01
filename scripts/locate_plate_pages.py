"""Locate each plate's explanatory-text page range inside the full-book PDF.

data/raw/book_pages.json (produced by extracting kunstformenderna00haec.pdf page-by-page, see
the top of this repo's history) holds 554 pages of OCR'd text: front matter, then for each of
the 100 plates a 1-3 page explanation (taxonomy + a numbered per-figure species list) followed
by a short, noisy OCR of the plate image itself, followed eventually (~page 495 on) by a back
"systematic overview" supplement that is NOT per-plate description and must be excluded.

Strategy: each plate's explanation page always opens with "Tafel N. -- Genus." as a heading, so
the plate's own genus name (already verified visually in data/processed/plate_index.csv) is a
reliable anchor -- reliable in nearly all cases; OCR still garbles a handful of genus names, so
those get a fuzzy fallback confined to the page range bounded by their neighboring plates (which
are already anchored), rather than a blind full-book fuzzy search.

Output: data/raw/plate_page_map.json -- {plate: {"start": int, "end": int, "resolved_by":
"exact"|"fuzzy"|"manual"}}
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_JSON = REPO_ROOT / "data" / "raw" / "book_pages.json"
INDEX_CSV = REPO_ROOT / "data" / "processed" / "plate_index.csv"
OUT_JSON = REPO_ROOT / "data" / "raw" / "plate_page_map.json"

BACK_MATTER_START = 495  # "systematische Übersicht" supplement begins around here
FRONT_MATTER_END = 8     # title page / preface, before Lieferung 1's ToC

# A handful of plates have their heading OCR'd too badly for even the fuzzy pass to clear the
# similarity threshold (the scanned page itself is degraded, not just an OCR quirk -- e.g. plate
# 76's page 374 heading is missing "Tafel 76"/"Alima" entirely, but the body text's order name
# "Thoracostraca" matches plate 76's own order exactly). Found by eye from each one's bounded
# neighbor range (see the UNRESOLVED ranges a first run prints) and confirmed by content, not
# just a name match: plate 63's page 312/313 discusses Phallus impudicus / "Morchel-Giftpilz"
# (fungi), matching order Basimycetes; plate 95's page 468 reads "Amphoridea. Urnensterne."
# matching plate 95's own order exactly.
# Stragglers even after fixing the ToC-detection and word-boundary bugs above. Confirmed by
# content, not just page proximity:
MANUAL_OVERRIDES = {
    # page 312/313 discusses Phallus impudicus / "Morchel-Giftpilz" (fungi), matching plate 63's
    # own order Basimycetes exactly. The scan itself is degraded here (not an anchoring miss),
    # so order-anchoring can't rescue this one automatically.
    "063": 312,
    # plate 90's own explanation is missing its opening page (the "Stamm der..." lineage plus
    # figs 1-3) entirely from this scan -- but its CONTINUATION page (figs 4-15, including
    # "Callocystis Jewetti (Hall)" at fig. 8, confirming it's genuinely plate 90's own content)
    # survived, wrongly trapped inside plate 89's oversized span (page 441 doesn't repeat
    # "Testudo"/"Chelonia" -- plate 89's own genus/order -- at all). Confirmed by content.
    "090": 441,
}

WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]{3,}\b")

# The heading "Tafel N. -- Genus." always sits at/near the very top of the plate's own
# explanation page. Genus names also turn up deeper in OTHER plates' body text (e.g. "Lagena"
# is cited as an example Foraminifera inside Tafel 2's explanation, long before Tafel 81 -- its
# own plate -- appears; likewise a word can be a substring of an unrelated one, e.g. "Asterias"
# inside "Micrasterias"), so matches must be a whole word anchored near the page start, not just
# a substring anywhere on the page, or the search grabs the wrong plate's page entirely.
HEADING_WINDOW = 200


LISTING_RE = re.compile(r"[TSC]afel\s+\d+\.?\s*[A-Za-z]")


def is_toc_page(text: str) -> bool:
    """Each installment ('Heft') opens with an "Inhalts-Verzeichnis" (table of contents) page
    listing that Heft's 10 plates -- these OCR inconsistently ("Inhalts-Derzeichnis",
    "Inhalts-Beyzeichnis", ...) but always start with some recognizable form of "Inhalt". A
    naive "how many times does 'Tafel' appear on this page" check is NOT reliable here: real
    explanation pages legitimately cross-reference several *other* plates by number in running
    text (radiolarian plates especially, e.g. "vgl. Tafel 21"), which can trip a count-based
    threshold on a genuine content page.

    Separately, a few Heft boundaries (e.g. before Tafel 41) carry an unlabeled *second* kind of
    listing page -- no "Inhalt" heading, just "Tafel 41. Dorataspis. ... Tafel 42. Ostracion. ..."
    packed back to back for several plates. A genuine explanation page mentions its own "Tafel N."
    once at the very top (and rarely a second time, e.g. "Zur Tafel N" on a diagram-legend page);
    one of these listing pages packs several "Tafel <number> <Capitalized word>" headings within
    the first 600 characters, which a real explanation page's own heading plus occasional running
    cross-references never does.
    """
    if "inhalt" in text[:60].lower():
        return True
    # >=2 was too aggressive: a genuine explanation page can legitimately cross-reference one
    # other plate in running prose (e.g. "...Farnpflanzen, Tafel 52 u. 92, zu den..."), which
    # also matches this pattern once more. Real listing pages pack ~5 sequential plate headers
    # into the first 600 chars; a single incidental cross-reference tops out at 2.
    return len(LISTING_RE.findall(text[:600])) >= 3


def find_genus(text: str, genus: str) -> bool:
    return re.search(rf"\b{re.escape(genus)}\b", text, re.IGNORECASE) is not None


def find_exact(pages: list[str], genus: str) -> int | None:
    """Anchor on genus alone.

    Order name is NOT usable as a global anchor: many orders repeat across several plates
    (Siphonophorae alone covers plates 7, 17, 37, 59, 77), so an order-anchored search just
    returns the *first* plate with that order anywhere in the book, not necessarily the right
    one. See `repair_order_anchored_gap` below for where order name is used safely instead --
    bounded to a narrow, already-suspect gap between two genus-anchored plates.
    """
    for i in range(FRONT_MATTER_END, BACK_MATTER_START):
        if not is_toc_page(pages[i]) and find_genus(pages[i][:HEADING_WINDOW], genus):
            return i
    return None


def repair_order_anchored_gap(pages: list[str], order: str, lo: int, hi: int) -> int | None:
    """Look for a plate's real explanation page hiding earlier than its genus-matched page.

    A handful of plates' real explanation opens directly with the order name and never repeats
    the genus at all (e.g. plate 37's real page reads "Siphonophorae. Staatsquallen. Stamm
    der..." with no "Discolabe" anywhere near the top) -- genus-only anchoring skips straight
    past that real page to the short plate-image-caption page later on (which always restates
    "Tafel N -- Genus", so it still matches on genus, just on the wrong, later page).

    Only called for plates the safety-net check below already flagged as suspiciously thin, and
    only searches the narrow gap strictly between the previous (trusted, genus-anchored) plate's
    own start and this plate's own genus-matched page -- NOT the whole book -- so a repeated
    order name elsewhere is not in scope to false-match against.
    """
    best_page = None
    for i in range(lo, hi):
        if is_toc_page(pages[i]):
            continue
        page = pages[i]
        if len(page) > 800 and find_genus(page[:HEADING_WINDOW], order):
            best_page = i
            break  # first substantial match in the gap, in page order
    return best_page


def find_fuzzy(pages: list[str], genus: str, lo: int, hi: int) -> tuple[int | None, str | None]:
    """Search only the page range bounded by neighboring (already-anchored) plates."""
    best_page, best_word, best_ratio = None, None, 0.0
    for i in range(lo, min(hi, BACK_MATTER_START)):
        if is_toc_page(pages[i]):
            continue
        for word in WORD_RE.findall(pages[i][:HEADING_WINDOW]):
            ratio = difflib.SequenceMatcher(None, word.lower(), genus.lower()).ratio()
            if ratio > best_ratio:
                best_page, best_word, best_ratio = i, word, ratio
    if best_ratio >= 0.6:
        return best_page, best_word
    return None, None


def main() -> int:
    with open(PAGES_JSON, encoding="utf-8") as f:
        pages = json.load(f)
    with open(INDEX_CSV, encoding="utf-8") as f:
        plates = list(csv.DictReader(f))

    starts: dict[str, dict] = {}
    unresolved = []

    for row in plates:
        plate, genus = row["plate"], row["latin_name"]
        if plate in MANUAL_OVERRIDES:
            starts[plate] = {"start": MANUAL_OVERRIDES[plate], "resolved_by": "manual"}
            continue
        page = find_exact(pages, genus)
        if page is not None:
            starts[plate] = {"start": page, "resolved_by": "exact"}
        else:
            unresolved.append(plate)

    # Fuzzy fallback for anything exact matching missed, bounded by its resolved neighbors.
    plate_nums = [row["plate"] for row in plates]
    for plate in unresolved:
        idx = plate_nums.index(plate)
        prev_plate = next((plate_nums[j] for j in range(idx - 1, -1, -1) if plate_nums[j] in starts), None)
        next_plate = next((plate_nums[j] for j in range(idx + 1, len(plate_nums)) if plate_nums[j] in starts), None)
        lo = starts[prev_plate]["start"] + 1 if prev_plate else FRONT_MATTER_END
        hi = starts[next_plate]["start"] if next_plate else BACK_MATTER_START
        genus = next(r["latin_name"] for r in plates if r["plate"] == plate)
        page, matched_word = find_fuzzy(pages, genus, lo, hi)
        if page is not None:
            starts[plate] = {"start": page, "resolved_by": "fuzzy", "matched_word": matched_word}
        else:
            print(f"UNRESOLVED: plate {plate} ({genus}) in range [{lo}, {hi})")

    def recompute_ends():
        ordered = [p for p in plate_nums if p in starts]
        for i, plate in enumerate(ordered):
            end = starts[ordered[i + 1]]["start"] if i + 1 < len(ordered) else BACK_MATTER_START
            starts[plate]["end"] = end
        return ordered

    ordered = recompute_ends()

    # Safety net: a wrong page range doesn't just mis-locate content, it can leave a plate with
    # only the short plate-image-caption junk to work with -- which silently invites a downstream
    # LLM cleanup pass to *fabricate* a plausible-looking explanation instead of reporting "no
    # real content found" (this happened for real: plate 37 first shipped 5 invented figures
    # before this check existed). A real explanation page always opens with one of Haeckel's
    # taxonomy-lineage phrases ("Stamm der", "Klasse der", "Ordnung der", "Legion der", plus
    # common OCR corruptions of "Stamm"/"Klasse"); anything shorter than ~400 chars or missing
    # all of these is almost certainly junk and must be checked by hand, not trusted.
    # Loose on purpose: OCR renders "Stamm" as "Stanım"/"Stanm"/"Stamm", "Klasse" as
    # "Klaffe"/"Klafie"/"Klaf(f)e" (long-s -> f), etc. Matching root fragments without requiring
    # an exact suffix or adjacent "der" avoids the false-positive-empty result an over-precise
    # regex gave on a first pass (it flagged plates already confirmed by hand to have real prose).
    TAXONOMY_MARKERS = re.compile(r"(stam|stanı?m|klaf|hauptklaf|ordnung|legion)", re.IGNORECASE)
    order_by_plate = {r["plate"]: r["order"] for r in plates}
    all_orders = sorted(set(order_by_plate.values()), key=len, reverse=True)  # longest first

    def fuzzy_order_match(head: str) -> str | None:
        """Which (if any) known order name heads this page, exact or close OCR variant."""
        for order in all_orders:
            if find_genus(head, order):
                return order
        for order in all_orders:
            for w in re.findall(r"[A-Za-z]{4,}", head):
                if difflib.SequenceMatcher(None, w.lower(), order.lower()).ratio() >= 0.8:
                    return order
        return None

    def find_thin(plate_list):
        """Two independent checks, either one is disqualifying:

        1. length/generic-marker: catches ranges with no real taxonomy prose at all (037, 085).
        2. per-page contamination: catches ranges that DO contain substantial, legitimate-looking
           taxonomy prose, but it's a NEIGHBOR's, not this plate's own -- e.g. plate 79's assigned
           range technically contains "Lacertilia" (its own order, but only in a two-word caption
           line) while the substantial multi-paragraph page in that same range actually opens
           "Blastoidea..." (plate 80's own order). A whole-block "does the order appear anywhere"
           check missed this (the short caption alone satisfied it); checking each *substantial*
           constituent page's own heading for a DIFFERENT plate's order name catches it instead.
        """
        thin = []
        for plate in plate_list:
            info = starts[plate]
            own_order = order_by_plate[plate]
            block_pages = pages[info["start"]:info["end"]]
            block = "\n".join(block_pages)
            if len(block) < 400 or not TAXONOMY_MARKERS.search(block):
                thin.append(plate)
                continue
            contaminated = False
            for page in block_pages:
                if len(page) <= 800:
                    continue
                matched = fuzzy_order_match(page[:HEADING_WINDOW])
                if matched is not None and matched != own_order:
                    contaminated = True
                    break
            if contaminated:
                thin.append(plate)
        return thin

    # Fixed-point loop: repairing one plate's start shrinks its PREVIOUS neighbor's end (ends are
    # "next plate's start"), which can retroactively turn a previously-fine neighbor thin. A
    # single repair pass missed exactly this -- plate 76's repair shrank plate 75's span out from
    # under it, and nothing re-checked plate 75 afterward. Loop until a round repairs nothing.
    all_repaired = []
    for _ in range(5):
        thin = find_thin(ordered)
        repaired = []
        for plate in thin:
            idx = plate_nums.index(plate)
            prev_plate = next((plate_nums[j] for j in range(idx - 1, -1, -1) if plate_nums[j] in starts), None)
            if prev_plate is None or plate not in starts:
                continue
            lo = starts[prev_plate]["start"] + 1
            hi = starts[plate]["start"]
            order = next(r["order"] for r in plates if r["plate"] == plate)
            page = repair_order_anchored_gap(pages, order, lo, hi)
            if page is not None and page != starts[plate]["start"]:
                starts[plate] = {"start": page, "resolved_by": "order-repair"}
                repaired.append(plate)
        if not repaired:
            break
        ordered = recompute_ends()
        all_repaired.extend(repaired)

    if all_repaired:
        print(f"\nRepaired {len(all_repaired)} plate(s) via bounded order-name anchoring: {all_repaired}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(starts, f, ensure_ascii=False, indent=2)

    exact = sum(1 for v in starts.values() if v["resolved_by"] == "exact")
    fuzzy = sum(1 for v in starts.values() if v["resolved_by"] == "fuzzy")
    print(f"Resolved {len(starts)}/100 plates ({exact} exact, {fuzzy} fuzzy).")
    print(f"Wrote {OUT_JSON}")

    still_thin = find_thin(ordered)
    if still_thin:
        print(f"\nWARNING: {len(still_thin)} plate(s) STILL have no real taxonomy prose after repair —")
        print("verify by hand before trusting any downstream extraction from these:")
        for plate in still_thin:
            info = starts[plate]
            length = len("\n".join(pages[info["start"]:info["end"]]))
            print(f"  plate {plate}: only {length} chars, no taxonomy marker found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
