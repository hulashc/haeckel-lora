# data/processed

## Reproducing from scratch

```
python scripts/scrape_plates.py          # -> data/raw/plates/ (the 100 illustration plates)
python scripts/crop_captions.py          # -> data/raw/caption_crops/ (for plate_index.csv;
                                          #    that file itself was transcribed by hand/vision,
                                          #    see below -- not a script step)
python scripts/scrape_text_volume.py     # -> data/raw/text_book.pdf (full book incl. Haeckel's
                                          #    own explanatory text, a DIFFERENT archive.org item)
python scripts/extract_book_pages.py     # -> data/raw/book_pages.json
python scripts/locate_plate_pages.py     # -> data/raw/plate_page_map.json
python scripts/extract_plate_text.py     # -> data/raw/plate_raw_text.json + plate_explanations/
# then the structuring pass itself (raw text -> taxonomy + figures) -- an LLM read/transcribe
# step, not a deterministic script; see "The pipeline, and what went wrong along the way" below
python scripts/merge_plate_details.py    # -> data/processed/plate_details.json
python scripts/build_captions.py         # -> data/processed/dataset/ (LoRA training captions)
```

## plate_index.csv

Per-plate headline metadata for all 100 numbered plates (Tafel 001-100; Tafel 000 is the book's
cover page, not a content plate, and is excluded here and from the training set).

Columns:
- `plate` — matches the `NNN` in `data/raw/plates/Tafel_NNN_300.jpg`
- `latin_name` — the genus/species printed top-right on the plate itself ("Tafel N — Name.")
- `order` — the taxonomic order/family Haeckel printed at the bottom of the plate
- `german_name` — the German common name printed alongside it (Fraktur in the original)

Transcribed directly off each plate's own printed caption — not from an external index — because
Wikimedia Commons only has a named category for 51 of the 100 plates, and even where it does,
it's not always reliable: Commons lists plate 44 as "Ammonoidea", but the plate itself clearly
reads "Ammonitida. — Ammonshörner." (verified visually). Two crop bands per plate (top: publisher
line + "Tafel N — Name."; bottom: order + German name) were produced by `scripts/crop_captions.py`
into `data/raw/caption_crops/`, then transcribed visually in parallel batches of 10 — no OCR
engine is installed in this environment, and the stylized Victorian serif/Fraktur typefaces would
likely fare worse under OCR than direct reading anyway. One transcription error was caught this
way and fixed: plate 41 was first read as "Doraspis", corrected to "Dorataspis" after
cross-checking against the full-book explanatory text (see below), then re-verified against the
plate image directly.

## plate_details.json

The richer layer: for every plate, Haeckel's own taxonomy lineage (Stamm/Klasse/Ordnung) and
**every individual numbered figure** on the plate — not just the one headline genus `plate_index`
captures. Most plates show several distinct specimens (e.g. Tafel 8 shows 4 different jellyfish,
numbered 1-4), each with its own species name, author, and a short factual note, sourced from
Haeckel's companion explanatory volume, not invented.

### Where the source text comes from

The plate scans themselves (`data/raw/plates/`) don't carry this — they're just the ~100
illustration plates, reprinted without their accompanying text. The explanatory prose lives in a
*different* archive.org item: **`kunstformenderna00haec`**, a BHL-sourced scan of the actual bound
volumes (both "Sammlungen", 554 pages total), which does include Haeckel's own descriptions.
`scripts/scrape_text_volume.py` pulls that PDF; `extract_book_pages.py`-style logic dumps it to
per-page OCR'd text (`data/raw/book_pages.json`).

### The pipeline, and what went wrong along the way

1. **`scripts/locate_plate_pages.py`** finds each plate's page range inside those 554 pages,
   anchored on the plate's own (already-verified) genus name. This took several iterations to get
   right, and the failures are worth recording because a couple of them produced genuinely
   convincing wrong output before being caught:
   - Plain substring matching grabbed genus names mentioned in *other* plates' body text (e.g.
     "Lagena", plate 81's genus, is also cited as an example inside plate 2's explanation) —
     fixed by requiring a whole-word match anchored near the top of the page.
   - A naive "how many times does 'Tafel' appear" table-of-contents detector misfired on pages
     that legitimately cross-reference other plates in running prose — fixed by keying off the
     literal "Inhalts-Verzeichnis" heading instead, plus a separate detector for a second,
     unlabeled kind of listing page that packs ~5 plates' headers back to back.
   - A handful of plates' real explanation opens directly with the *order* name and never repeats
     the genus at all (e.g. plate 37's real page reads "Siphonophorae. Staatsquallen. Stamm
     der..." with no "Discolabe" anywhere near the top). Genus-only anchoring skipped straight
     past the real page to the short plate-image-caption page later on, which restates "Tafel N —
     Genus" and so still matched — on the wrong, junk page. Fixed with a bounded repair pass:
     for any plate whose assigned range comes up suspiciously thin, search the narrow *gap*
     between it and the previous (trusted) plate for the order name instead. (Order name is
     deliberately **not** used as a primary anchor — many orders repeat across several plates,
     e.g. Siphonophorae covers plates 7, 17, 37, 59 and 77, so an unbounded order search just
     returns the first plate with that order anywhere in the book.)
   - The worst one: two *adjacent* plates (79 and 80) each resolved to a page that contained
     real, substantial, legitimate-*looking* taxonomy prose — just the neighbor's, not their own.
     Plate 79's assigned range technically contained the word "Lacertilia" (its own order), but
     only in a two-word image caption; the actual multi-paragraph page in that same range opened
     "Blastoidea..." — plate 80's content. A whole-block "does the order appear anywhere" check
     didn't catch this (the caption alone satisfied it); the fix checks each *individual*,
     substantial page within a plate's range for a competing order name in its own heading.
   - A repair to one plate can shrink its *neighbor's* range after the fact (ranges are computed
     as "up to the next plate's start"), which can turn a previously-fine neighbor thin without
     anything re-checking it. `locate_plate_pages.py` runs the whole detect-and-repair pass in a
     loop until a round fixes nothing, not just once.
2. **`scripts/extract_plate_text.py`** slices the page dump into one raw (still noisy) text block
   per plate.
3. **The structuring pass** — reading each plate's raw block and producing the cleaned taxonomy +
   figure list — was done by parallel LLM agents in batches of 10, then corrected/re-run for the
   subset of plates whose page ranges changed across the fixes above (see
   `scripts/merge_plate_details.py`'s `OVERRIDES` list for exactly which and why).

### Two real fabrication incidents, and how they were caught

Despite explicit instructions to only use the real source prose, the structuring pass **invented
entirely fictional figures** for two plates on the first attempt, when their assigned source text
turned out to be near-empty junk (an image caption, a few noise characters):
- **Plate 37**: 5 fully-detailed invented figures for a siphonophore that don't exist in the
  source at all.
- **Plate 85**: 12 invented ascidian species with specific enlargement factors and colony counts,
  again from a source block that, on inspection, contains no real explanatory prose anywhere.

Both were caught by a systematic **species-name traceability check**
(`difflib`-fuzzy-matching every extracted species' genus word against its own plate's raw source
text) run across all 100 plates after the first pass, not by having asked the model to
self-report. Plate 37's real content was recovered once the page-boundary bug above was fixed.
Plate 85's was not: a full-book search for both its genus ("Cynthia") and its order ("Ascidiae")
confirms neither turns up in a substantial explanation page anywhere in the 554-page scan — the
page is genuinely missing from this particular digitization, not just misfiled. **Plate 85's
entry in `plate_details.json` has an empty `figures` list and says so explicitly** rather than
being backfilled with plausible-sounding invention.

Two further plates have partial rather than total gaps, also stated directly in their own
`taxonomy` field rather than silently patched over:
- **Plate 63**: the page carrying its own opening heading and figure 1 is too degraded to read
  (confirmed by eye, not just automated detection) — figures start from #2.
- **Plate 90**: the opening page (taxonomy lineage + figures 1-3) is missing from the scan
  entirely, but its continuation page survived, including "Callocystis Jewetti (Hall)" at figure
  8 — confirming it's genuinely plate 90's own content, correctly reassigned after being trapped
  inside plate 89's over-wide range. Figures start from #4.

If you're extending this pipeline to other plates or another Haeckel volume: the lesson generalizes
past this specific dataset — verify a generation step's output traces back to real source text
before trusting it, especially when the source might be thin. A model given "here's some source
text, plus what this is supposed to be about" will readily produce something confident and
plausible-sounding from the second part alone when the first part doesn't actually support it.
