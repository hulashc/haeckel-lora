# Haeckel LoRA: Generative Diffusion on *Kunstformen der Natur*

> **Status: data pipeline done, no training run yet.** Scraping, dedup, and captioning are
> complete and reproducible from scratch via the scripts below. Training and eval are not
> written yet. This README describes the plan for those — not a result.

## Progress

- [x] **Scrape** — all 100 plates (`scripts/scrape_plates.py`), sourced from the [Internet
      Archive item](https://archive.org/details/KunstformenDerNaturErnstHaeckel) (BioLib.de
      300dpi scans). Verified: 100 numbered plates + 1 cover page, ~2450x3600px each, no missing
      or corrupt files.
- [x] **Dedup** — `scripts/dedup_plates.py` (perceptual hash, threshold 10). Result: 0
      near-duplicates across all 100 plates — expected, since these are 100 distinct book
      illustrations rather than a scraped photo corpus with reprint/crop variation.
- [x] **Caption** — `scripts/build_captions.py`. Every plate carries its own printed caption
      (genus top-right, order + German name at the bottom), transcribed directly off each plate
      (see `data/processed/README.md` for why — Wikimedia Commons only categorizes 51/100 plates,
      and one of those 51 is outright wrong). Captions use the plate's own order name as the
      structural tag rather than a handful of coarse guessed buckets.
- [x] **Full per-figure documentation** (`data/processed/plate_details.json`) — beyond the one
      headline genus per plate, every individual numbered specimen (most plates show several,
      e.g. Tafel 8 has 4 distinct jellyfish) with its own species name, author, and description,
      sourced from Haeckel's actual companion explanatory volume (a *different* archive.org item
      than the plates themselves — see `data/processed/README.md`). 1102 figures across 100
      plates. This was a genuinely bug-prone pipeline (OCR page-boundary bugs, and two real
      LLM-fabrication incidents caught by cross-checking every extracted species name against its
      own source text) — the full incident writeup is in `data/processed/README.md` because the
      failure modes are worth knowing if this pipeline gets reused elsewhere. Two plates have
      honestly-documented gaps rather than invented content: plate 85's explanatory text is
      confirmed absent from the scan (empty `figures` list, stated why in its `taxonomy` field);
      plates 63 and 90 are missing their first 1-3 figures to scan damage.
- [ ] **Train** — rank-16 LoRA on SD1.5's UNet cross-attention layers.
- [ ] **Eval** — fixed prompt set sampled against training plates.

A LoRA fine-tune of Stable Diffusion 1.5 on Ernst Haeckel's *Kunstformen der Natur* (1899-1904),
exploring what a small, tight visual domain teaches a generative model that a broad one can't.

## Why Haeckel

*Kunstformen der Natur* is public domain, ~100 plates, and every plate follows the same recipe:
a single biological specimen, often radially symmetric, rendered as a lithograph on a warm cream
background with fine line work and saturated earth-tone ink. The visual signature is strong enough
that an output either looks like Haeckel or obviously doesn't, which makes the model's progress
genuinely inspectable rather than a vibes-based "looks cool to me" call.

Haeckel was also drawing things (deep-sea radiolaria, siphonophores) that nobody had seen in this
kind of detail before. A model that interpolates between plates should produce specimens that don't
exist but feel like they could — a version of what Haeckel was doing a century earlier, only now
the "field notes" are weights instead of microscope slides.

## Plan

1. **Scrape** the ~100 plates from a public-domain source (300 dpi scans available via the
   Internet Archive / BioLib, and the Biodiversity Heritage Library).
2. **Deduplicate** near-duplicate scans.
3. **Caption** each plate with a short Haeckel-specific trigger phrase plus a structural tag
   (`radiolaria` / `medusa` / `siphonophore` / `fern`).
4. **Fine-tune** Stable Diffusion 1.5 via a rank-16 LoRA adapter on the UNet cross-attention
   layers.

LoRA over a full fine-tune is the deliberate choice: the base model already knows what a
"jellyfish" looks like in a generic sense, and the goal is a *style and composition prior*
layered on top of that, not a model that forgets jellyfish exist. That's also why the caption
set needs structural tags rather than just `haeckel_plate` everywhere — it keeps the adapter
from collapsing every subject into one shape.

## Inference

Standard `diffusers` pipeline with the LoRA weights merged in, classifier-free guidance around
7.5, and 30-40 DDIM steps. The harder part won't be the sampling — it'll be evaluating a batch
from a fixed prompt set against the training plates (same composition, same species) to see
where the model stays faithful and where it invents.

## Stack

PyTorch, Diffusers, [PEFT](https://github.com/huggingface/peft) (LoRA), Stable Diffusion 1.5,
CUDA.

## Repo layout

```
data/
  raw/          scraped plates, untouched
  processed/    deduplicated + captioned training set
src/            scraping, captioning, training, inference code
scripts/        one-off CLI entry points (scrape, train, sample, eval)
notebooks/      exploration / eval notebooks
outputs/
  checkpoints/  LoRA adapter weights
  samples/      generated batches for eval
```

## Related

Companion/follow-up project: **Haeckel's Mistakes** — training two adapters on the same hand
drawing the same radiolarians in two registers (1887 Challenger Report vs. 1899-1904 Kunstformen)
to isolate and measure the idealization between them. Depends on this repo's training pipeline;
not started.

## License

Code: MIT (see `LICENSE`). Training data (Haeckel's plates) is public domain; no license file is
needed for it, but see `data/raw/SOURCES.md` (added once scraping starts) for provenance per plate.
