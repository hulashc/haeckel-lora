# Haeckel LoRA: Generative Diffusion on *Kunstformen der Natur*

> **Status: trained, evaluated, and one real bug found and fixed — not a finished adapter.**
> Data pipeline, training, and eval are all done and reproducible from scratch via the scripts
> below. A 124-epoch/3,100-step run confirmed loss plateaus around epoch 50 (more epochs on this
> 100-image dataset stopped helping). A full seeded eval against real checkpoints found the model
> guessing wrong background colors — root-caused to captions never mentioning background color at
> all — and fixed by measuring each plate's real background from its pixels. Two follow-up
> fine-tuning rounds later, that fix is real but partial (6/11 on the eval sample) and stopped
> improving further, which is an accepted, understood limitation rather than a target for more
> compute. Genus-specific anatomy (e.g. a jellyfish prompt rendering as an abstract shape) is
> still open. See `CLAUDE.md` for the full detailed history, and the
> **[live showcase](https://hulashc.github.io/haeckel-lora/)** for an interactive look at the
> actual results — drag-to-compare sliders, a training-run scrubber, and a clickable
> real-vs-generated explorer.

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
      structural tag rather than a handful of coarse guessed buckets, plus a `black background` /
      `pale background` tag measured directly from each plate's own pixels (added after eval
      found the model guessing background color with no signal to learn it from — see below).
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
- [x] **Train (script)** — `scripts/train_lora.py`: rank-16 LoRA (via `peft`) on SD1.5's UNet
      cross-attention layers (`to_q`/`to_k`/`to_v`/`to_out.0`, plus an optional text-encoder LoRA
      via `--train_text_encoder`), designed and verified against the project's real 6GB VRAM
      ceiling (bf16 autocast, gradient checkpointing, batch size 1 + gradient accumulation, frozen
      fp16 VAE/text-encoder). Supports resuming from a checkpoint (`--resume_from`) and, since the
      background-color fix below, `--oversample_pale` for testing exposure-based fixes to a
      learned bias.
- [x] **Train (run)** — 124 epochs / 3,100 steps on the full 100-image dataset. Loss drops for
      roughly the first 50 epochs then plateaus flat — confirmed from the real per-epoch loss
      curve, not assumed. Two further short fine-tuning rounds (`outputs/checkpoints/
      bg-caption-run/`, `bg-caption-run-2/`) applied the background-color caption fix below.
- [x] **Eval** — `scripts/eval_lora.py`: two fixed, seeded prompt sets (real captions
      side-by-side with generations, and hand-written novel recombinations), run for real against
      multiple checkpoints, not just smoke-tested. Findings: style/layout/caption-position
      converge well and hold up; a real caption bug (no background-color signal at all) was found
      and fixed, landing at 6/11 background-color accuracy on the eval sample after two rounds of
      fine-tuning; genus-specific anatomy (e.g. jellyfish → abstract shape) is still open. Full
      writeup in `CLAUDE.md`; interactive results at the
      [live showcase](https://hulashc.github.io/haeckel-lora/).

A LoRA fine-tune of Stable Diffusion 1.5 on Ernst Haeckel's *Kunstformen der Natur* (1899-1904),
exploring what a small, tight visual domain teaches a generative model that a broad one can't.

## Why Haeckel

*Kunstformen der Natur* is public domain, ~100 plates, and every plate follows the same recipe:
one or more biological specimens, often radially symmetric, arranged in a multi-panel grid and
rendered as a lithograph with fine line work and saturated ink — on either a pale cream or a
stark black background, split roughly evenly across the set (52 black, 48 pale, measured directly
from the scans). The visual signature is strong enough that an output either looks like Haeckel
or obviously doesn't, which makes the model's progress genuinely inspectable rather than a
vibes-based "looks cool to me" call.

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
  raw/plates/            scraped plates + cover, untouched (gitignored, regenerate via
                         scripts/scrape_plates.py)
  processed/             plate_index.csv + plate_details.json (tracked), dataset/ (gitignored,
                         regenerate via scripts/build_captions.py)
scripts/                 every pipeline stage as a standalone CLI entry point: scrape, dedup,
                         caption, train (train_lora.py), eval (eval_lora.py) -- see each
                         script's own docstring for details and known gotchas
outputs/
  checkpoints/           LoRA adapter weights (gitignored)
  samples/               periodic validation samples from training (gitignored)
  eval/                  eval_lora.py contact sheets, one subdir per checkpoint (gitignored)
docs/                    live GitHub Pages showcase (github.io/haeckel-lora) -- interactive,
                         built on the real eval images and training logs, not illustrative mockups
```

(`src/` and `notebooks/` exist as empty leftover placeholders from initial scaffolding and were
never actually used — everything real lives in `scripts/`.)

## Related

Companion/follow-up project: **Haeckel's Mistakes** — training two adapters on the same hand
drawing the same radiolarians in two registers (1887 Challenger Report vs. 1899-1904 Kunstformen)
to isolate and measure the idealization between them. Depends on this repo's training pipeline;
not started.

## License

Code: MIT (see `LICENSE`). Training data (Haeckel's plates) is public domain; no license file is
needed for it, but see `data/raw/SOURCES.md` for provenance per plate.
