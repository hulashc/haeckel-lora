# Haeckel LoRA — working notes for Claude

Read this before doing anything else in this repo. `README.md` has the project pitch and plan;
this file is state — what's actually done, what broke, what's next.

## Status (as of 2026-09-01)

Data pipeline is done and verified. Training hasn't started — no run, no checkpoint, nothing in
`outputs/`. Don't describe this project as further along than that.

**Done:**
1. Scrape — 100 plates + cover, `data/raw/plates/` (`scripts/scrape_plates.py`)
2. Dedup — 0 near-duplicates found (`scripts/dedup_plates.py`)
3. Caption — `data/processed/plate_index.csv` + `data/processed/dataset/` (image+caption pairs
   for training), via `scripts/build_captions.py`
4. Full per-figure documentation — `data/processed/plate_details.json`, 1102 figures across 100
   plates, sourced from Haeckel's own explanatory text (a *different* archive.org item than the
   plates). Two plates have honest gaps (85 empty, 63/90 missing early figures) rather than
   invented content — **read `data/processed/README.md` before touching this pipeline again**,
   it documents several real bugs (OCR page-boundary mismatches) and two actual LLM-fabrication
   incidents that were caught and fixed. The lesson generalizes: verify generated output traces
   back to real source text before trusting it.

**Not started:** training script, inference/eval script, any actual training run.

## Environment

- `.venv/` exists in this repo (Windows, `./.venv/Scripts/python.exe`), has `requests`,
  `Pillow`, `imagehash`, `pymupdf` installed. Does **not** yet have `torch`/`diffusers`/
  `transformers`/`accelerate`/`peft`/`safetensors` — those are in `requirements.txt` but not
  installed, since they're heavy and weren't needed until training.
- GPU: NVIDIA RTX 3060, **6GB VRAM**. This is tight for SD1.5 — the training script needs to be
  designed for it from the start (fp16/bf16 mixed precision, gradient checkpointing, small batch
  size + gradient accumulation, xformers or PyTorch SDPA attention, probably 512x512 resolution
  not higher). Don't write a training script that assumes a datacenter GPU and then discover it
  OOMs; account for the real constraint up front.
- Windows + PowerShell/Git Bash — the training script should still be plain Python/CLI (no
  Windows-only assumptions), just know `nvidia-smi` confirms the driver + CUDA 13.1 are there.

## Repo layout

```
data/
  raw/plates/            100 plate images + cover (gitignored, regenerate via scrape_plates.py)
  raw/*.json, *.pdf       intermediate pipeline artifacts (gitignored, regeneratable)
  processed/plate_index.csv     per-plate genus/order/German name (tracked in git)
  processed/plate_details.json  per-figure documentation, 1102 figures (tracked in git)
  processed/dataset/     image+caption .txt pairs for LoRA training (gitignored, regenerate via
                          build_captions.py)
scripts/                all pipeline scripts, each with a docstring explaining what it does and
                         why (several document real bugs found along the way — worth reading)
outputs/checkpoints/    empty — LoRA weights go here once training exists
outputs/samples/        empty — eval/inference samples go here
```

## Next step

Write the training script: rank-16 LoRA adapter on SD1.5's UNet cross-attention layers, via
`diffusers` + `peft`, training on `data/processed/dataset/`. See "Plan" and "Inference" in
`README.md` for the intended design (LoRA over full fine-tune, why, guidance scale, DDIM steps).
Given the 6GB VRAM ceiling, plan the training config (resolution, batch size, precision, attention
backend) explicitly before writing the loop, not as an afterthought once it fails to fit.
