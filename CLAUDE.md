# Haeckel LoRA — working notes for Claude

Read this before doing anything else in this repo. `README.md` has the project pitch and plan;
this file is state — what's actually done, what broke, what's next.

## Status (as of 2026-09-02)

Data pipeline is done and verified. Training script exists and has now completed a real
(if short) training run with a genuinely trained checkpoint to show for it. Eval script exists
and is smoke-tested. Don't describe this project as further along than that — the current
checkpoint is a 50-epoch probe, not a finished adapter; nobody's looked at a full eval report
from it yet.

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
5. Training script — `scripts/train_lora.py`. Rank-16 LoRA (peft) on SD1.5's UNet cross-attention
   (`to_q`/`to_k`/`to_v`/`to_out.0`), 512x512, bf16 autocast (fp16 fallback via
   `--mixed_precision`), gradient checkpointing, batch size 1 + gradient accumulation, no
   Accelerate (plain torch, single GPU). Frozen VAE/text-encoder cast to fp16 permanently; UNet
   base stays fp32 with only the ~3.2M LoRA params trainable. Saves diffusers-standard LoRA
   safetensors (loadable via `pipe.load_lora_weights()`) to `outputs/checkpoints/`, optional
   periodic validation samples to `outputs/samples/` via `--validation_prompt`.
6. First real training run — 50 epochs / 1250 steps on the full 100-image dataset (~2h25m
   wall-clock; see "known issue: thermal throttling" below for why that's much slower than the
   ~1h25m the dry run predicted). Avg loss 0.167 → 0.146 (first 100 vs. last 100 steps) — real but
   modest, expected for a short probe run on inherently noisy diffusion MSE loss. Checkpoints at
   step-250/500/750/1000/1250 and `final/`, all in `outputs/checkpoints/`. Validation samples
   across the run show the model visibly converging onto Haeckel's actual visual signature (black
   background, radial multi-panel plate layout, fine cross-hatch line work) rather than staying
   close to generic SD1.5 output — a good sign the config works, not a claim the adapter is done.
   One caveat spotted in a step-1250 sample: a faint repeating watermark-like texture, which is a
   known SD1.5 *base model* artifact (stock-photo watermark contamination in its original LAION
   training data) — not something introduced by this project's own pipeline. Worth rechecking if
   it gets more prominent with further training.
7. Eval script — `scripts/eval_lora.py`. Loads a checkpoint's LoRA weights onto the base SD1.5
   pipeline and runs two fixed (seeded, reproducible across checkpoints) prompt sets: a
   **faithfulness set** (real training captions, evenly sampled across all 100 plates, each
   generated image placed side-by-side with its real plate) and a **novel set** (hand-written
   prompts recombining training vocabulary into combinations that don't exist in the dataset, per
   README's "produce specimens that don't exist but feel like they could"). Writes an
   `index.html` contact sheet per checkpoint to `outputs/eval/<checkpoint-name>/`. Smoke-tested
   with a 2-plate run; not yet run as a full evaluation against a real checkpoint.

**Known issue: thermal throttling.** Partway through the 50-epoch run the RTX 3060 **Laptop** GPU
hit 84°C and throttled hard — clock dropped from ~2100MHz to ~315MHz, step time went from ~6s to
~20s. It self-recovered later in the same run without intervention (clock back to ~1770MHz, ~3s/
step) and training wasn't corrupted by it, but expect wall-clock estimates to be unreliable on
long runs on this hardware, and don't be surprised if it happens again. If it gets worse (doesn't
recover, or GPU errors out), the fix is external cooling/airflow, not a code change — checkpoints
every N steps mean a throttle-triggered stall or a hard stop doesn't lose more than N steps of
progress.

**Not started:** a longer training run to actually converge (the 50-epoch run was explicitly a
probe to validate the config, not a finished adapter), running the eval script against a real
checkpoint and reading the resulting contact sheet.

## Environment

- `.venv/` exists in this repo (Windows, `./.venv/Scripts/python.exe`) with the **full**
  `requirements.txt` installed, including `torch==2.6.0+cu124`, `diffusers==0.40.0`,
  `transformers==5.16.1`, `accelerate==1.14.0`, `peft==0.20.0`, `safetensors==0.8.0`.
  `torch.cuda.is_available()` confirmed True, GPU name resolves correctly. If `.venv/` is ever
  missing again (it's gitignored, so it doesn't survive a fresh clone or a different machine —
  this happened once already, see git history if curious), recreate with `python -m venv .venv`
  then `pip install -r requirements.txt` plus the CUDA-specific torch index
  (`--index-url https://download.pytorch.org/whl/cu124`) — the CUDA wheel isn't on plain PyPI.
- SD1.5 base weights cache in the default HF hub cache
  (`~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5`, ~4.5GB) —
  first run downloads them, subsequent runs are local. Note **`runwayml/stable-diffusion-v1-5`
  was pulled from the Hub in 2023** — the script defaults to the `stable-diffusion-v1-5/`
  community-mirror org instead; don't "fix" it back to runwayml.
- GPU: NVIDIA RTX 3060 **Laptop** GPU, **6GB VRAM**, driver 591.86 / CUDA 13.1. This is tight for
  SD1.5 — `train_lora.py` accounts for it (fp16/bf16 mixed precision, gradient checkpointing,
  batch size 1 + gradient accumulation, torch SDPA attention, 512x512). Don't casually raise
  resolution or batch size without checking VRAM headroom first.
- Windows + PowerShell/Git Bash — the training script is plain Python/CLI (no Windows-only
  assumptions). `data/processed/dataset/` and `data/raw/plates/` are gitignored/regeneratable —
  both were empty at the start of this session (needed `scripts/scrape_plates.py` then
  `scripts/build_captions.py` to repopulate) and may be empty again on a fresh machine.

## Repo layout

```
data/
  raw/plates/            100 plate images + cover (gitignored, regenerate via scrape_plates.py)
  raw/*.json, *.pdf       intermediate pipeline artifacts (gitignored, regeneratable)
  processed/plate_index.csv     per-plate genus/order/German name (tracked in git)
  processed/plate_details.json  per-figure documentation, 1102 figures (tracked in git)
  processed/dataset/     image+caption .txt pairs for LoRA training (gitignored, regenerate via
                          build_captions.py)
scripts/                all pipeline + training scripts, each with a docstring explaining what it
                         does and why (several document real bugs found along the way — worth
                         reading); scripts/train_lora.py trains, scripts/eval_lora.py evaluates
outputs/checkpoints/    LoRA checkpoints (gitignored) — step-N/ + final/, from the 50-epoch probe run
outputs/samples/        periodic validation samples from training (gitignored)
outputs/eval/           eval_lora.py output, one subdir per checkpoint name (gitignored)
```

## Next step

Resume training from `outputs/checkpoints/final` (via `--resume_from`) for a longer real run —
150-250 more epochs is a reasonable next step now that the config's validated, not gospel. Watch
`outputs/checkpoints/loss_log.csv` and validation samples rather than picking a stopping point
blind, and keep an eye on GPU temp (see thermal throttling note above) on a run this long. Once
there's a checkpoint worth calling "trained," run `scripts/eval_lora.py` against it for real (not
just the 2-plate smoke test) and actually look at the resulting `index.html` contact sheet before
deciding whether it's done or needs more training / a different learning rate / more epochs.
