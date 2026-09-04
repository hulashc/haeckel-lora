# Haeckel LoRA — working notes for Claude

Read this before doing anything else in this repo. `README.md` has the project pitch and plan;
this file is state — what's actually done, what broke, what's next.

## Status (as of 2026-09-03)

Latest: a real caption-truncation bug was found and fixed (item 10) — the model was silently
never told about a plate's structural details (tentacle counts, symmetry, color words) on 16/100
plates because the truncation algorithm gave up after the first sentence that didn't fit a
77-token budget, even with token budget to spare. Fixed, retrained, evaluated: real, verifiable
shape-fidelity progress on the targeted plates (not a full fix), no regression on the novel set,
and a small *incidental* dip in background-color accuracy (5/11 vs. 6/11) consistent with the
already-documented "continued training reshuffles background correctness without a systematic
direction" noise pattern — not something to chase per the existing item 9 decision.

Data pipeline is done and verified. Training ran far longer than the original 50-epoch probe —
up to step 3100 (~epoch 124) — but loss plateaued around epoch 50 and never really moved after
that; the extra ~74 epochs didn't buy much on the loss curve. That run stopped without a clean
`final/` checkpoint (no error log found; still unexplained). A real eval against
`outputs/checkpoints/step-3000` found the background-color problem described below, which was
traced to a real root cause (captions never mentioned background color at all), fixed with a
25-epoch continuation (`bg-caption-run/final`) and then a further 40-epoch attempt with pale-
example oversampling (`bg-caption-run-2/final`) — net result across both: 6/11 faithfulness-set
plates get the right background, no further gain from the second round. **User decision: this is
good enough, stop chasing it** (see item 9). Don't describe this as a finished adapter — real
progress with a known, accepted limitation, not something validated as "done" in every respect.

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
6. Training run(s) — the documented 50-epoch/1250-step probe kept going past that point in what
   the loss-log step numbering shows was one continuous run, up to **step 3100 / epoch ~124**,
   without CLAUDE.md being updated to say so. Checkpoints saved at step-500/1000/1500/2000/2500/
   3000 in `outputs/checkpoints/` (no `final/` — the run stopped without a clean finish and left
   no error log; as of 2026-09-02 afternoon the GPU is idle and nothing is training). Loss:
   0.168 avg (steps 1-100) → 0.144 avg (steps 1251-1350, i.e. right at the old documented
   endpoint) → 0.147 avg (steps 3001-3100, the very end) — **it plateaued around epoch 50 and the
   next 74 epochs made essentially no further difference to the loss curve.** On a 100-image
   dataset that reads as expected saturation, not a bug. One caveat spotted in an earlier
   step-1250 sample: a faint repeating watermark-like texture, a known SD1.5 *base model* artifact
   (LAION stock-photo watermark contamination), not introduced by this project's pipeline — worth
   rechecking if it gets more prominent in any future run.
7. Eval script — `scripts/eval_lora.py`. Loads a checkpoint's LoRA weights onto the base SD1.5
   pipeline and runs two fixed (seeded, reproducible across checkpoints) prompt sets: a
   **faithfulness set** (real training captions, evenly sampled across all 100 plates, each
   generated image placed side-by-side with its real plate) and a **novel set** (hand-written
   prompts recombining training vocabulary into combinations that don't exist in the dataset, per
   README's "produce specimens that don't exist but feel like they could"). Writes an
   `index.html` contact sheet per checkpoint to `outputs/eval/<checkpoint-name>/`. **Run for real**
   against `outputs/checkpoints/step-3000` on 2026-09-02 (default 12 faithfulness pairs + all 6
   novel prompts) — results at `outputs/eval/step-3000/index.html`. Findings:
   - **Style/genre is solidly learned**: every faithfulness-set output is unmistakably a Haeckel
     plate — radial multi-panel grid layout, fine cross-hatch linework, pseudo-Latin/German
     caption text in the right place. Best matches (Chiroptera bat faces, Filicinae fern
     rainforest scene) are genuinely close to their real counterparts.
   - **Per-plate fidelity is inconsistent**: background color is frequently wrong or inverted
     (e.g. the real black-bg Diatomea plate came out cream-bg, and vice versa on Phaeodaria), and
     genus-specific morphology sometimes drifts hard (the Carmaris jellyfish prompt produced an
     abstract birdcage-like shape rather than a jellyfish; trunkfish shape read correctly but in
     the wrong color palette). This lines up with the plateaued loss — likely a data/caption
     signal-density issue (100 images, lots of distinct genera) rather than something more epochs
     on the same data would fix.
   - **Novel set is the encouraging part**: several of the 6 recombined prompts (a green-bg
     radiolarian grid, a black-bg spiky radiolarian cluster, a coral/sponge grid) are convincing
     "specimens that don't exist but feel like they could" — the project's actual stated goal.
     One (brittle-star × siphonophore hybrid) came out as a murkier, less legible tangle — the
     more conceptually blended the prompt, the less coherent the result.
8. Background-color caption fix — `scripts/build_captions.py` now has `detect_background()`,
   which measures each plate's real illustration background (median grayscale luminance of a
   12%-88% central crop, excluding the cream page margin/caption text around every plate) and
   adds a `"black background"` / `"pale background"` tag to the caption. This is *measured from
   pixels*, never guessed/LLM-generated, per this project's zero-fabrication caption policy. All
   100 plates split cleanly across a wide gap (roughly <=160 vs. >=180 out of 255) into 52 black /
   48 pale — added because the step-3000 eval (above) showed the model guessing the wrong
   background color, and the pre-fix captions never mentioned background color at all, so there
   was no textual signal to learn it from. `data/processed/captions.jsonl` and
   `data/processed/dataset/` have been regenerated with the new tag; regenerate again if you ever
   rerun `build_captions.py` on a fresh clone.

   Ran a short, cheap continuation (not a from-scratch retrain): resumed both UNet and
   text-encoder LoRA adapters from `step-3000`, 25 epochs / 625 steps, lower LR (5e-5, below
   where the old run's cosine schedule had decayed to) since this is a fine-tuning tail for one
   new signal, not fresh learning. Output deliberately went to a **separate**
   `outputs/checkpoints/bg-caption-run/` dir (not the shared `outputs/checkpoints/` root) so it
   couldn't collide with or overwrite the original run's step-N checkpoints or `loss_log.csv`.
   Finished cleanly with a proper `final/` checkpoint (unlike the original run). Re-ran
   `scripts/eval_lora.py` against it with `--eval_dir outputs/eval/bg-caption-run` — results at
   `outputs/eval/bg-caption-run-final/index.html` (renamed after the fact; see item 9's near-miss
   below -- **always pass a distinct `--eval_dir` per run**, not the default, since checkpoint
   dirs are both literally named `final` and `eval_lora.py` only namespaces output by checkpoint
   dir *name*, not its full path).

   **Result: partial fix, confirmed by direct before/after comparison, not fully generalized.**
   - Fixed: plate 084 (Diatomea, tagged black) now generates correctly black-bg, matching the
     real plate almost exactly. Plate 001 (Phaeodaria, tagged pale) now generates correctly
     pale-bg, also matching well. Both were wrong (inverted) in the step-3000 eval.
   - Still wrong: plate 042 (Ostracion, tagged pale) still generates a dark teal background.
     Plate 026 (Carmaris, tagged pale) still generates a black/dark-green background. Same two
     failure cases as before, caption tag notwithstanding.
   - Novel set: re-checked 2 of the 6 prompts, no quality regression from the extra fine-tuning.
   - Working theory for the split: 25 epochs may be enough to shift background-color association
     for some genus/order pairings but not others -- possibly plates where the note text or
     genus-specific visual signal is especially strong (jellyfish tentacles, fish body shape)
     have that visual prior dominating over the new background token. Not confirmed, just the
     leading hypothesis for whoever picks this up next.
9. Tried `--oversample_pale 2` (new `train_lora.py` flag, duplicates pale-background examples
   per epoch) for 40 more epochs from `bg-caption-run/final`, to directly target the black-bias
   asymmetry in item 8. Result: **no net improvement** -- re-ran the full eval
   (`outputs/eval/bg-caption-run-2/final/index.html`) and the faithfulness ratio was identical,
   6/11, both before and after. One plate flipped wrong->right (026 Carmaris), another flipped
   right->wrong (001 Phaeodaria); net zero. Since eval prompts are seeded/reproducible, this is a
   genuine shift, not sampling noise -- it just means "more pale exposure" isn't the actual lever.
   The teal/dark-teal background specifically (017, 042, 059 -- all aquatic Siphonophorae/
   Ostraciontes subjects) was a *consistent* failure across both runs, suggesting the model has
   learned "aquatic subject -> teal" as a stronger association than the literal caption tag, i.e.
   genus content overriding the background instruction on those specific plates, not a general
   undertraining-of-background-color problem. (Near-miss: this eval was first launched with the
   default `--eval_dir`, which would have overwritten `outputs/eval/final/` from item 8 since
   both checkpoints are named `final` -- caught and stopped before it overwrote anything, output
   renamed to `outputs/eval/bg-caption-run-final/` and the rerun used `--eval_dir
   outputs/eval/bg-caption-run-2` instead. Always use a distinct `--eval_dir` per run.)

   **Decision (user, 2026-09-03): stop chasing this.** Background-color fidelity is explicitly
   not a priority -- "more colors and all" is fine. Don't spend more training compute on it.
   `bg-caption-run/final` and `bg-caption-run-2/final` are both acceptable checkpoints as-is;
   no further background-color work is planned. If anyone revisits it anyway: check the caption's
   tag *position* (currently right after the trigger word) or a bigger LR shock rather than more
   epochs at 5e-5, since two rounds of "more epochs, same approach" produced zero net gain.
10. **Found and fixed a real caption-truncation bug** (2026-09-03), prompted by the user reporting
    generated images "aren't great at all" and specifically pointing at the showcase's before/after
    slider for plate 026 (Carmaris, Trachomedusae -- a jellyfish), whose generated version was an
    abstract cage shape, not a jellyfish. First ruled out a display/CSS bug (the slider's real/
    generated image pairs are pixel-identical, `object-fit: cover` applied identically to both
    layers -- confirmed via a dedicated investigation, no scaling mismatch in the UI). That
    confirmed it was a content-fidelity problem, which led to inspecting `scripts/
    build_captions.py`'s 77-token truncation algorithm (`_greedy_chain`/`fit_caption`) against the
    real source notes in `plate_details.json`.

    **The bug**: the algorithm stopped permanently at the first clause of a plate's note that
    didn't fit the token budget, never trying later clauses or comma-segments even with plenty of
    budget left over. Plate 026's caption used only 47/77 tokens and silently dropped the *only*
    clause mentioning any jellyfish structure at all: "...view from below shows six red
    leaf-shaped gonads forming a six-rayed rosette around the closed mouth, with twelve auditory
    vesicles and twelve tentacles at the umbrella margin." The model was never told about
    tentacles or radial symmetry for this plate. Same pattern on plates 051, 059 (also dropped an
    explicit color word, "pale-bluish," on a plate that's one of item 9's persistent teal-
    background failures), and 092.

    **The fix**: rewrote `_greedy_chain`/`fit_caption` to skip a clause/comma-segment that doesn't
    fit and keep trying later ones, instead of stopping dead -- always real text, original order,
    no fabrication or reordering (same discipline as the background-color tag). Caught and fixed a
    self-introduced punctuation artifact along the way (an over-eager version stripped legitimate
    mid-sentence punctuation, e.g. turning "...lattice shell; four equatorial..." into "...lattice
    shell four equatorial..." -- fixed by only stripping punctuation at the specific comma-joiner
    boundary that actually needed it, not universally). Verified end-to-end: regenerating
    `data/processed/captions.jsonl` changes exactly 16/100 captions, zero regressions (no caption
    gets shorter/worse), 042 and 076 confirmed unaffected (their full notes already fit under 77
    tokens -- a separate, out-of-scope problem: the genus-matched figure just isn't very
    descriptive; would need picking a different sibling figure, not a truncation fix).

    **Retrained**: continued from `bg-caption-run-2/final`, 30 epochs / 750 steps, LR 7e-5 (between
    the 5e-5 fine-tuning-tail rate and the original 1e-4), `--train_text_encoder` kept on since
    this is new *text* content the text encoder needs room to move on -> `outputs/checkpoints/
    caption-fix-run-1/final`. Clean finish with a proper `final/` checkpoint. Full eval at
    `outputs/eval/caption-fix-run-1/`.

    **Honest result -- real but partial, not a full fix**:
    - Plate 026: the new "six-rayed rosette around the closed mouth" content is now clearly and
      consistently visible in generated output (a direct, traceable win -- content that was
      silently dropped now visibly influences the result). The eval's single fixed seed (45)
      landed on an ornamental medallion, not a bell-with-tentacles, which looked like a
      seed-sensitivity concern -- **resolved by a follow-up 6-seed check** (`scripts/
      _seed_check_026.py`, `outputs/eval/seed-check-026/`, same checkpoint/prompt/settings as the
      real eval, seeds 45/1/7/100/2026/9999): 4 of 6 seeds show clear jellyfish bell-and-tentacle
      forms (domed caps, dangling wavy fringe) in multiple panels, and **6 of 6 show the rosette
      motif consistently**. The eval's fixed seed (45) turned out to be one of only two
      (with seed 7) that land on the more abstract floral-medallion look -- i.e. the real eval
      contact sheet happened to sample an unrepresentative case. The fix works meaningfully better
      than the single-seed eval suggested; it just isn't visible from one contact-sheet image.
    - Plates 051, 059, 092: modest, consistent shape-detail enrichment tracking their new caption
      content (capsule texture, stacked bells, fern crown structure). No dramatic transformation,
      no regression.
    - Novel set: no regression vs. `bg-caption-run-2`; one prompt (`discomedusae, "aurelia nova"`)
      now shows an actual jellyfish-bell shape -- a plausible sign of some transferable jellyfish-
      structure concept, not just plate-026-specific memorization.
    - Background-color: **incidentally dipped**, 5/11 vs. the prior 6/11 -- plates 001 and 026
      both flipped correct->wrong, none flipped the other way. Plate 001's caption didn't even
      change in this run, so this is the same "continued training reshuffles background
      correctness without a systematic direction" noise pattern already documented in item 9, not
      a new problem, and not something this run was targeting. Checked per plan, not chased.

**Known issue: thermal throttling.** Partway through the 50-epoch run the RTX 3060 **Laptop** GPU
hit 84°C and throttled hard — clock dropped from ~2100MHz to ~315MHz, step time went from ~6s to
~20s. It self-recovered later in the same run without intervention (clock back to ~1770MHz, ~3s/
step) and training wasn't corrupted by it, but expect wall-clock estimates to be unreliable on
long runs on this hardware, and don't be surprised if it happens again. If it gets worse (doesn't
recover, or GPU errors out), the fix is external cooling/airflow, not a code change — checkpoints
every N steps mean a throttle-triggered stall or a hard stop doesn't lose more than N steps of
progress.

**Not started / open questions:** why the run stopped around step 3100 without a `final/`
checkpoint or any error log (nobody's investigated — it may just have been manually interrupted
in a prior session); the still out-of-scope figure-selection issue on plates 042/076 (item 10).
(Plate 026's seed-sensitivity question from item 10 is resolved — see the 6-seed check
documented there; the caption fix genuinely helped, the eval's single seed just wasn't
representative.)

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
outputs/checkpoints/    LoRA checkpoints (gitignored) — step-N/ through step-3000/, no final/ (see
                        status above); loss_log.csv covers the whole run, steps 1-3100.
                        bg-caption-run/, bg-caption-run-2/, caption-fix-run-1/ are separate
                        continuations (own step-N/, final/, loss_log.csv each) -- see items 8-10
                        above. caption-fix-run-1/final is the current best checkpoint.
outputs/samples/        periodic validation samples from training (gitignored); bg-caption-run/,
                        bg-caption-run-2/, caption-fix-run-1/ subdirs for those continuation runs
outputs/eval/           eval_lora.py output (gitignored) -- step-3000/ (pre-fix baseline),
                        bg-caption-run-final/ (25-epoch fix, renamed from a `final`-name
                        collision, see item 9), bg-caption-run-2/final/ (pale-oversample attempt),
                        caption-fix-run-1/final/ (truncation-bug fix, item 10 -- current best).
                        Always pass --eval_dir explicitly per run; the default namespaces only by
                        checkpoint dir *name*, and multiple checkpoints are named `final`.
```

## Next step

Background-color fidelity is a closed topic per user decision (item 9 above) — don't reopen it
without being asked; the item-10 dip is expected noise, not a regression to fix. Current best
checkpoint is `caption-fix-run-1/final`. Reasonable next moves:
1. The showcase (`docs/`) still reflects `bg-caption-run-2/final`, not `caption-fix-run-1/final`
   — it was last updated before this fix landed, so it neither shows the improved plate-026
   result nor mentions the truncation bug. Worth refreshing now that the 6-seed check confirms
   the fix is real, so the showcase doesn't undersell it.
2. The out-of-scope figure-selection issue from item 10 (plates 042, 076 have thin captions even
   at full length because the genus-matched figure isn't very descriptive) — would need
   `select_note` to consider sibling figures on the same plate, not just the genus match.
3. Consider applying the same 6-seed-check discipline to a few other faithfulness plates before
   trusting any single-seed eval result at face value going forward — item 10's experience is
   that one fixed seed can land on an unrepresentative outcome even when the underlying fix
   works; this project's `eval_lora.py` has always used one seed per prompt for reproducibility,
   which is good for comparing checkpoints but apparently not sufficient on its own for judging
   whether a fix "worked" on a single plate.
4. Otherwise, treat the adapter as usable for its stated goal (novel Haeckel-style generation)
   rather than continuing to chase per-plate faithfulness metrics — the novel set has
   consistently been the stronger, more relevant result across every eval run so far.
