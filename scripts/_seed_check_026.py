"""One-off diagnostic: generate plate 026's (Carmaris, Trachomedusae -- a jellyfish) caption
across several seeds against a single checkpoint, to check whether the eval's single fixed seed
(45 = base 42 + prompt index 3) happened to land on an unrepresentative sample, or whether the
model's jellyfish-shape output is consistently non-jellyfish-like regardless of seed.

Reuses scripts/eval_lora.py's load_pipeline() and identical generation settings (512 res, 35
steps, guidance 7.5) so results are directly comparable to the real eval contact sheet.

Not part of the regular pipeline (leading underscore, not registered anywhere else) -- kept
rather than deleted because CLAUDE.md item 10 cites this script and its output
(outputs/eval/seed-check-026/) as the evidence trail for the plate-026 seed-sensitivity finding.
Feel free to delete both once that finding is no longer load-bearing for anything.

Usage:
    python scripts/_seed_check_026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from eval_lora import DEFAULT_PRETRAINED_MODEL, load_pipeline  # noqa: E402

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "caption-fix-run-1" / "final"
OUT_DIR = REPO_ROOT / "outputs" / "eval" / "seed-check-026"
CAPTION = (
    "haeckel_kunstformen, pale background, trachomedusae, carmaris, A large Geryonid medusa "
    "from Australia, natural size and named for the artist Adolf Giltsch, view from below "
    "shows six red leaf-shaped gonads forming a six-rayed rosette around the closed mouth, "
    "natural history lithograph illustration"
)
SEEDS = [45, 1, 7, 100, 2026, 9999]  # 45 = the actual seed the real eval used for this prompt


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {DEFAULT_PRETRAINED_MODEL} + LoRA from {CHECKPOINT_DIR}")
    pipeline = load_pipeline(DEFAULT_PRETRAINED_MODEL, CHECKPOINT_DIR, device)

    for seed in SEEDS:
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipeline(
            CAPTION,
            num_inference_steps=35,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
        ).images[0]
        out_path = OUT_DIR / f"seed-{seed}.png"
        image.save(out_path)
        print(f"seed {seed}: wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
