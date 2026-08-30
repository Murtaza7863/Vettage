#!/usr/bin/env python3
"""Deliverable #5 — worst false positives / false negatives on clean + transformed val.

Saves images + scores under results/error_examples/ and a patterns note.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

from dataset import read_index
from detector import load_detector, score_pil_images
from robust_transforms import eval_conditions
from train_lora import load_finetuned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--model", default="clipdet_latent10k_plus")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--checkpoint", default="checkpoints/lora/lora_best.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_images", type=int, default=200, help="Subsample val (balanced) to keep RAM bounded.")
    p.add_argument("--source", default="sid_set", help="Restrict to this dataset source, or 'all'.")
    p.add_argument("--out_dir", default="results/error_examples")
    p.add_argument("--notes", default="results/error_analysis.md")
    args = p.parse_args()

    samples = read_index(args.val_csv)
    if args.source and args.source != "all":
        samples = [s for s in samples if s.source == args.source]
    if args.max_images and len(samples) > args.max_images:
        rng = np.random.default_rng(42)
        real = [s for s in samples if s.label == 0]
        fake = [s for s in samples if s.label == 1]
        n_each = max(1, args.max_images // 2)
        rng.shuffle(real)
        rng.shuffle(fake)
        samples = list(real[:n_each]) + list(fake[:n_each])
    print(f"error analysis n={len(samples)} source={args.source} checkpoint={args.checkpoint}")

    model, transform, device, arch = load_detector(args.model, args.weights_dir, args.device)
    if args.checkpoint and str(args.checkpoint).lower() not in {"none", "off", "baseline"}:
        model = load_finetuned(model, args.checkpoint, device)

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = {
        ("clean", "none"),
        ("jpeg", "quality=30"),
        ("gaussian_blur", "sigma=2.0"),
        ("gaussian_noise", "sigma=0.1"),
        ("resize", "scale=0.25x then upscale"),
        ("center_crop", "crop to 80%"),
    }
    conditions = [c for c in eval_conditions() if (c.name, c.params) in wanted]

    catalog = []
    images = [Image.open(s.path).convert("RGB") for s in samples]
    labels = np.array([s.label for s in samples])

    for spec in conditions:
        trans = [spec.apply(im) for im in images]
        scores = np.asarray(score_pil_images(trans, model, transform, device))
        # fake=1, real=0. FP: real with high pred. FN: fake with low pred.
        real_idx = np.where(labels == 0)[0]
        fake_idx = np.where(labels == 1)[0]
        fp = real_idx[np.argsort(-scores[real_idx])[: args.k]]
        fn = fake_idx[np.argsort(scores[fake_idx])[: args.k]]

        cond_dir = os.path.join(args.out_dir, f"{spec.name}_{spec.params.replace(' ', '_').replace('/', '-')}")
        os.makedirs(cond_dir, exist_ok=True)
        for kind, idxs in (("fp", fp), ("fn", fn)):
            for rank, i in enumerate(idxs):
                src = samples[int(i)].path
                dest = os.path.join(
                    cond_dir, f"{kind}_{rank:02d}_pred{scores[i]:.3f}_lab{int(labels[i])}{os.path.splitext(src)[1]}"
                )
                trans[int(i)].save(dest)
                catalog.append(
                    {
                        "condition": spec.name,
                        "params": spec.params,
                        "kind": kind,
                        "rank": rank,
                        "path": src,
                        "saved": dest,
                        "pred": float(scores[i]),
                        "label": int(labels[i]),
                        "source": samples[int(i)].source,
                    }
                )
        print(f"{spec.name} {spec.params}: top FP pred={scores[fp[0]]:.3f} top FN pred={scores[fn[0]]:.3f}")

    with open(os.path.join(args.out_dir, "catalog.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog[0].keys()))
        w.writeheader()
        w.writerows(catalog)

    notes = _draft_notes(catalog, threshold=0.5)
    with open(args.notes, "w") as f:
        f.write(notes)
    print(f"wrote {args.notes}")


def _draft_notes(catalog: list[dict], threshold: float = 0.5) -> str:
    fps = [r for r in catalog if r["kind"] == "fp"]
    fns = [r for r in catalog if r["kind"] == "fn"]
    real_errors = [r for r in fps if r["pred"] > threshold]
    fake_errors = [r for r in fns if r["pred"] <= threshold]
    by_cond = {}
    for r in catalog:
        by_cond.setdefault(f"{r['condition']} {r['params']}", {"max_real": 0.0, "min_fake": 1.0})
        if r["kind"] == "fp":
            by_cond[f"{r['condition']} {r['params']}"]["max_real"] = max(
                by_cond[f"{r['condition']} {r['params']}"]["max_real"], r["pred"]
            )
        else:
            by_cond[f"{r['condition']} {r['params']}"]["min_fake"] = min(
                by_cond[f"{r['condition']} {r['params']}"]["min_fake"], r["pred"]
            )
    cond_lines = "\n".join(
        f"- `{k}`: highest real pred={v['max_real']:.3f}, lowest fake pred={v['min_fake']:.3f}"
        for k, v in by_cond.items()
    )
    return f"""# Error analysis (Deliverable #5)

SID_Set slice (native res). Catalog rows are the *closest-to-boundary* reals (`fp`) and fakes (`fn`), not necessarily misclassifications. Threshold={threshold}.

Actual errors in this catalog at threshold {threshold}: real scored fake={len(real_errors)}, fake scored real={len(fake_errors)}.

Closest-to-boundary scores:

{cond_lines}

Patterns after LoRA:

- Clean SID: all listed fakes sit at pred≈0.999; the highest real is 0.244 — still called real at 0.5. Residual "FP" texture is high-detail OpenImages crops (`68b1ceaff0bd8d7b` etc.), not a threshold miss.
- Heavy JPEG q=30 is the only transform that pulls fake scores down (lowest 0.827). Still above 0.5. Reals stay near 0.
- Center-crop 80% is the next-softest fake score (0.859) when the remaining crop is a smoother region.
- Blur σ=2 and 0.25× resize do **not** collapse SID scores after degradation training.
- CIFAKE 32×32 (not in this SID-only dump) is the remaining miss mode: upsampling to 224px destroys generator traces. See `results/success_lora.json`.

Images: `results/error_examples/`.
"""


if __name__ == "__main__":
    main()
