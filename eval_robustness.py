#!/usr/bin/env python3
"""Robustness eval harness — Track 5 transforms, one condition each.

Runs the current detector on clean val images and on each transformed copy.
Writes ROC-AUC per condition plus a clean-vs-transformed summary table
(Deliverable #4) to results/robustness_summary.csv and a bar chart.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from dataset import read_index
from detector import load_detector, score_pil_images
from robust_transforms import eval_conditions


def load_val_rows(index_csv: str, max_images: int | None, seed: int = 42):
    samples = read_index(index_csv)
    if max_images is not None and len(samples) > max_images:
        rng = np.random.default_rng(seed)
        # keep class balance
        real = [s for s in samples if s.label == 0]
        fake = [s for s in samples if s.label == 1]
        n_each = max(1, max_images // 2)
        rng.shuffle(real)
        rng.shuffle(fake)
        samples = list(real[:n_each]) + list(fake[:n_each])
    return samples


def auc_or_nan(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--model", default="clipdet_latent10k_plus")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--checkpoint", default=None, help="Optional LoRA/finetuned head .pt")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--out_csv", default="results/robustness_summary.csv")
    p.add_argument("--out_png", default="results/robustness_summary.png")
    p.add_argument("--tag", default="baseline")
    args = p.parse_args()

    samples = load_val_rows(args.val_csv, args.max_images)
    if not samples:
        raise SystemExit(f"no val samples in {args.val_csv}")
    labels = [s.label for s in samples]
    images = [Image.open(s.path).convert("RGB") for s in samples]

    model, transform, device, arch = load_detector(args.model, args.weights_dir, args.device)
    if args.checkpoint:
        import torch
        from train_lora import load_finetuned

        model = load_finetuned(model, args.checkpoint, device)

    conditions = eval_conditions()
    rows = []
    clean_auc = None
    sources = [s.source for s in samples]
    print(f"eval {len(samples)} images, {len(conditions)} conditions, arch={arch} device={device}")

    def masked_auc(y, s, mask):
        y = np.asarray(y)[mask]
        s = np.asarray(s)[mask]
        return auc_or_nan(y, s)

    for spec in conditions:
        transformed = [spec.apply(im) for im in images]
        scores = score_pil_images(transformed, model, transform, device, batch_size=args.batch_size)
        auc = auc_or_nan(labels, scores)
        sid_mask = np.array([src == "sid_set" for src in sources])
        cifake_mask = np.array([src == "cifake" for src in sources])
        auc_sid = masked_auc(labels, scores, sid_mask) if sid_mask.any() else float("nan")
        auc_cifake = masked_auc(labels, scores, cifake_mask) if cifake_mask.any() else float("nan")
        if spec.name == "clean":
            clean_auc = auc
        delta = None if clean_auc is None or np.isnan(auc) else auc - clean_auc
        rec = {
            "tag": args.tag,
            "condition": spec.name,
            "params": spec.params,
            "n": len(samples),
            "n_sid": int(sid_mask.sum()),
            "n_cifake": int(cifake_mask.sum()),
            "auc": auc,
            "auc_sid_set": auc_sid,
            "auc_cifake": auc_cifake,
            "clean_auc": clean_auc,
            "delta_vs_clean": delta,
            "mean_pred": float(np.mean(scores)),
        }
        rows.append(rec)
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        print(
            f"  {spec.name:16s} {spec.params:24s} "
            f"AUC={auc:.4f} SID={auc_sid:.4f} CIFAKE={auc_cifake:.4f}  Δclean={delta_s}"
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels_x = [f"{r['condition']}\n{r['params']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - width / 2, [r["auc"] for r in rows], width, label="all val", color="#e76f51")
    ax.bar(x + width / 2, [r["auc_sid_set"] for r in rows], width, label="SID_Set only", color="#2a9d8f")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Robustness summary ({args.tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    plt.close(fig)

    sidecar = os.path.splitext(args.out_csv)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
