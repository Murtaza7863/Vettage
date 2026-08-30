#!/usr/bin/env python3
"""Labeled success-rate eval: accuracy / precision / recall / AUC on a CSV index.

pred > threshold => synthetic. Labels: 0 real, 1 fake.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import read_index
from detector import load_detector, score_images


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y)
    scores = np.asarray(scores)
    if len(y) == 0:
        nan = float("nan")
        return {
            "n": 0,
            "n_real": 0,
            "n_fake": 0,
            "accuracy": nan,
            "precision_fake": nan,
            "recall_fake": nan,
            "f1_fake": nan,
            "auc": nan,
            "true_real": 0,
            "false_fake": 0,
            "false_real": 0,
            "true_fake": 0,
            "real_success": nan,
            "fake_success": nan,
            "mean_pred_real": nan,
            "mean_pred_fake": nan,
        }
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int(len(y) - (y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "true_real": int(tn),
        "false_fake": int(fp),
        "false_real": int(fn),
        "true_fake": int(tp),
        "real_success": float(tn / max(int((y == 0).sum()), 1)),
        "fake_success": float(tp / max(int((y == 1).sum()), 1)),
        "mean_pred_real": float(scores[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_pred_fake": float(scores[y == 1].mean()) if (y == 1).any() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--model", default="clipdet_latent10k_plus")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out_json", default="results/success_rate.json")
    args = p.parse_args()

    samples = read_index(args.val_csv)
    if not samples:
        raise SystemExit(f"no samples in {args.val_csv}")

    model, transform, device, arch = load_detector(args.model, args.weights_dir, args.device)
    if args.checkpoint:
        from train_lora import load_finetuned

        model = load_finetuned(model, args.checkpoint, device)

    print(f"scoring {len(samples)} images  arch={arch} device={device} threshold={args.threshold}", flush=True)
    rows = score_images([s.path for s in samples], model, transform, device, batch_size=args.batch_size)
    scores = np.array([r["pred"] for r in rows])
    labels = np.array([s.label for s in samples])
    sources = [s.source for s in samples]

    overall = metrics(labels, scores, args.threshold)
    by_source = {}
    for src in sorted(set(sources)):
        mask = np.array([s == src for s in sources])
        by_source[src] = metrics(labels[mask], scores[mask], args.threshold)

    sweep = []
    for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        m = metrics(labels, scores, t)
        sid_mask = np.array([s == "sid_set" for s in sources])
        sid = metrics(labels[sid_mask], scores[sid_mask], t)
        sweep.append(
            {
                "threshold": t,
                "acc": m["accuracy"],
                "fake_ok": m["fake_success"],
                "real_ok": m["real_success"],
                "f1": m["f1_fake"],
                "sid_acc": sid["accuracy"],
                "sid_fake_ok": sid["fake_success"],
                "sid_real_ok": sid["real_success"],
            }
        )

    report = {
        "arch": arch,
        "n": len(samples),
        "threshold": args.threshold,
        "checkpoint": args.checkpoint,
        "overall": overall,
        "by_source": by_source,
        "threshold_sweep": sweep,
    }

    def line(title: str, m: dict) -> None:
        print(
            f"{title:12s}  n={m['n']:5d}  acc={m['accuracy']:.3f}  "
            f"AUC={m['auc']:.3f}  "
            f"real_ok={m['real_success']:.3f} ({m['true_real']}/{m['n_real']})  "
            f"fake_ok={m['fake_success']:.3f} ({m['true_fake']}/{m['n_fake']})  "
            f"F1={m['f1_fake']:.3f}  "
            f"mean_pred real={m['mean_pred_real']:.3f} fake={m['mean_pred_fake']:.3f}",
            flush=True,
        )

    print()
    line("OVERALL", overall)
    for src, m in by_source.items():
        line(src, m)
    print(
        f"confusion overall  TN={overall['true_real']} FP={overall['false_fake']}  "
        f"FN={overall['false_real']} TP={overall['true_fake']}",
        flush=True,
    )
    print("\nthreshold sweep (overall / SID_Set):")
    print(f"{'thr':>6}  {'acc':>6}  {'real_ok':>8}  {'fake_ok':>8}  {'F1':>6}  {'SID_acc':>8}  {'SID_fake':>8}  {'SID_real':>8}")
    for s in sweep:
        print(
            f"{s['threshold']:6.2f}  {s['acc']:6.3f}  {s['real_ok']:8.3f}  {s['fake_ok']:8.3f}  "
            f"{s['f1']:6.3f}  {s['sid_acc']:8.3f}  {s['sid_fake_ok']:8.3f}  {s['sid_real_ok']:8.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
