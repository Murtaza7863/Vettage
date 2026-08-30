#!/usr/bin/env python3
"""Train a linear head for tiny images (CIFAKE 32×32) on frozen LoRA CLIP features.

SID LoRA already saturates native-res detection. CIFAKE is a different problem
(32×32 upsampled to 224) so we keep a second 1025-param head and gate it on
min(width, height) at inference. CLIP stays frozen; only this head is fit.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import read_index
from detector import load_detector
from robust_transforms import jpeg_compress
from train_lora import attach_tiny_head, load_finetuned, save_finetuned, set_mode


@torch.no_grad()
def extract_features(samples, model, transform, device, batch_size: int, jpeg_quality: int | None):
    feats, labels, sources = [], [], []
    batch, y_batch, src_batch = [], [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch, 0).to(device)
        f = model.forward_features(x).detach().cpu().numpy()
        feats.append(f)
        labels.extend(y_batch)
        sources.extend(src_batch)
        batch.clear()
        y_batch.clear()
        src_batch.clear()

    for s in samples:
        img = Image.open(s.path).convert("RGB")
        if jpeg_quality is not None:
            img = jpeg_compress(img, jpeg_quality)
        batch.append(transform(img))
        y_batch.append(int(s.label))
        src_batch.append(s.source)
        if len(batch) >= batch_size:
            flush()
    flush()
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, model.num_features), dtype=np.float32)
    return X.astype(np.float32), np.asarray(labels, dtype=np.int64), sources


def fit_probe(X, y, Xv, yv):
    best = None
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X, y)
        scores = clf.decision_function(Xv)
        auc = float(roc_auc_score(yv, scores)) if len(np.unique(yv)) > 1 else float("nan")
        acc = float(accuracy_score(yv, (scores > 0).astype(int)))
        rec = {"C": C, "val_auc": auc, "val_acc": acc}
        print(f"  C={C:g}  val_auc={auc:.4f}  val_acc={acc:.4f}", flush=True)
        if best is None or auc > best["val_auc"]:
            best = {**rec, "clf": clf}
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--checkpoint", default="checkpoints/lora/lora_best.pt")
    p.add_argument("--out_ckpt", default="checkpoints/lora/lora_best.pt")
    p.add_argument("--model", default="clipdet_latent10k_plus")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tiny_max_side", type=int, default=64)
    p.add_argument("--source", default="cifake")
    args = p.parse_args()

    train_s = [s for s in read_index(args.train_csv) if s.source == args.source]
    val_s = [s for s in read_index(args.val_csv) if s.source == args.source]
    if not train_s or not val_s:
        raise SystemExit(f"no {args.source} rows in indexes")

    model, transform, device, arch = load_detector(args.model, args.weights_dir, args.device)
    model = load_finetuned(model, args.checkpoint, device)
    set_mode(model, False)
    print(f"extract {len(train_s)} train + {len(val_s)} val  arch={arch} device={device}", flush=True)

    feat_cache = os.path.join(os.path.dirname(args.out_ckpt) or ".", "cifake_feats.npz")
    if os.path.isfile(feat_cache):
        z = np.load(feat_cache)
        Xtr, ytr, Xv, yv = z["Xtr"], z["ytr"], z["Xv"], z["yv"]
        print(f"loaded cached features {feat_cache} train={Xtr.shape} val={Xv.shape}", flush=True)
    else:
        Xtr, ytr, _ = extract_features(train_s, model, transform, device, args.batch_size, None)
        Xj, yj, _ = extract_features(train_s, model, transform, device, args.batch_size, 50)
        Xtr = np.concatenate([Xtr, Xj], axis=0)
        ytr = np.concatenate([ytr, yj], axis=0)
        Xv, yv, _ = extract_features(val_s, model, transform, device, args.batch_size, None)
        os.makedirs(os.path.dirname(feat_cache) or ".", exist_ok=True)
        np.savez(feat_cache, Xtr=Xtr, ytr=ytr, Xv=Xv, yv=yv)
        print(f"features train={Xtr.shape} val={Xv.shape} (cached {feat_cache})", flush=True)

    best = fit_probe(Xtr, ytr, Xv, yv)
    clf = best["clf"]
    print(f"best C={best['C']} val_auc={best['val_auc']:.4f} val_acc={best['val_acc']:.4f}", flush=True)

    attach_tiny_head(model)
    device_w = model.fc.weight.device
    model.fc_tiny.to(device_w)
    with torch.no_grad():
        model.fc_tiny.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32, device=device_w))
        model.fc_tiny.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32, device=device_w))
    model.tiny_max_side = args.tiny_max_side

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    extra = dict(ckpt.get("extra") or {})
    extra.update(
        {
            "tiny_max_side": args.tiny_max_side,
            "tiny_source": args.source,
            "tiny_val_auc": best["val_auc"],
            "tiny_val_acc": best["val_acc"],
            "tiny_C": best["C"],
            "tiny_n_train": int(len(ytr)),
        }
    )
    save_finetuned(model, args.out_ckpt, extra=extra)
    sidecar = os.path.splitext(args.out_ckpt)[0] + "_tiny.json"
    with open(sidecar, "w") as f:
        json.dump({k: extra[k] for k in extra if k != "history"}, f, indent=2, default=str)
    print(f"wrote {args.out_ckpt} (fc_tiny + LoRA SID head)", flush=True)


if __name__ == "__main__":
    main()
