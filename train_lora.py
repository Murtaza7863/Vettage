#!/usr/bin/env python3
"""LoRA-tune the CLIP ViT detector on transform-augmented SID_Set + CIFAKE.

Freezes OpenCLIP base weights. Trains LoRA adapters on the vision encoder plus
the linear classifier head. Applies Track 5 degradations to a portion of each
batch so the model sees JPEG / blur / resize / noise / jitter / crop at train time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import read_index
from detector import build_transform, count_loaded_params, get_config, pick_device
from networks import create_architecture, load_weights
from robust_transforms import random_train_degradation


class AugmentedRealFake(Dataset):
    def __init__(self, index_csv: str, transform, degrade_p: float = 0.5):
        self.samples = read_index(index_csv)
        self.transform = transform
        self.degrade_p = degrade_p

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        # Augment at model scale, not native 1024px — otherwise JPEG/blur/noise
        # explode RAM and the network never sees those extra pixels anyway.
        w, h = img.size
        short = min(w, h)
        if short > 256:
            scale = 256 / short
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        img = random_train_degradation(img, p=self.degrade_p)
        return self.transform(img), torch.tensor([s.label], dtype=torch.float32)


# OpenCLIP attention is nn.MultiheadAttention, which consumes out_proj.weight
# through F.multi_head_attention_forward instead of calling the submodule. A LoRA
# adapter on out_proj would attach but never affect the forward pass, so we only
# target the MLP projections, which are invoked as real modules.
LORA_TARGET_MODULES = ["c_fc", "c_proj"]


def attach_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> dict:
    """Freeze CLIP, wrap the ViT with LoRA, train adapters + classifier head."""
    visual = model.bb[0].visual
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model.bb[0].visual = get_peft_model(visual, cfg)
    for p in model.bb[0].parameters():
        p.requires_grad = False
    n_lora = 0
    for name, p in model.bb[0].visual.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            n_lora += p.numel()
    for p in model.fc.parameters():
        p.requires_grad = True
    model.tune_visual = True
    trainable = sum(p.numel() for p in _trainable(model))
    return {
        "lora_params": n_lora,
        "head_params": sum(p.numel() for p in model.fc.parameters()),
        "trainable_total": trainable,
        "lora_r": r,
        "lora_alpha": alpha,
        "target_modules": list(LORA_TARGET_MODULES),
    }


def _auc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _auc_subset(labels, scores, sources, want: str) -> float:
    idx = [i for i, s in enumerate(sources) if s == want]
    if not idx:
        return float("nan")
    return _auc([labels[i] for i in idx], [scores[i] for i in idx])


def set_mode(model, training: bool) -> None:
    """Drive both the head and the (unregistered) CLIP backbone into one mode.

    model.bb is a plain list, so model.train()/eval() never reaches the
    backbone. Without this, LoRA dropout stays active during validation.
    """
    model.train(training)
    visual = model.bb[0].visual
    visual.train(training) if training else visual.eval()


def _trainable(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p
    if hasattr(model, "bb"):
        for p in model.bb[0].parameters():
            if p.requires_grad:
                yield p


def unique_trainable(model):
    seen = set()
    params = []
    for p in _trainable(model):
        if id(p) in seen:
            continue
        seen.add(id(p))
        params.append(p)
    return params


def save_finetuned(model, path: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "fc": model.fc.state_dict(),
        "lora": get_peft_model_state_dict(model.bb[0].visual),
        "extra": extra or {},
    }
    if getattr(model, "fc_tiny", None) is not None:
        payload["fc_tiny"] = model.fc_tiny.state_dict()
    torch.save(payload, path)


def attach_tiny_head(model):
    from networks.resnet_mod import ChannelLinear

    current = getattr(model, "fc_tiny", None)
    if isinstance(current, torch.nn.Module):
        return model
    head = ChannelLinear(model.num_features, 1)
    torch.nn.init.normal_(head.weight.data, 0.0, 0.02)
    model.fc_tiny = head
    return model


def load_finetuned(model, path: str, device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    extra = ckpt.get("extra") or {}
    if not getattr(model, "tune_visual", False) or not hasattr(model.bb[0].visual, "peft_config"):
        attach_lora(
            model,
            r=int(extra.get("lora_r", 8)),
            alpha=int(extra.get("lora_alpha", 16)),
        )
    model.fc.load_state_dict(ckpt["fc"])
    set_peft_model_state_dict(model.bb[0].visual, ckpt["lora"])
    model.tune_visual = True
    if ckpt.get("fc_tiny") is not None:
        attach_tiny_head(model)
        model.fc_tiny.load_state_dict(ckpt["fc_tiny"])
        model.tiny_max_side = int(extra.get("tiny_max_side", 64))
    else:
        model.tiny_max_side = 0
    model = model.to(device)
    set_mode(model, False)
    return model


def _reject_main_checkpoint_dir(out_dir: str) -> None:
    """generalization-v2 must never overwrite the shipped main LoRA."""
    abs_out = os.path.abspath(out_dir)
    forbidden = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints", "lora"))
    if abs_out == forbidden or os.path.abspath(os.path.join(abs_out, "lora_best.pt")) == os.path.abspath(
        os.path.join(os.path.dirname(__file__), "checkpoints", "lora", "lora_best.pt")
    ):
        raise SystemExit(
            f"refusing to write {out_dir!r} — that overwrites the main checkpoint. "
            "Use --out_dir checkpoints/gen_v2"
        )


def train(args):
    _reject_main_checkpoint_dir(args.out_dir)
    device = pick_device(args.device)
    _, model_path, arch, norm_type, patch_size = get_config(args.model, args.weights_dir)
    model = load_weights(create_architecture(arch), model_path)
    if args.init_checkpoint:
        print(f"init from {args.init_checkpoint}", flush=True)
        model = load_finetuned(model, args.init_checkpoint, device)
        extra = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False).get("extra") or {}
        lora_info = {
            "lora_params": extra.get("lora_params"),
            "head_params": extra.get("head_params"),
            "trainable_total": extra.get("trainable_total"),
            "lora_r": extra.get("lora_r", args.lora_r),
            "lora_alpha": extra.get("lora_alpha", args.lora_alpha),
            "target_modules": extra.get("target_modules", list(LORA_TARGET_MODULES)),
            "init_checkpoint": args.init_checkpoint,
        }
        if getattr(model, "fc_tiny", None) is not None:
            for p in model.fc_tiny.parameters():
                p.requires_grad = False
    else:
        lora_info = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        model = model.to(device)
    model.train()
    model.tune_visual = True

    counts = count_loaded_params(model)
    print("param counts", json.dumps(counts, indent=2))
    print("lora", json.dumps(lora_info, indent=2))
    assert counts["under_2b"]

    transform = build_transform(norm_type, patch_size)
    train_ds = AugmentedRealFake(args.train_csv, transform, degrade_p=args.degrade_p)
    val_ds = AugmentedRealFake(args.val_csv, transform, degrade_p=0.0)
    loader_kw = {}
    if args.num_workers > 0:
        loader_kw = {"num_workers": args.num_workers, "persistent_workers": True, "prefetch_factor": 2}
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    val_sources = [s.source for s in val_ds.samples]

    pos_weight = None
    if args.pos_weight and args.pos_weight != 1.0:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        print(f"pos_weight={args.pos_weight} (boosts fake-class loss)", flush=True)

    opt = torch.optim.AdamW(unique_trainable(model), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )
    history = []
    os.makedirs(args.out_dir, exist_ok=True)
    best_auc = -1.0
    best_path = os.path.join(args.out_dir, "lora_best.pt")

    for epoch in range(args.epochs):
        set_mode(model, True)
        model.tune_visual = True
        t0 = time.time()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        set_mode(model, False)
        v_running, v_n, correct = 0.0, 0, 0
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                v_running += float(loss.item()) * x.size(0)
                v_n += x.size(0)
                probs = torch.sigmoid(logits)
                correct += int(((probs >= 0.5).float() == y).sum().item())
                all_scores.extend(probs.flatten().cpu().tolist())
                all_labels.extend(y.flatten().cpu().tolist())
        val_loss = v_running / max(v_n, 1)
        val_acc = correct / max(v_n, 1)
        val_auc = _auc(all_labels, all_scores)
        rec = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_auc_sid_set": _auc_subset(all_labels, all_scores, val_sources, "sid_set"),
            "val_auc_cifake": _auc_subset(all_labels, all_scores, val_sources, "cifake"),
            "lr": float(sched.get_last_lr()[0]),
            "sec": time.time() - t0,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        epoch_path = os.path.join(args.out_dir, f"lora_epoch{epoch+1}.pt")
        save_finetuned(model, epoch_path, extra={**lora_info, **rec})
        # Prefer SID_Set AUC when present — CIFAKE 32x32 inverts CLIP and
        # would make us keep a worse native-res detector.
        score = rec["val_auc_sid_set"]
        if score != score:  # NaN
            score = val_auc
        if score > best_auc:
            best_auc = score
            save_finetuned(model, best_path, extra={**lora_info, **rec})
            print(f"  saved {best_path} (sid_auc={score:.4f})", flush=True)

    last_path = os.path.join(args.out_dir, "lora_last.pt")
    save_finetuned(model, last_path, extra={**lora_info, "history": history})
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump({"lora": lora_info, "history": history, "args": vars(args)}, f, indent=2)
    print(f"done. best={best_path}")
    return best_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train_sid_cached.csv")
    p.add_argument("--val_csv", default="data/processed/val_sid_cached.csv")
    p.add_argument("--model", default="clipdet_latent10k_plus")
    p.add_argument("--weights_dir", default="./weights")
    p.add_argument("--out_dir", default="checkpoints/gen_v2")
    p.add_argument(
        "--init_checkpoint",
        default=None,
        help="Continue from an existing LoRA (does not overwrite it; writes to --out_dir).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--degrade_p", type=float, default=0.6)
    p.add_argument("--pos_weight", type=float, default=2.0, help="BCE weight on the fake class.")
    p.add_argument("--num_workers", type=int, default=0, help="0 avoids MPS dataloader hangs.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
