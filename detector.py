"""Shared CLIP-based synthetic-image detector used by predict.py and eval scripts."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn.functional import sigmoid
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from networks import create_architecture, load_weights
from utils.processing import make_normalize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(input_dir: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def get_config(model_name: str, weights_dir: str = "./weights"):
    with open(os.path.join(weights_dir, model_name, "config.yaml")) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data["weights_file"])
    return data["model_name"], model_path, data["arch"], data["norm_type"], data["patch_size"]


def build_transform(norm_type: str, patch_size):
    transform = []
    if patch_size == "Clip224":
        transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((224, 224)))
    elif isinstance(patch_size, (tuple, list)):
        transform.append(Resize(*patch_size))
        transform.append(CenterCrop(patch_size[0]))
    elif patch_size is not None and patch_size > 0:
        transform.append(CenterCrop(patch_size))
    transform.append(make_normalize(norm_type))
    return Compose(transform)


def load_detector(
    model_name: str = "clipdet_latent10k_plus",
    weights_dir: str = "./weights",
    device: str | None = None,
):
    """Load a named detector from weights/<model_name>/."""
    device_obj = pick_device(device)
    _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)
    model = load_weights(create_architecture(arch), model_path)
    model = model.to(device_obj).eval()
    transform = build_transform(norm_type, patch_size)
    return model, transform, device_obj, arch


def _module_param_count(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_loaded_params(model) -> dict:
    """Count CLIP backbone + head even when the backbone is stored in model.bb."""
    registered = _module_param_count(model)
    backbone_total = 0
    vision = 0
    text = 0
    if hasattr(model, "bb") and model.bb:
        backbone = model.bb[0]
        backbone_total = _module_param_count(backbone)
        if hasattr(backbone, "visual"):
            vision = _module_param_count(backbone.visual)
        if hasattr(backbone, "transformer"):
            text = _module_param_count(backbone.transformer)
    head = _module_param_count(model.fc) if hasattr(model, "fc") else registered
    # Image-path inference uses the vision encoder + linear head.
    inference_total = (vision if vision else backbone_total) + head
    grand_total = backbone_total + head if backbone_total else registered
    return {
        "registered_nn_module": registered,
        "clip_full_backbone": backbone_total,
        "clip_vision_encoder": vision,
        "clip_text_encoder": text,
        "classifier_head": head,
        "inference_vision_plus_head": inference_total,
        "loaded_module_total": grand_total,
        "under_2b": grand_total < 2_000_000_000,
    }


def logit_to_confidence(llr: np.ndarray | float) -> np.ndarray | float:
    """Convert detector LLR / logit to a [0, 1] synthetic-image confidence."""
    tens = torch.as_tensor(llr, dtype=torch.float32)
    conf = sigmoid(tens).cpu().numpy()
    if np.ndim(llr) == 0:
        return float(conf)
    return conf


def _llr_from_output(out: np.ndarray) -> np.ndarray:
    if out.ndim == 2 and out.shape[1] == 1:
        return out[:, 0]
    if out.ndim == 2 and out.shape[1] == 2:
        return out[:, 1] - out[:, 0]
    return np.reshape(out, (out.shape[0], -1)).mean(axis=1)


def _std(xs) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size < 2:
        return 0.0
    return float(np.std(xs, ddof=1))


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


@torch.no_grad()
def score_images(
    paths: Iterable[str],
    model,
    transform,
    device: torch.device,
    batch_size: int = 8,
) -> list[dict]:
    path_list = list(paths)
    rows: list[dict] = []
    batch_imgs: list[torch.Tensor] = []
    batch_paths: list[str] = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, 0).to(device)
        llr = _llr_from_output(model(x).cpu().numpy())
        for p, score in zip(batch_paths, llr):
            rows.append(
                {
                    "image_path": p,
                    "pred": float(logit_to_confidence(score)),
                    "llr": float(score),
                }
            )
        batch_imgs.clear()
        batch_paths.clear()

    for path in path_list:
        batch_imgs.append(transform(load_rgb(path)))
        batch_paths.append(path)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return rows


@torch.no_grad()
def score_pil_images(
    images: list[Image.Image],
    model,
    transform,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Score in-memory PIL images. Returns sigmoid confidence in [0, 1]."""
    llrs = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        x = torch.stack([transform(im.convert("RGB")) for im in chunk], 0).to(device)
        llrs.append(_llr_from_output(model(x).cpu().numpy()))
    if not llrs:
        return np.array([], dtype=np.float32)
    return logit_to_confidence(np.concatenate(llrs, axis=0))


@torch.no_grad()
def score_images_repeated(
    paths: Iterable[str],
    model,
    transform,
    device: torch.device,
    n_repeats: int = 5,
) -> list[dict]:
    """Score each image n_repeats times (reload from disk each pass).

    `pred` is the mean confidence so the required JSON contract still holds.
    Extra keys record per-repeat drift.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    rows: list[dict] = []
    model.eval()
    for path in paths:
        preds: list[float] = []
        llrs: list[float] = []
        for _ in range(n_repeats):
            x = transform(load_rgb(path)).unsqueeze(0).to(device)
            llr = float(_llr_from_output(model(x).cpu().numpy())[0])
            llrs.append(llr)
            preds.append(float(logit_to_confidence(llr)))
        mean_pred = float(np.mean(preds))
        rows.append(
            {
                "image_path": path,
                "pred": mean_pred,
                "pred_mean": mean_pred,
                "pred_std": _std(preds),
                "pred_min": float(np.min(preds)),
                "pred_max": float(np.max(preds)),
                "pred_range": float(np.max(preds) - np.min(preds)),
                "n_repeats": n_repeats,
                "repeats": preds,
                "llr_mean": float(np.mean(llrs)),
                "llr_repeats": llrs,
            }
        )
    return rows


def aggregate_predictions(rows: list[dict], threshold: float = 0.5) -> dict:
    """Folder-level summary. `pred` on each row is already the per-image mean."""
    if not rows:
        return {"n_images": 0, "threshold": threshold}
    preds = np.array([r["pred"] for r in rows], dtype=np.float64)
    within = np.array([float(r.get("pred_std") or 0.0) for r in rows], dtype=np.float64)
    n_repeats = int(rows[0].get("n_repeats") or 1)
    synthetic = preds > threshold
    return {
        "n_images": len(rows),
        "n_repeats": n_repeats,
        "threshold": threshold,
        "mean_pred": float(np.mean(preds)),
        "std_pred": _std(preds),
        "median_pred": float(np.median(preds)),
        "min_pred": float(np.min(preds)),
        "max_pred": float(np.max(preds)),
        "n_synthetic": int(np.sum(synthetic)),
        "n_real": int(np.sum(~synthetic)),
        "frac_synthetic": float(np.mean(synthetic)),
        "mean_within_image_std": float(np.mean(within)),
        "max_within_image_std": float(np.max(within)),
        "max_within_image_range": float(max(float(r.get("pred_range") or 0.0) for r in rows)),
        "stable": bool(np.max(within) < 1e-6),
    }
