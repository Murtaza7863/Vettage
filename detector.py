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
        out = model(x).cpu().numpy()
        if out.ndim == 2 and out.shape[1] == 1:
            llr = out[:, 0]
        elif out.ndim == 2 and out.shape[1] == 2:
            llr = out[:, 1] - out[:, 0]
        else:
            llr = np.reshape(out, (out.shape[0], -1)).mean(axis=1)
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
        img = Image.open(path).convert("RGB")
        batch_imgs.append(transform(img))
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
        out = model(x).cpu().numpy()
        if out.ndim == 2 and out.shape[1] == 1:
            llr = out[:, 0]
        elif out.ndim == 2 and out.shape[1] == 2:
            llr = out[:, 1] - out[:, 0]
        else:
            llr = np.reshape(out, (out.shape[0], -1)).mean(axis=1)
        llrs.append(llr)
    if not llrs:
        return np.array([], dtype=np.float32)
    return logit_to_confidence(np.concatenate(llrs, axis=0))
