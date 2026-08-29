"""Problem-statement robustness transforms. Used at eval (always) and train (stochastic)."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: str
    apply: callable


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_then_upscale(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = img.resize((nw, nh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng()
    noisy = arr + rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(
    img: Image.Image,
    magnitude: float = 0.20,
    brightness: float | None = None,
    contrast: float | None = None,
    saturation: float | None = None,
    randomize: bool = False,
) -> Image.Image:
    """brightness / contrast / saturation each ±magnitude (default ±20%)."""
    if randomize:
        b = 1.0 + random.uniform(-magnitude, magnitude)
        c = 1.0 + random.uniform(-magnitude, magnitude)
        s = 1.0 + random.uniform(-magnitude, magnitude)
    else:
        b = 1.0 + magnitude if brightness is None else brightness
        c = 1.0 - magnitude if contrast is None else contrast
        s = 1.0 + magnitude if saturation is None else saturation
    out = ImageEnhance.Brightness(img).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    out = ImageEnhance.Color(out).enhance(s)
    return out


def center_crop_frac(img: Image.Image, frac: float = 0.80) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(round(w * frac))), max(1, int(round(h * frac)))
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def eval_conditions() -> list[TransformSpec]:
    """Exactly the Track 5 robustness conditions, each as its own eval setting."""
    specs: list[TransformSpec] = [
        TransformSpec("clean", "none", lambda im: im.convert("RGB")),
    ]
    for q in (90, 70, 50, 30):
        specs.append(TransformSpec("jpeg", f"quality={q}", lambda im, q=q: jpeg_compress(im, q)))
    for s in (0.5, 1.0, 2.0):
        specs.append(TransformSpec("gaussian_blur", f"sigma={s}", lambda im, s=s: gaussian_blur(im, s)))
    for sc in (0.5, 0.25):
        specs.append(
            TransformSpec(
                "resize",
                f"scale={sc}x then upscale",
                lambda im, sc=sc: resize_then_upscale(im, sc),
            )
        )
    for s in (0.02, 0.05, 0.10):
        specs.append(TransformSpec("gaussian_noise", f"sigma={s}", lambda im, s=s: gaussian_noise(im, s)))
    specs.append(TransformSpec("color_jitter", "b/c/s ±20%", lambda im: color_jitter(im, 0.20)))
    specs.append(TransformSpec("center_crop", "crop to 80%", lambda im: center_crop_frac(im, 0.80)))
    return specs


def random_train_degradation(img: Image.Image, p: float = 0.5) -> Image.Image:
    """Apply one problem-statement transform with probability p (training-time)."""
    if random.random() >= p:
        return img.convert("RGB")
    ops = [
        lambda im: jpeg_compress(im, random.choice([90, 70, 50, 30])),
        lambda im: gaussian_blur(im, random.choice([0.5, 1.0, 2.0])),
        lambda im: resize_then_upscale(im, random.choice([0.5, 0.25])),
        lambda im: gaussian_noise(im, random.choice([0.02, 0.05, 0.10])),
        lambda im: color_jitter(im, 0.20, randomize=True),
        lambda im: center_crop_frac(im, 0.80),
    ]
    return random.choice(ops)(img.convert("RGB"))
