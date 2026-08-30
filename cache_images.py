#!/usr/bin/env python3
"""Build a resized copy of the training images to make augmentation affordable.

Why this is safe: the detector's input pipeline is Resize(224, short side) +
CenterCrop(224), so a 1024x1024 source is already downsampled to 224px before
the model sees it. Caching at short side 256 therefore preserves essentially
all information the network can use, while cutting JPEG decode and the
full-resolution augmentation cost (blur / noise / re-encode) by ~16x.

Eval still runs on the original files at native resolution, so the reported
robustness numbers keep their original meaning.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

from dataset import Sample, read_index, write_index


def _resize_one(task: tuple[str, str, int, int]) -> tuple[str, str] | None:
    src, dest, short_side, quality = task
    if os.path.exists(dest):
        return src, dest
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = short_side / min(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            im.save(dest, format="JPEG", quality=quality, subsampling=0)
        return src, dest
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {src}: {exc}")
        return None


def build(index_csv: str, out_csv: str, cache_root: str, short_side: int, quality: int, workers: int):
    samples = read_index(index_csv)
    tasks = []
    mapping: dict[str, str] = {}
    for s in samples:
        rel = os.path.relpath(s.path, "data/processed")
        dest = os.path.join(cache_root, os.path.splitext(rel)[0] + ".jpg")
        mapping[s.path] = dest
        tasks.append((s.path, dest, short_side, quality))

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_resize_one, t) for t in tasks]
        for f in as_completed(futs):
            if f.result():
                done += 1
            if done % 2000 == 0:
                print(f"  cached {done}/{len(tasks)}", flush=True)

    out: list[Sample] = []
    for s in samples:
        dest = mapping[s.path]
        if os.path.exists(dest):
            out.append(Sample(path=dest, label=s.label, source=s.source, image_id=s.image_id, sha1=s.sha1))
    write_index(out_csv, out)
    print(f"{index_csv} -> {out_csv}: {len(out)}/{len(samples)} images cached at short side {short_side}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--val_csv", default="data/processed/val.csv")
    p.add_argument("--cache_root", default="data/cache256")
    p.add_argument("--short_side", type=int, default=256)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    build(args.train_csv, "data/processed/train_cached.csv", args.cache_root, args.short_side, args.quality, args.workers)
    build(args.val_csv, "data/processed/val_cached.csv", args.cache_root, args.short_side, args.quality, args.workers)


if __name__ == "__main__":
    main()
