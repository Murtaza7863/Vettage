#!/usr/bin/env python3
"""Build generalization-v2 indexes from WildFake train zips + a held-out family.

Hold-out family: Midjourney (mjv5). Never trained on. Evaluated from the
WildFake *eval* subset (`techjam-aigc/wildfake-eval-subset`, cross_generator)
which we are not allowed to train on anyway.

Train generators (WildFake training zips, sampled — full archive is ~1.29TB
and several families ship as 50GB part zips):
  - DDIM, DDPM  (fakes)
  - CelebA-HQ, AFHQ  (reals)
Never download: DALLE.zip (demo DALL·E Advanced), coco.zip (demo reals).

Also mixes in existing SID_Set + CIFAKE training indexes.
"""

from __future__ import annotations

import argparse
import os
import random
import zipfile

import requests
from PIL import Image

from dataset import Sample, read_index, sha1_bytes, write_index, _save_image

MS_BASE = "https://www.modelscope.cn/datasets/hy2628982280/WildFake/resolve/master"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Problem-statement demo split. Do not download or train on these.
FORBIDDEN_ZIPS = {
    "Images/Diffusion_based/DALLE.zip",
    "Images/Real/coco.zip",
}

TRAIN_ZIPS = {
    "ddim": "Images/Diffusion_based/DDIM.zip",
    "ddpm": "Images/Diffusion_based/DDPM.zip",
    "celebahq": "Images/Real/celebahq.zip",
    "afhq": "Images/Real/afhq.zip",
}

HOLD_OUT_FAKE_SOURCE = "midjourney_v5"
HOLD_OUT_REAL_SOURCE = "laion5b"


def download_modelscope(relpath: str, dest: str) -> str:
    if relpath in FORBIDDEN_ZIPS:
        raise SystemExit(f"refusing to download demo zip: {relpath}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = f"{MS_BASE}/{relpath}"
    pos = os.path.getsize(dest) if os.path.isfile(dest) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    print(f"GET {relpath}  resume={pos}", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        mode = "ab" if pos and r.status_code == 206 else "wb"
        if pos and r.status_code == 200:
            mode = "wb"
            pos = 0
        total = pos + int(r.headers.get("Content-Length") or 0)
        done = pos
        last = -1
        with open(dest, mode) as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else 0
                if pct != last and pct % 2 == 0:
                    print(f"  {os.path.basename(dest)}  {done/1e9:.2f}/{total/1e9:.2f} GB ({pct}%)", flush=True)
                    last = pct
    print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)
    return dest


def _is_image_name(name: str) -> bool:
    if name.endswith("/") or name.startswith("__MACOSX"):
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in IMAGE_EXTS


def extract_sample(
    zip_path: str,
    out_dir: str,
    source: str,
    label: int,
    n: int,
    seed: int,
) -> list[Sample]:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if _is_image_name(n)]
        rng.shuffle(names)
        names = names[:n]
        samples: list[Sample] = []
        for i, name in enumerate(names):
            ext = os.path.splitext(name)[1].lower() or ".jpg"
            img_id = f"{source}_{i:06d}"
            dest = os.path.join(out_dir, f"{img_id}{ext if ext == '.png' else '.jpg'}")
            if not os.path.isfile(dest):
                with zf.open(name) as src:
                    img = Image.open(src).convert("RGB")
                    _save_image(img, dest)
            with open(dest, "rb") as f:
                digest = sha1_bytes(f.read())
            samples.append(
                Sample(path=dest, label=label, source=source, image_id=img_id, sha1=digest)
            )
            if (i + 1) % 500 == 0:
                print(f"  extracted {source} {i+1}/{len(names)}", flush=True)
    print(f"{source}: {len(samples)} from {zip_path}", flush=True)
    return samples


def export_holdout(out_dir: str, n_fake: int, n_real: int, seed: int) -> list[Sample]:
    """Eval-only. Do not mix these paths into any training CSV."""
    from datasets import load_dataset

    rng = random.Random(seed)
    print("loading WildFake eval subset config=cross_generator (eval only, never train)", flush=True)
    ds = load_dataset("techjam-aigc/wildfake-eval-subset", "cross_generator", split="validation")
    fake_idx, real_idx = [], []
    for i, row in enumerate(ds):
        src = str(row["source"])
        if src == HOLD_OUT_FAKE_SOURCE and int(row["label"]) == 1:
            fake_idx.append(i)
        elif src == HOLD_OUT_REAL_SOURCE and int(row["label"]) == 0:
            real_idx.append(i)
    rng.shuffle(fake_idx)
    rng.shuffle(real_idx)
    fake_idx = fake_idx[:n_fake]
    real_idx = real_idx[:n_real]
    samples: list[Sample] = []

    def dump(indices, source, label):
        dest_dir = os.path.join(out_dir, source)
        os.makedirs(dest_dir, exist_ok=True)
        for i in indices:
            row = ds[int(i)]
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            img_id = str(row.get("id") or f"{source}_{i}").replace("/", "_")
            dest = os.path.join(dest_dir, f"{img_id}.jpg")
            _save_image(img, dest)
            with open(dest, "rb") as f:
                digest = sha1_bytes(f.read())
            samples.append(
                Sample(path=dest, label=label, source=source, image_id=img_id, sha1=digest)
            )

    dump(fake_idx, HOLD_OUT_FAKE_SOURCE, 1)
    dump(real_idx, HOLD_OUT_REAL_SOURCE, 0)
    print(f"hold-out exported fake={len(fake_idx)} real={len(real_idx)}", flush=True)
    return samples


def mix_existing(train_sid_csv: str, train_all_csv: str, include_cifake: bool) -> list[Sample]:
    sid = [s for s in read_index(train_sid_csv) if s.source == "sid_set"]
    extra = []
    if include_cifake:
        extra = [s for s in read_index(train_all_csv) if s.source == "cifake"]
    print(f"existing train SID={len(sid)} CIFAKE={len(extra)}", flush=True)
    return sid + extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw/wildfake")
    p.add_argument("--out_dir", default="data/processed/gen_v2")
    p.add_argument("--n_ddim", type=int, default=2500)
    p.add_argument("--n_ddpm", type=int, default=2500)
    p.add_argument("--n_celebahq", type=int, default=1500)
    p.add_argument("--n_afhq", type=int, default=1500)
    p.add_argument("--n_holdout_fake", type=int, default=999)
    p.add_argument("--n_holdout_real", type=int, default=999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--holdout_only", action="store_true")
    p.add_argument("--no_cifake", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    holdout = export_holdout(
        os.path.join(args.out_dir, "holdout"),
        n_fake=args.n_holdout_fake,
        n_real=args.n_holdout_real,
        seed=args.seed,
    )
    write_index(os.path.join(args.out_dir, "val_unseen_midjourney.csv"), holdout)
    hold_sha = {s.sha1 for s in holdout}
    if args.holdout_only:
        return

    zips = {}
    for key, rel in TRAIN_ZIPS.items():
        dest = os.path.join(args.raw_dir, os.path.basename(rel))
        if not args.skip_download or not os.path.isfile(dest):
            download_modelscope(rel, dest)
        zips[key] = dest

    wf: list[Sample] = []
    wf.extend(
        extract_sample(
            zips["ddim"], os.path.join(args.out_dir, "ddim"), "wildfake_ddim", 1, args.n_ddim, args.seed
        )
    )
    wf.extend(
        extract_sample(
            zips["ddpm"], os.path.join(args.out_dir, "ddpm"), "wildfake_ddpm", 1, args.n_ddpm, args.seed + 1
        )
    )
    wf.extend(
        extract_sample(
            zips["celebahq"],
            os.path.join(args.out_dir, "celebahq"),
            "wildfake_celebahq",
            0,
            args.n_celebahq,
            args.seed + 2,
        )
    )
    wf.extend(
        extract_sample(
            zips["afhq"], os.path.join(args.out_dir, "afhq"), "wildfake_afhq", 0, args.n_afhq, args.seed + 3
        )
    )
    leaked = [s for s in wf if s.sha1 in hold_sha]
    if leaked:
        print(f"dropping {len(leaked)} sha1 overlaps with hold-out", flush=True)
        wf = [s for s in wf if s.sha1 not in hold_sha]

    existing = mix_existing(
        "data/processed/train_sid_cached.csv",
        "data/processed/train_cached.csv",
        include_cifake=not args.no_cifake,
    )
    train = existing + wf
    rng = random.Random(args.seed)
    rng.shuffle(train)
    write_index(os.path.join(args.out_dir, "train.csv"), train)
    from collections import Counter
    import json

    counts = Counter((s.source, s.label) for s in train)
    protocol = {
        "hold_out_family": "Midjourney (mjv5)",
        "hold_out_eval": "techjam-aigc/wildfake-eval-subset:cross_generator",
        "train_wildfake_zips": TRAIN_ZIPS,
        "forbidden": sorted(FORBIDDEN_ZIPS),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "note": "Do not write checkpoints to checkpoints/lora. Use checkpoints/gen_v2.",
    }
    with open(os.path.join(args.out_dir, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
