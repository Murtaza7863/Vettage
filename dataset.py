#!/usr/bin/env python3
"""Training-data loader for SID_Set + CIFAKE.

Rules:
- Train only on SID_Set *train* and CIFAKE *train*.
- Never touch the problem-statement WildFake / COCO val2017 demo split.
- Never use SID_Set official validation/test as our held-out set; we carve val
  from the downloaded training pool after dedup.
- SID_Set labels: 0 real, 1 full-synthetic, 2 tampered. Binary task uses
  real vs full-synthetic by default (tampered excluded unless --include_tampered).

CIFAKE is pulled from the Hugging Face mirror of the Kaggle dataset
`birdy654/cifake-real-and-ai-generated-synthetic-images` because this machine
has no Kaggle API token. The mirror is `dragonintelligence/CIFAKE-image-dataset`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SID_HF = "saberzl/SID_Set"
CIFAKE_HF = "dragonintelligence/CIFAKE-image-dataset"
# Original Kaggle source (requires kaggle.json). Same pixels as CIFAKE_HF.
CIFAKE_KAGGLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
DEFACTIFY_HF = "Rajarshi-Roy-research/Defactify_Image_Dataset"


@dataclass
class Sample:
    path: str
    label: int  # 0 real, 1 fake
    source: str
    image_id: str
    sha1: str


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _save_image(img: Image.Image, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    img = img.convert("RGB")
    ext = os.path.splitext(dest)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(dest, format="JPEG", quality=95)
    else:
        img.save(dest)


def download_sid_set(
    out_dir: str,
    max_real: int = 2000,
    max_fake: int = 2000,
    include_tampered: bool = False,
    seed: int = 42,
) -> list[Sample]:
    """Stream SID_Set train split and write a balanced real/fake subset to disk."""
    from datasets import load_dataset

    rng = random.Random(seed)
    # Reservoir-style accept until quotas fill. Streaming order is not shuffled
    # by label, so we keep everything that still has quota.
    want = {0: max_real, 1: max_fake}
    if include_tampered:
        want[2] = max_fake  # extra fake bucket; mapped to label 1 on disk
    got = {0: 0, 1: 0, 2: 0}
    samples: list[Sample] = []

    print(f"streaming {SID_HF} split=train (official val/test are not used)", flush=True)
    ds = load_dataset(SID_HF, split="train", streaming=True)
    for row in ds:
        lab = int(row["label"])
        if lab not in want or got[lab] >= want[lab]:
            if all(got[k] >= want[k] for k in want):
                break
            continue
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = img.convert("RGB") if hasattr(img, "convert") else Image.fromarray(img)
        img_id = str(row.get("img_id") or f"sid_{lab}_{got[lab]:06d}")
        binary = 0 if lab == 0 else 1
        split_name = "real" if binary == 0 else "fake"
        dest = os.path.join(out_dir, "sid_set", split_name, f"{img_id}.jpg")
        _save_image(img, dest)
        with open(dest, "rb") as f:
            digest = sha1_bytes(f.read())
        samples.append(
            Sample(path=dest, label=binary, source="sid_set", image_id=img_id, sha1=digest)
        )
        got[lab] += 1
        if sum(got.values()) % 100 == 0:
            print(f"  SID_Set saved {got}", flush=True)
    print(f"SID_Set done {got} -> {len(samples)} files", flush=True)
    rng.shuffle(samples)
    return samples


def download_cifake(
    out_dir: str,
    max_real: int = 4000,
    max_fake: int = 4000,
    seed: int = 42,
) -> list[Sample]:
    """Download CIFAKE train split from the HF mirror of the Kaggle dataset."""
    from datasets import load_dataset

    rng = random.Random(seed)
    print(f"loading {CIFAKE_HF} split=train (Kaggle source: {CIFAKE_KAGGLE})")
    ds = load_dataset(CIFAKE_HF, split="train")
    # HF mirror: 0 FAKE, 1 REAL
    buckets = {0: [], 1: []}
    for i, row in enumerate(ds):
        lab = int(row["label"])
        buckets[lab].append(i)
    rng.shuffle(buckets[0])
    rng.shuffle(buckets[1])
    pick_fake = buckets[0][:max_fake]
    pick_real = buckets[1][:max_real]
    samples: list[Sample] = []
    for i in pick_real + pick_fake:
        row = ds[int(i)]
        hf_lab = int(row["label"])
        binary = 0 if hf_lab == 1 else 1  # REAL->0, FAKE->1
        img = row["image"]
        img_id = f"cifake_{'real' if binary == 0 else 'fake'}_{i:06d}"
        dest = os.path.join(
            out_dir, "cifake", "real" if binary == 0 else "fake", f"{img_id}.jpg"
        )
        _save_image(img, dest)
        with open(dest, "rb") as f:
            digest = sha1_bytes(f.read())
        samples.append(
            Sample(path=dest, label=binary, source="cifake", image_id=img_id, sha1=digest)
        )
    print(f"CIFAKE done real={len(pick_real)} fake={len(pick_fake)}")
    return samples


def download_defactify(
    out_dir: str,
    max_real: int = 3000,
    max_fake: int = 3000,
    seed: int = 42,
    split: str = "train",
) -> list[Sample]:
    """Download Defactify dataset: SD2.1, SDXL, SD3, DALL-E 3, Midjourney v6 + COCO real."""
    from datasets import load_dataset

    rng = random.Random(seed)
    print(f"loading {DEFACTIFY_HF} split={split}", flush=True)
    ds = load_dataset(DEFACTIFY_HF, split=split)

    # Label_A: 0=real, 1=fake
    # Label_B: 0=real, 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney
    gen_names = {0: "real", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}

    buckets: dict[int, list[int]] = {0: [], 1: []}
    for i, row in enumerate(ds):
        buckets[int(row["Label_A"])].append(i)
    rng.shuffle(buckets[0])
    rng.shuffle(buckets[1])
    pick_real = buckets[0][:max_real]
    pick_fake = buckets[1][:max_fake]

    samples: list[Sample] = []
    for i in pick_real + pick_fake:
        row = ds[int(i)]
        binary = int(row["Label_A"])  # 0=real, 1=fake
        gen = gen_names.get(int(row["Label_B"]), "unknown")
        img = row["Image"]
        if not isinstance(img, Image.Image):
            img = img.convert("RGB") if hasattr(img, "convert") else Image.fromarray(img)
        img_id = f"defactify_{gen}_{i:06d}"
        split_name = "real" if binary == 0 else "fake"
        dest = os.path.join(out_dir, "defactify", split_name, f"{img_id}.jpg")
        _save_image(img, dest)
        with open(dest, "rb") as f:
            digest = sha1_bytes(f.read())
        samples.append(
            Sample(path=dest, label=binary, source="defactify", image_id=img_id, sha1=digest)
        )
    print(f"Defactify done real={len(pick_real)} fake={len(pick_fake)}", flush=True)
    return samples


def download_synthwildx(
    out_dir: str,
    synthwildx_dir: str = "data/synthwildx",
) -> list[Sample]:
    """Load SynthWildX images (DALL-E 3, Midjourney, Firefly from Twitter/X).
    
    These are all fake — no real images in this set.
    Run data/synthwildx/download_synthwildx.py first to fetch the images.
    """
    import pandas as pd

    csv_path = os.path.join(synthwildx_dir, "list.csv")
    if not os.path.exists(csv_path):
        print(f"SynthWildX list.csv not found at {csv_path}, skipping")
        return []

    tab = pd.read_csv(csv_path)
    samples: list[Sample] = []
    missing = 0
    for _, row in tab.iterrows():
        fpath = os.path.join(synthwildx_dir, row["filename"])
        if not os.path.isfile(fpath):
            missing += 1
            continue
        img_id = f"synthwildx_{row['typ']}_{os.path.splitext(os.path.basename(fpath))[0]}"
        # Copy to out_dir structure
        dest = os.path.join(out_dir, "synthwildx", "fake", os.path.basename(fpath))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            import shutil
            shutil.copy2(fpath, dest)
        with open(dest, "rb") as f:
            digest = sha1_bytes(f.read())
        samples.append(
            Sample(path=dest, label=1, source="synthwildx", image_id=img_id, sha1=digest)
        )
    print(f"SynthWildX done fake={len(samples)} (missing={missing})", flush=True)
    return samples


def dedup(samples: list[Sample]) -> list[Sample]:
    seen_hash: set[str] = set()
    seen_id: set[tuple[str, str]] = set()
    out: list[Sample] = []
    n_drop = 0
    for s in samples:
        key_id = (s.source, s.image_id)
        if s.sha1 in seen_hash or key_id in seen_id:
            n_drop += 1
            continue
        seen_hash.add(s.sha1)
        seen_id.add(key_id)
        out.append(s)
    print(f"dedup dropped {n_drop}, kept {len(out)}")
    return out


def stratified_split(
    samples: list[Sample], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_key: dict[tuple[int, str], list[Sample]] = {}
    for s in samples:
        by_key.setdefault((s.label, s.source), []).append(s)
    train, val = [], []
    for _, bucket in by_key.items():
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_frac))) if len(bucket) > 1 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    print(
        f"split train={len(train)} val={len(val)} "
        f"(val carved from training pool, not demo WildFake/COCO)"
    )
    return train, val


def write_index(path: str, samples: list[Sample]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "image_id", "sha1"])
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))


def read_index(path: str) -> list[Sample]:
    rows: list[Sample] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                Sample(
                    path=row["path"],
                    label=int(row["label"]),
                    source=row["source"],
                    image_id=row["image_id"],
                    sha1=row["sha1"],
                )
            )
    return rows


class RealFakeDataset(Dataset):
    """Simple real(0)/fake(1) image dataset from a CSV index."""

    def __init__(self, index_csv: str, transform=None):
        self.samples = read_index(index_csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label, s.path


def build_pipeline(
    out_dir: str = "data/processed",
    sid_real: int = 2000,
    sid_fake: int = 2000,
    cifake_real: int = 4000,
    cifake_fake: int = 4000,
    defactify_real: int = 3000,
    defactify_fake: int = 3000,
    val_frac: float = 0.1,
    seed: int = 42,
    skip_sid: bool = False,
    skip_cifake: bool = False,
    skip_defactify: bool = False,
    skip_synthwildx: bool = False,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    samples: list[Sample] = []
    if not skip_sid:
        samples.extend(
            download_sid_set(out_dir, max_real=sid_real, max_fake=sid_fake, seed=seed)
        )
    if not skip_cifake:
        samples.extend(
            download_cifake(out_dir, max_real=cifake_real, max_fake=cifake_fake, seed=seed)
        )
    if not skip_defactify:
        samples.extend(
            download_defactify(out_dir, max_real=defactify_real, max_fake=defactify_fake, seed=seed)
        )
    if not skip_synthwildx:
        samples.extend(
            download_synthwildx(out_dir)
        )
    samples = dedup(samples)
    train, val = stratified_split(samples, val_frac=val_frac, seed=seed)
    write_index(os.path.join(out_dir, "train.csv"), train)
    write_index(os.path.join(out_dir, "val.csv"), val)
    summary = {
        "n_train": len(train),
        "n_val": len(val),
        "train_real": sum(s.label == 0 for s in train),
        "train_fake": sum(s.label == 1 for s in train),
        "val_real": sum(s.label == 0 for s in val),
        "val_fake": sum(s.label == 1 for s in val),
        "sources": sorted({s.source for s in samples}),
        "excluded": [
            "SID_Set official validation/test",
            "WildFake / COCO val2017 demo split from the problem statement",
        ],
        "sid_hf": SID_HF,
        "cifake_hf": CIFAKE_HF,
        "cifake_kaggle": CIFAKE_KAGGLE,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--sid_real", type=int, default=2000)
    p.add_argument("--sid_fake", type=int, default=2000)
    p.add_argument("--cifake_real", type=int, default=4000)
    p.add_argument("--cifake_fake", type=int, default=4000)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_sid", action="store_true")
    p.add_argument("--skip_cifake", action="store_true")
    p.add_argument("--defactify_real", type=int, default=3000)
    p.add_argument("--defactify_fake", type=int, default=3000)
    p.add_argument("--skip_defactify", action="store_true")
    p.add_argument("--skip_synthwildx", action="store_true")
    args = p.parse_args()
    build_pipeline(
        out_dir=args.out_dir,
        sid_real=args.sid_real,
        sid_fake=args.sid_fake,
        cifake_real=args.cifake_real,
        cifake_fake=args.cifake_fake,
        defactify_real=args.defactify_real,
        defactify_fake=args.defactify_fake,
        val_frac=args.val_frac,
        seed=args.seed,
        skip_sid=args.skip_sid,
        skip_cifake=args.skip_cifake,
        skip_defactify=args.skip_defactify,
        skip_synthwildx=args.skip_synthwildx,
    )


if __name__ == "__main__":
    main()