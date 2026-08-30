# Vettage — Track 5: robust AI-generated image detection

Prototype detector for **image-level AIGC vs authentic** photos. The hard part is not clean lab images — it is JPEG re-encodes, blur, thumbnails, noise, filters, and crops after an image has been posted. This repo is a CLIP ViT-L/14 detector (Cozzolino et al., CVPRW 2024) with a **LoRA adapter** for native-resolution generators and a **second linear head** for tiny (CIFAR-scale) images.

Loaded module ≈ **431M parameters** (vision + unused text encoder + LoRA + two heads). Under the **2B** cap. **Not trained** on the WildFake / COCO val2017 demonstration split.

`pred > 0.5` means synthetic. `pred` is `sigmoid(LLR)` in `[0, 1]`.

## Setup

Python 3.11+, Git LFS (pretrained heads in `weights/`).

```bash
git clone https://github.com/Murtaza7863/Vettage.git
cd Vettage
git lfs pull
pip install -r requirements.txt
```

GPU: CUDA via `--device cuda`, Apple Silicon via `--device mps` (auto-detected). CPU works, slower.

Trained adapters: `checkpoints/lora/lora_best.pt` (committed). Pass `--checkpoint none` to run the frozen paper head only.

## Inference (required script)

```bash
python predict.py --input_dir /path/to/images --output_json out.json
```

Writes a JSON **array** of `{image_path, pred}` only (hackathon contract). A sidecar `out.aggregate.json` has folder stats.

```bash
# End-to-end smoke on the bundled samples
python predict.py --input_dir sample_images --output_json /tmp/vettage.json
```

Useful flags: `--device mps|cuda|cpu`, `--checkpoint none`, `--repeats 5` (stability log; scores are deterministic), `--probe_transforms` (same file under Track 5 degradations), `--threshold 0.5`.

## How it works

1. **Backbone:** OpenCLIP ViT-L/14 CommonPool. Linear head from GRIP-UNINA `clipdet_latent10k_plus`.
2. **Native photos (short side > 64px):** LoRA on ViT MLP `c_fc` / `c_proj` (not `out_proj` — OpenCLIP never calls that submodule) + SID head. Trained on SID_Set train only, with random Track 5 degradations.
3. **Tiny images (short side ≤ 64px):** same CLIP features, **separate** 1025-param logistic head fit on CIFAKE. Mixing CIFAKE into the SID LoRA inverted CLIP (32×32 upsampled to 224 looks “fake” even when real).
4. CLIP base weights stay frozen. Trainable count during LoRA ≈ 3.9M.

## Results (held-out val, never in train)

`data/processed/val.csv`, n=1600, threshold 0.5. Val is a stratified slice of the **training pools**, not SID official val/test and not WildFake/COCO.

| Split | Metric | Frozen CLIP | + SID LoRA | + tiny head |
|---|---|---|---|---|
| SID_Set (n=1000) | AUC / acc / fake-catch | 0.859 / 62% / 26% | **1.00 / 99.9% / 100%** | **same** |
| CIFAKE 32×32 (n=600) | AUC / acc / fake-catch | 0.32 / 38% / 20% | 0.66 / 57% / 21% | **0.997 / 97% / 97%** |
| Overall | AUC / acc / fake-catch | 0.67 / 53% / 24% | 0.95 / 84% / 70% | **0.999 / 98.9% / 98.9%** |

Numbers: `results/success_lora_tiny.json`. Params: `MODEL_INFO.md`.

### Robustness (Track 5 transforms)

Same 600-image mixed val. Full table: `results/robustness_lora.csv`, chart: `results/robustness_lora.png`.

| Condition | Frozen overall / SID | Ours overall / SID / CIFAKE |
|---|---|---|
| clean | 0.67 / 0.87 | **0.999 / 1.000 / 0.996** |
| JPEG q=30 | 0.57 / 0.73 | **0.998 / 1.000 / 0.988** |
| blur σ=2 | 0.78 / 0.92 | **0.981 / 1.000 / 0.879** |
| resize 0.25× then up | 0.79 / 0.92 | **0.974 / 1.000 / 0.845** |
| noise σ=0.10 | 0.60 / 0.75 | **0.979 / 1.000 / 0.864** |
| jitter ±20% | 0.68 / 0.86 | **0.997 / 1.000 / 0.984** |
| center crop 80% | 0.63 / 0.86 | **0.997 / 1.000 / 0.981** |

SID stays ~1.0 AUC under every listed transform. Residual misses are **tiny** images after heavy blur / 0.25× (32px → 8px → 32px).

## Reproduce

Data is **not** in git (`data/processed/` is gitignored). Rebuild:

```bash
python dataset.py   # SID_Set train + CIFAKE train from Hugging Face; carves val
python cache_images.py
python train_lora.py \
  --train_csv data/processed/train_sid_cached.csv \
  --val_csv data/processed/val_sid_cached.csv \
  --epochs 2 --batch_size 8 --lr 1e-4 --lora_r 16 --lora_alpha 32 \
  --degrade_p 0.6 --pos_weight 2.0 --num_workers 0
python train_tiny_head.py --checkpoint checkpoints/lora/lora_best.pt
python eval_success.py --val_csv data/processed/val.csv --checkpoint checkpoints/lora/lora_best.pt
python eval_robustness.py --val_csv data/processed/val.csv --max_images 600 \
  --checkpoint checkpoints/lora/lora_best.pt --tag lora_tiny \
  --out_csv results/robustness_lora.csv
python error_analysis.py --checkpoint checkpoints/lora/lora_best.pt
```

Never pass the WildFake / COCO val2017 demo split into `train_lora.py` / `train_tiny_head.py`.

## Error analysis

See `results/error_analysis.md`. Short version: on native SID, the decision boundary is wide (highest real ≈ 0.24, lowest fake ≈ 0.83 even after JPEG q=30). Remaining errors at threshold 0.5 are almost all CIFAKE after brutal downscale/blur, plus a handful of high-detail real photos.

## Limitations (what we would do with more time)

- **Generator shift.** LoRA is fit on SID_Set (and CIFAKE only for the tiny head). Unseen commercial tools (newer Midjourney / Firefly / etc.) can drift. The frozen CLIP head was built for that; we did not re-benchmark it on Synthbuster.
- **Tiny + heavy laundering.** 0.25× resize of 32×32 (AUC 0.85) is the worst cell. A dedicated low-res restoration path would help more than more SID epochs.
- **False positives.** High-frequency real textures (OpenImages crops) are the SID FP mode. We kept threshold 0.5 rather than chasing 100% fake-catch with a lower cut.
- **CLIP 224 crop.** Both heads see a resized center crop. Full-res forensics (noiseprint, JPEG grid) are unused.
- **No production stack.** No moderation UI, no video/audio, no calibrated probabilities beyond sigmoid(LLR).

**Do not train more SID LoRA** — cached val AUC was already ~1.0 after two epochs; more epochs overfit JPEG cache.

## Team

Add names before Devpost:

- _TBD_

## License / upstream

Apache-2.0 as in `LICENSE.md`. Paper: [Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195v2). Code origin: [grip-unina/ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection).
