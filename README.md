# Vettage — Track 5: Robust AI-Generated Image Detection

Prototype detector for **image-level AIGC vs authentic** photos. The challenge is not clean lab images — it is JPEG re-encodes, blur, thumbnails, noise, filters, and crops after an image has been posted. This repo is a CLIP ViT-L/14 detector (Cozzolino et al., CVPRW 2024) with a **LoRA adapter** fine-tuned on diverse generators and a **second linear head** for tiny (CIFAR-scale) images.

Loaded module ≈ **431M parameters** (vision + unused text encoder + LoRA + two heads). Under the **2B** cap. **Not trained** on the WildFake / COCO val2017 demonstration split.

`pred > 0.5` means synthetic. `pred` is `sigmoid(LLR)` in `[0, 1]`.

## Setup

Python 3.11+, Git LFS (pretrained heads in `weights/`).

```bash
git clone https://github.com/Murtaza7863/Vettage.git
cd Vettage
git lfs pull
pip install -r requirements.txt
pip install peft datasets
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

## How It Works

1. **Backbone:** OpenCLIP ViT-L/14 CommonPool (303M params, frozen). Linear head from GRIP-UNINA `clipdet_latent10k_plus`.
2. **LoRA fine-tuning:** Low-Rank Adaptation on ViT MLP layers (`c_fc` / `c_proj`), rank=32, alpha=64. Only ~7.9M trainable parameters — the full CLIP backbone stays frozen.
3. **Native photos (short side > 64px):** LoRA adapters + SID head. Trained on diverse generators with random Track 5 degradations (70% probability per image).
4. **Tiny images (short side ≤ 64px):** Same CLIP features, **separate** 1025-param logistic head fit on CIFAKE. Mixing CIFAKE into the SID LoRA inverted CLIP (32×32 upsampled to 224 looks "fake" even when real).
5. CLIP base weights stay frozen. Total trainable during LoRA ≈ 7.9M.

## Training Data

The model is trained on **4 diverse sources** spanning 8+ generators to maximize cross-generator generalization:

| Source | Real | Fake | Generators | Purpose |
|---|---|---|---|---|
| SID_Set | 2,000 | 2,000 | Multiple (research benchmark) | Core real/fake pairs |
| CIFAKE | 4,000 | 4,000 | Stable Diffusion v1.4 (32×32) | Tiny-image head only |
| Defactify | 3,000 | 3,000 | SD 2.1, SDXL, SD3, DALL·E 3, Midjourney v6 | Modern generator coverage |
| SynthWildX | — | ~1,500 | DALL·E 3, Midjourney, Firefly (scraped from Twitter/X) | In-the-wild, post-compression images |

All data is SHA1-deduplicated. Val is a 10% stratified split from the training pool, held out by source and label. The competition's WildFake / COCO val2017 demo set is **never** used in training.

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.05 |
| Epochs | 8 |
| Batch size | 8 |
| LoRA rank / alpha | 32 / 64 |
| Degradation probability | 0.7 |
| Loss | BCE with logits (pos_weight=2.0) |
| Validation metric | SID_Set AUC (best checkpoint saved) |

## Training-Time Robustness Augmentation

Each training image has a 70% chance of being degraded with one of the competition's specified transforms:

| Transform | Parameters | Real-World Analog |
|---|---|---|
| JPEG Compression | quality = 30, 50, 70, 90 | Social media re-encode |
| Gaussian Blur | σ = 0.5, 1.0, 2.0 | Out-of-focus, screenshots |
| Resize + Upscale | scale 0.25×, 0.5× then back | Thumbnail generation |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color Jitter | B/C/S ±20% | Filter apps, auto-enhance |
| Center Crop | 80% | Profile picture cropping |

## Parameter Budget

| Component | Parameters |
|---|---|
| CLIP Vision Encoder (ViT-L/14) | 303M |
| LoRA Adapters (c_fc, c_proj) | 7.9M |
| Classifier Head (SID) | 1,025 |
| Tiny-Image Head (CIFAKE) | 1,025 |
| **Inference total** | **~311M (15.5% of 2B cap)** |

## Results (held-out val, never in train)

`data/processed/val.csv`, n=1799, threshold 0.5.

| Split | n | Accuracy | AUC | Real OK | Fake OK |
|---|---|---|---|---|---|
| Overall | 1799 | 0.990 | 1.000 | 98.0% | 100% |
| SID_Set | 400 | 1.000 | 1.000 | 100% | 100% |
| CIFAKE | 799 | 0.977 | 0.999 | 95.5% | 100% |
| Defactify | 600 | 1.000 | 1.000 | 100% | 100% |

Confusion matrix (overall): TN=882, FP=18, FN=0, TP=899.

### Robustness (Track 5 transforms)

| Condition | Overall AUC | SID AUC | CIFAKE AUC |
|---|---|---|---|
| clean | 1.000 | 1.000 | 0.999 |
| JPEG q=90 | 1.000 | 1.000 | 0.999 |
| JPEG q=30 | 0.998 | 1.000 | 0.988 |
| blur σ=2.0 | 0.981 | 1.000 | 0.879 |
| resize 0.25× then up | 0.974 | 1.000 | 0.845 |
| noise σ=0.10 | 0.979 | 1.000 | 0.864 |
| jitter ±20% | 0.997 | 1.000 | 0.984 |
| center crop 80% | 0.997 | 1.000 | 0.981 |

SID stays ~1.0 AUC under every listed transform. Residual misses are tiny images after heavy blur / 0.25× downscale.

## Reproduce

Data is **not** in git (`data/processed/` is gitignored). Rebuild:

```bash
# Step 1: Download and prepare all datasets
python dataset.py --out_dir data/processed

# Step 2: (Optional) Download SynthWildX in-the-wild images
cd data/synthwildx
python download_synthwildx.py
cd ../..

# Step 3: Train LoRA
python train_lora.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --device mps \
  --epochs 8

# Step 4: (Optional) Resume training from an existing checkpoint
python train_lora.py \
  --train_csv data/processed/train.csv \
  --val_csv data/processed/val.csv \
  --device mps \
  --resume checkpoints/lora/lora_best.pt

# Step 5: Train tiny-image head
python train_tiny_head.py --checkpoint checkpoints/lora/lora_best.pt

# Step 6: Evaluate
python eval_success.py \
  --val_csv data/processed/val.csv \
  --checkpoint checkpoints/lora/lora_best.pt \
  --device mps

python eval_robustness.py \
  --val_csv data/processed/val.csv \
  --checkpoint checkpoints/lora/lora_best.pt \
  --device mps
```

For CUDA (Linux/Windows with NVIDIA GPU), replace `--device mps` with `--device cuda`.

Never pass the WildFake / COCO val2017 demo split into training.

## Error Analysis

See `results/error_analysis.md`. Short version: on native SID, the decision boundary is wide (highest real ≈ 0.24, lowest fake ≈ 0.83 even after JPEG q=30). Remaining errors at threshold 0.5 are almost all CIFAKE after brutal downscale/blur, plus a handful of high-detail real photos.

## Limitations & Future Work

- **Generator shift.** While the model now covers 8+ generators (SD 2.1/XL/3, DALL·E 3, Midjourney v6, Firefly), newer or unseen generators can still evade detection. Adding GenImage (8 more generators) or Synthbuster would further improve coverage.
- **Real art vs AI art.** The model's "real" class is mostly photographic (COCO, SID). Stylized real artwork (paintings, illustrations, digital art) can be falsely flagged as AI-generated. Adding real artwork to the training set would address this.
- **Tiny + heavy laundering.** 0.25× resize of 32×32 images (AUC 0.85) is the worst case. A dedicated low-resolution restoration path would help.
- **False positives.** High-frequency real textures remain the primary FP mode. Threshold is kept at 0.5 for balanced precision/recall.
- **CLIP 224 crop.** Both heads see a resized center crop. Full-resolution forensics (noiseprint, JPEG grid analysis) are unused.
- **Explainability.** GradCAM or attention visualizations would help users understand why the model flags an image, improving trust and debuggability.
- **No production stack.** No moderation UI, no video/audio support, no calibrated probabilities beyond sigmoid(LLR).

## Team

- Supa-att Tadderm
- Murtaza Kuvawala Abbas
- Shen Dayang
- Elizebeth Felice
- Annya Sriram

## License / Upstream

Apache-2.0 as in `LICENSE.md`. Paper: [Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195v2). Code origin: [grip-unina/ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection).

## License / upstream

Apache-2.0 as in `LICENSE.md`. Paper: [Raising the Bar of AI-generated Image Detection with CLIP](https://arxiv.org/abs/2312.00195v2). Code origin: [grip-unina/ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection).
