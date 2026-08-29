# MODEL_INFO

Copy-paste ready inventory for the Devpost written description. Parameter counts below were measured by loading the actual CLIP backbone + classifier head (`python count_params.py`).

## Constraint

**Total loaded parameters must be < 2,000,000,000.** Verified before committing to this backbone.

Measured 2026-08-29 by `python count_params.py` after loading `clipdet_latent10k_plus` (`opencliplinearnext_clipL14commonpool`). Source of truth also at `results/param_count.json`.

| Component | Parameters | Notes |
|---|---|---|
| CLIP vision encoder (ViT-L/14 CommonPool) | **303,179,776** | used at inference |
| CLIP text encoder | 85,054,464 | loaded with OpenCLIP, unused by `predict.py` |
| Linear classifier head | **1,025** | `weights/clipdet_latent10k_plus/weights.pth` |
| **Inference total (vision + head)** | **303,180,801** | 15.2% of the 2B cap |
| **Loaded OpenCLIP module + head** | **426,831,106** | 21.3% of the 2B cap |
| Under 2B? | **YES** | `426,831,106 < 2,000,000,000` |

## Dev tools

- Python 3.11 (conda env `vettage`)
- Git + Git LFS (pretrained detector heads)
- macOS Apple Silicon (MPS) for local inference/training; CUDA accepted via `--device`

## Models / checkpoints

- Backbone: OpenCLIP **ViT-L/14** pretrained on CommonPool (`laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K`)
- Detector head: GRIP-UNINA `clipdet_latent10k_plus` (Cozzolino et al., CVPRW 2024)
- Optional unused-in-default-path checkpoint: `Corvi2023` ResNet-50 (`res50nodown`, ~25M params) — not loaded by `predict.py`

## Libraries / frameworks

- `torch`, `torchvision`
- `open_clip_torch`
- `timm`, `huggingface-hub`
- `pillow`, `pandas`, `scikit-learn`, `pyyaml`, `tqdm`
- `peft` (LoRA, Phase 4)
- `datasets` (Hugging Face SID_Set / CIFAKE loaders)

## Datasets / assets used

- **Not used for training:** WildFake / COCO val2017 demo-validation subset from the problem statement.
- Pretrained CLIP weights from LAION / OpenCLIP (feature extractor only).
- Detector linear head from `grip-unina/ClipBased-SyntheticImageDetection`.
- Training data (Phase 2): `saberzl/SID_Set` (Hugging Face) and CIFAKE (`dragonintelligence/CIFAKE-image-dataset` Hugging Face mirror of `birdy654/cifake-real-and-ai-generated-synthetic-images`).

## Inference contract

`predict.py --input_dir DIR --output_json OUT.json` writes:

```json
[{"image_path": "...", "pred": 0.0}, {"image_path": "...", "pred": 1.0}]
```

`pred` is `sigmoid(LLR)` in `[0, 1]`. Values > 0.5 mean the detector leans synthetic.
