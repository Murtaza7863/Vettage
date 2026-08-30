# MODEL_INFO

Copy-paste ready inventory for the Devpost written description. Parameter counts measured by loading the CLIP backbone + classifier head (`python count_params.py`) and the LoRA payload in `checkpoints/lora/lora_best.pt`.

## Constraint

**Total loaded parameters must be < 2,000,000,000.**

| Component | Parameters | Notes |
|---|---|---|
| CLIP vision encoder (ViT-L/14 CommonPool) | **303,179,776** | used at inference |
| CLIP text encoder | 85,054,464 | loaded with OpenCLIP, unused by `predict.py` |
| Linear classifier head (SID) | **1,025** | trained with LoRA |
| Tiny-image head (CIFAKE, min-side ≤ 64) | **1,025** | logistic probe on frozen LoRA features |
| LoRA adapters (`c_fc`, `c_proj`, r=16 α=32) | **3,932,160** | only trainable CLIP weights |
| **Inference total (vision + LoRA + both heads)** | **307,113,986** | 15.4% of the 2B cap |
| **Loaded OpenCLIP module + LoRA + heads** | **430,764,291** | 21.5% of the 2B cap |
| Under 2B? | **YES** | |

Trainable at fine-tune time: **3,933,185** (LoRA + head). CLIP base weights stay frozen.

## Dev tools

- Python 3.11 (conda env `vettage`)
- Git + Git LFS (pretrained detector heads)
- macOS Apple Silicon (MPS) for local inference/training; CUDA via `--device`

## Models / checkpoints

- Backbone: OpenCLIP **ViT-L/14** pretrained on CommonPool (`laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K`)
- Detector head origin: GRIP-UNINA `clipdet_latent10k_plus` (Cozzolino et al., CVPRW 2024)
- Fine-tune: `checkpoints/lora/lora_best.pt` — LoRA on vision MLP + SID linear head + tiny-image head. Default for `predict.py`.
- Optional unused checkpoint: `Corvi2023` ResNet-50 (`res50nodown`, ~25M) — not loaded by `predict.py`

## Libraries / frameworks

- `torch`, `torchvision`, `open_clip_torch`, `timm`, `huggingface-hub`
- `peft` (LoRA)
- `pillow`, `pandas`, `scikit-learn`, `pyyaml`, `tqdm`, `matplotlib`
- `datasets` (Hugging Face SID_Set / CIFAKE loaders)

## Datasets / assets used

- **Not used for training:** WildFake / COCO val2017 demo-validation subset from the problem statement.
- Pretrained CLIP weights from LAION / OpenCLIP.
- Detector linear head from `grip-unina/ClipBased-SyntheticImageDetection`.
- Fine-tune data: `saberzl/SID_Set` **train split only** for LoRA (real vs full-synthetic). CIFAKE (`dragonintelligence/CIFAKE-image-dataset`) trains **only** the tiny-image head (images with min side ≤ 64). Not mixed into SID LoRA.
- Held-out val carved from that same training pool after SHA1 dedup — not SID official val/test.

## Inference contract

`predict.py --input_dir DIR --output_json OUT.json` writes:

```json
[{"image_path": "...", "pred": 0.0}, {"image_path": "...", "pred": 1.0}]
```

`pred` is `sigmoid(LLR)` in `[0, 1]`. Values **> 0.5** mean synthetic (LoRA operating point). Pass `--checkpoint none` to restore the frozen baseline head.

## Held-out results (native-res val, threshold 0.5)

| Split | n | Baseline | LoRA SID only | LoRA + tiny head |
|---|---|---|---|---|
| SID_Set | 1000 | 0.859 / 61.7% / 26.2% | **1.000 / 99.9% / 100%** | **1.000 / 99.9% / 100%** |
| CIFAKE | 600 | 0.315 / 37.8% / 19.7% | 0.656 / 57.2% / 21.0% | **0.997 / 97.3% / 97.0%** |
| Overall | 1600 | 0.666 / 52.7% / 23.8% | 0.949 / 83.9% / 70.4% | **0.999 / 98.9% / 98.9%** |
