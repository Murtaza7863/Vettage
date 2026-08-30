# Error analysis (Deliverable #5)

Threshold **0.5**. Catalog images (closest-to-boundary, not always misclassified) live in `results/error_examples/` (local only; not in git).

## Counts on the full held-out val (n=1600)

From `results/success_lora_tiny.json` after SID LoRA + tiny-image head:

| Split | n | FP (real→fake) | FN (fake→real) | Notes |
|---|---|---|---|---|
| SID_Set | 1000 | **1** | **0** | Native-res; fake-catch 100% |
| CIFAKE 32×32 | 600 | **7** | **9** | Tiny head; acc 97.3% |
| Overall | 1600 | **8** | **9** | acc 98.9%, AUC 0.999 |

The frozen CLIP head on the same val: 146 FP / 610 FN. Almost all of the remaining errors are CIFAKE, not SID.

## Closest-to-boundary (SID slice, n=200, native files)

These are the *nearest* reals/fakes, not failures (none of them cross 0.5):

| Condition | Highest real pred | Lowest fake pred |
|---|---|---|
| clean | 0.244 | 0.999 |
| JPEG q=30 | 0.023 | 0.827 |
| blur σ=2 | 0.091 | 1.000 |
| resize 0.25× | 0.142 | 0.999 |
| center crop 80% | 0.284 | 0.859 |

Patterns:

- **SID false-positive texture:** high-detail real OpenImages crops (`68b1ceaff0bd8d7b` and similar). Score 0.24 still called real.
- **SID fake scores after JPEG q=30** drop (lowest 0.827) but stay above threshold. Compression is the transform that most compresses the SID margin; degradation training was meant for this.
- **Center crop 80%** is the next-softest SID fake (0.859) when the remaining region is smoother (sky/wall).
- Blur σ=2 and 0.25× **do not** collapse SID after LoRA.

## CIFAKE / tiny-image trade-off

CIFAKE is 32×32 upsampled to CLIP’s 224px. A single SID LoRA head **inverts** that domain (reals score higher than fakes). The gated tiny head (min side ≤ 64) fixes clean CIFAKE (fake-catch 21% → 97%) without moving SID.

What still hurts CIFAKE (from `results/robustness_lora.csv`, mixed 600-image eval):

- 0.25× then upscale: CIFAKE AUC **0.845** (32→8→32)
- blur σ=2: **0.879**
- noise σ=0.10: **0.864**
- JPEG q=30 still **0.988**

Trade-off we accepted: **do not mix CIFAKE into SID LoRA**. A joint head would raise tiny-image recall by wrecking native-res precision — the opposite of a social-media / camera-photo deployment.

## What we would inspect next

Unseen generators (WildFake **demo** split, Synthbuster) as a display eval only — not as training data. High-detail real FPs if the operating threshold is lowered for “highest detection.”
