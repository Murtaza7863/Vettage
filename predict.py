#!/usr/bin/env python3
"""Directory-in, JSON-out adapter around the CLIP synthetic-image detector.

Converts the detector's log-likelihood ratio (LLR) to a [0, 1] confidence
via sigmoid. Required output records: {image_path, pred}.

Each image is scored --repeats times (reload from disk). `pred` is the mean.
A folder-level aggregate is printed and written next to the predictions JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from detector import (
    aggregate_predictions,
    count_loaded_params,
    list_images,
    load_detector,
    load_rgb,
    score_images_repeated,
    score_pil_images,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic-image detection over a folder.")
    parser.add_argument("--input_dir", required=True, help="Directory of images (walked recursively).")
    parser.add_argument("--output_json", required=True, help="Path to write JSON predictions.")
    parser.add_argument(
        "--aggregate_json",
        default=None,
        help="Path for the folder aggregate. Default: <output_json> with .aggregate.json suffix.",
    )
    parser.add_argument(
        "--weights_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
    )
    parser.add_argument(
        "--model",
        default="clipdet_latent10k_plus",
        help="Weight folder under --weights_dir. Default: CLIP ViT-L/14 detector.",
    )
    parser.add_argument("--device", default=None, help="cpu | mps | cuda. Auto-detected if omitted.")
    parser.add_argument("--batch_size", type=int, default=8, help="Unused when --repeats > 0 (kept for CLI compat).")
    parser.add_argument("--repeats", type=int, default=5, help="Independent forwards per image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="pred > threshold counts as synthetic.")
    parser.add_argument(
        "--probe_transforms",
        action="store_true",
        help="Also score each image under Track 5 degradations (same photo, changed pixels).",
    )
    parser.add_argument(
        "--log_params",
        action="store_true",
        help="Print backbone + head parameter counts before inference.",
    )
    return parser.parse_args()


def _name(path: str) -> str:
    return os.path.basename(path)


def print_repeat_table(rows: list[dict], aggregate: dict) -> None:
    n_rep = int(aggregate.get("n_repeats") or 1)
    name_w = max((len(_name(r["image_path"])) for r in rows), default=12)
    name_w = max(name_w, 12)
    hdr = f"{'image':<{name_w}}"
    for i in range(n_rep):
        hdr += f"  {f'r{i+1}':>8}"
    hdr += f"  {'mean':>8}  {'std':>8}  {'range':>8}  call"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{_name(r['image_path']):<{name_w}}"
        reps = r.get("repeats") or [r["pred"]]
        for v in reps:
            line += f"  {v:8.5f}"
        call = "synthetic" if r["pred"] > aggregate["threshold"] else "real"
        line += f"  {r['pred']:8.5f}  {r.get('pred_std', 0.0):8.5g}  {r.get('pred_range', 0.0):8.5g}  {call}"
        print(line)
    print("-" * len(hdr))
    print(
        f"AGGREGATE  n={aggregate['n_images']}  repeats={n_rep}  "
        f"mean={aggregate['mean_pred']:.5f}  std={aggregate['std_pred']:.5f}  "
        f"median={aggregate['median_pred']:.5f}  "
        f"min={aggregate['min_pred']:.5f}  max={aggregate['max_pred']:.5f}"
    )
    print(
        f"           synthetic={aggregate['n_synthetic']}  real={aggregate['n_real']}  "
        f"frac_synthetic={aggregate['frac_synthetic']:.3f}  "
        f"threshold={aggregate['threshold']}"
    )
    print(
        f"           within-image std mean={aggregate['mean_within_image_std']:.5g}  "
        f"max={aggregate['max_within_image_std']:.5g}  "
        f"max range={aggregate['max_within_image_range']:.5g}  "
        f"stable={aggregate['stable']}"
    )


def print_transform_probe(paths, model, transform, device, batch_size: int = 8) -> list[dict]:
    from robust_transforms import eval_conditions

    conditions = eval_conditions()
    images = [load_rgb(p) for p in paths]
    per_cond = {}
    for spec in conditions:
        scored = score_pil_images(
            [spec.apply(im) for im in images], model, transform, device, batch_size=batch_size
        )
        per_cond[(spec.name, spec.params)] = [float(x) for x in scored]

    clean_key = ("clean", "none")
    name_w = max(len(os.path.basename(p)) for p in paths)
    name_w = max(name_w, 12)
    keys = list(per_cond.keys())
    hdr = f"{'image':<{name_w}}"
    for name, params in keys:
        label = "clean" if name == "clean" else f"{name[:4]}_{params.split('=')[-1][:6]}"
        hdr += f"  {label:>10}"
    print("\nSame-image transform probe (score should move only when pixels change):")
    print(hdr)
    print("-" * len(hdr))
    probe_rows = []
    for i, p in enumerate(paths):
        line = f"{os.path.basename(p):<{name_w}}"
        rec = {"image_path": p, "scores": {}}
        for key in keys:
            v = per_cond[key][i]
            rec["scores"][f"{key[0]}|{key[1]}"] = v
            line += f"  {v:10.5f}"
        clean = per_cond[clean_key][i]
        rec["clean"] = clean
        rec["max_abs_delta"] = max(abs(per_cond[k][i] - clean) for k in keys)
        probe_rows.append(rec)
        print(line)
    print("-" * len(hdr))
    print("mean Δ vs clean:")
    for name, params in keys:
        if name == "clean":
            continue
        deltas = [per_cond[(name, params)][i] - per_cond[clean_key][i] for i in range(len(paths))]
        print(f"  {name:16s} {params:24s}  meanΔ={sum(deltas)/len(deltas):+.4f}  "
              f"max|Δ|={max(abs(d) for d in deltas):.4f}")
    return probe_rows


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.repeats < 1:
        print("error: --repeats must be >= 1", file=sys.stderr)
        sys.exit(1)

    paths = list_images(input_dir)
    if not paths:
        print(f"error: no images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    model, transform, device, arch = load_detector(
        model_name=args.model,
        weights_dir=args.weights_dir,
        device=args.device,
    )
    counts = count_loaded_params(model)
    print(f"arch={arch} device={device} images={len(paths)} repeats={args.repeats}")
    if args.log_params:
        for k, v in counts.items():
            print(f"  {k}: {v}")
    if not counts["under_2b"]:
        print("ERROR: loaded module exceeds 2B parameters — disqualified.", file=sys.stderr)
        sys.exit(2)

    rows = score_images_repeated(paths, model, transform, device, n_repeats=args.repeats)
    aggregate = aggregate_predictions(rows, threshold=args.threshold)
    print_repeat_table(rows, aggregate)

    probe_rows = None
    if args.probe_transforms:
        probe_rows = print_transform_probe(paths, model, transform, device, batch_size=max(args.batch_size, 1))
        aggregate["transform_probe"] = [
            {
                "image_path": r["image_path"],
                "clean": r["clean"],
                "max_abs_delta": r["max_abs_delta"],
                "scores": r["scores"],
            }
            for r in probe_rows
        ]

    payload = []
    for r in rows:
        payload.append(
            {
                "image_path": r["image_path"],
                "pred": r["pred"],
                "pred_std": r["pred_std"],
                "pred_min": r["pred_min"],
                "pred_max": r["pred_max"],
                "n_repeats": r["n_repeats"],
                "repeats": r["repeats"],
            }
        )

    out_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    if args.aggregate_json:
        agg_path = os.path.abspath(args.aggregate_json)
    else:
        root, ext = os.path.splitext(out_path)
        agg_path = root + ".aggregate.json"
    os.makedirs(os.path.dirname(agg_path) or ".", exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({"aggregate": aggregate, "predictions": payload}, f, indent=2)

    print(f"wrote {len(payload)} predictions -> {out_path}")
    print(f"wrote aggregate -> {agg_path}")


if __name__ == "__main__":
    main()
