#!/usr/bin/env python3
"""Directory-in, JSON-out adapter around the CLIP synthetic-image detector.

Converts the detector's log-likelihood ratio (LLR) to a [0, 1] confidence
via sigmoid. Required output records: {image_path, pred}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from detector import count_loaded_params, list_images, load_detector, score_images


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic-image detection over a folder.")
    parser.add_argument("--input_dir", required=True, help="Directory of images (walked recursively).")
    parser.add_argument("--output_json", required=True, help="Path to write JSON predictions.")
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--log_params",
        action="store_true",
        help="Print backbone + head parameter counts before inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
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
    if args.log_params or True:
        print(f"arch={arch} device={device}")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        if not counts["under_2b"]:
            print("ERROR: loaded module exceeds 2B parameters — disqualified.", file=sys.stderr)
            sys.exit(2)

    rows = score_images(paths, model, transform, device, batch_size=args.batch_size)
    payload = [{"image_path": r["image_path"], "pred": r["pred"]} for r in rows]

    out_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {len(payload)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
