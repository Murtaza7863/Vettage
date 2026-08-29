#!/usr/bin/env python3
"""Load the CLIP detector and print / persist parameter counts. Must stay < 2B."""

from __future__ import annotations

import argparse
import json
import os

from detector import count_loaded_params, load_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="clipdet_latent10k_plus")
    parser.add_argument("--weights_dir", default="./weights")
    parser.add_argument("--out", default="./results/param_count.json")
    args = parser.parse_args()

    model, _, device, arch = load_detector(args.model, args.weights_dir, device="cpu")
    counts = count_loaded_params(model)
    counts["arch"] = arch
    counts["model_name"] = args.model
    counts["device_used_for_count"] = str(device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts, indent=2))
    assert counts["under_2b"], "Parameter count exceeds 2B"


if __name__ == "__main__":
    main()
