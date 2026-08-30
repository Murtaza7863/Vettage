#!/usr/bin/env python3
"""Run the three generalization-v2 reports. Never writes to checkpoints/lora."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/gen_v2/lora_best.pt")
    p.add_argument("--tag", default="gen_v2")
    p.add_argument("--skip_tiny", action="store_true")
    p.add_argument("--skip_robustness", action="store_true")
    args = p.parse_args()
    if os.path.abspath(args.checkpoint) == os.path.abspath("checkpoints/lora/lora_best.pt"):
        raise SystemExit("refusing to eval-write the main checkpoint path")

    out = "results/gen_v2"
    os.makedirs(out, exist_ok=True)

    if not args.skip_tiny:
        run(
            [
                sys.executable,
                "train_tiny_head.py",
                "--checkpoint",
                args.checkpoint,
                "--out_ckpt",
                args.checkpoint,
                "--train_csv",
                "data/processed/train.csv",
                "--val_csv",
                "data/processed/val.csv",
            ]
        )

    run(
        [
            sys.executable,
            "eval_success.py",
            "--val_csv",
            "data/processed/val.csv",
            "--checkpoint",
            args.checkpoint,
            "--out_json",
            os.path.join(out, "success_in_family.json"),
        ]
    )
    run(
        [
            sys.executable,
            "eval_success.py",
            "--val_csv",
            "data/processed/gen_v2/val_unseen_midjourney.csv",
            "--checkpoint",
            args.checkpoint,
            "--out_json",
            os.path.join(out, "success_unseen_midjourney.json"),
        ]
    )
    if not args.skip_robustness:
        run(
            [
                sys.executable,
                "eval_robustness.py",
                "--val_csv",
                "data/processed/val.csv",
                "--checkpoint",
                args.checkpoint,
                "--tag",
                args.tag,
                "--out_csv",
                os.path.join(out, "robustness.csv"),
                "--out_png",
                os.path.join(out, "robustness.png"),
            ]
        )

    inf = json.load(open(os.path.join(out, "success_in_family.json")))
    ood = json.load(open(os.path.join(out, "success_unseen_midjourney.json")))
    main_ood_path = os.path.join(out, "success_unseen_midjourney_main.json")
    main_inf_path = "results/success_lora_tiny.json"
    report = {
        "checkpoint": args.checkpoint,
        "in_family": inf.get("by_source", {}),
        "in_family_overall": inf.get("overall"),
        "unseen_midjourney": ood.get("overall"),
    }
    if os.path.isfile(main_inf_path):
        report["main_in_family"] = json.load(open(main_inf_path)).get("by_source")
    if os.path.isfile(main_ood_path):
        report["main_unseen_midjourney"] = json.load(open(main_ood_path)).get("overall")
    with open(os.path.join(out, "compare.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
