import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Main entry point: TinyStories + Long-Range Key Retrieval Probe experiment.

Runs a weight-decay sweep as specified in the experiment design:
  wd ∈ {0.0, 1e-4, 3e-4, 1e-3, 3e-3}

For each wd, trains a GPT-small on TinyStories with probe injection,
monitors geometric signals (commutator defect + PCA), and logs
probe accuracy (in-distribution and OOD).

Usage:
    python main.py                          # full sweep
    python main.py --wd 1e-3                # single weight decay
    python main.py --wd 1e-3 --steps 5000   # quick test
    python main.py --wd 1e-3 --control 0.3  # with capability control
"""

import argparse
import json
import time
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from dataset import build_datasets
from train import train


WEIGHT_DECAYS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_sweep(args):
    """Run the weight-decay sweep (or a single run)."""

    if args.wd is not None:
        weight_decays = [args.wd]
    else:
        weight_decays = WEIGHT_DECAYS

    # Build datasets once (shared across all runs)
    cfg = Config(seed=args.seed)
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.p_probe:
        cfg.p_probe = args.p_probe

    set_seed(cfg.seed)

    print(f"{'='*70}")
    print(f"  TinyStories + Long-Range Key Retrieval Probe")
    print(f"  Weight decay sweep: {weight_decays}")
    print(f"  Total steps: {cfg.total_steps}")
    print(f"  Seed: {cfg.seed}")
    print(f"{'='*70}\n")

    data = build_datasets(cfg)

    all_results = {}

    for wd in weight_decays:
        run_cfg = Config(
            seed=cfg.seed,
            weight_decay=wd,
            total_steps=cfg.total_steps,
            batch_size=cfg.batch_size,
            p_probe=cfg.p_probe,
            control_lambda=args.control if args.control is not None else 0.0,
        )
        if args.steps:
            run_cfg.total_steps = args.steps

        lam_tag = f"_lam{args.control}" if args.control else ""
        run_tag = f"wd{wd}_s{cfg.seed}{lam_tag}"

        out_dir = Path(run_cfg.log_dir) / run_tag
        results_path = out_dir / "results.pt"

        if results_path.exists() and not args.force:
            print(f"\n[SKIP] {run_tag} — already exists. Use --force to overwrite.")
            cached = torch.load(results_path, map_location="cpu", weights_only=False)
            all_results[run_tag] = cached
            continue

        print(f"\n{'='*70}")
        print(f"  RUN: {run_tag}")
        print(f"{'='*70}")

        set_seed(run_cfg.seed)
        result = train(run_cfg, data, run_tag=run_tag)
        all_results[run_tag] = result

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'run':>30s}  {'val_loss':>8s}  {'probe_in':>8s}  "
          f"{'probe_ood':>9s}  {'emergence':>9s}  {'defect_on':>9s}  {'lead':>6s}")
    print(f"  {'---':>30s}  {'---':>8s}  {'---':>8s}  "
          f"{'---':>9s}  {'---':>9s}  {'---':>9s}  {'---':>6s}")

    for tag, result in all_results.items():
        metrics = result.get("metrics", [])
        if metrics:
            last = metrics[-1]
            val_loss = last.get("val_loss", 0)
            probe_in = last.get("probe_in_acc", 0)
            probe_ood = last.get("probe_ood_acc", 0)
        else:
            val_loss = probe_in = probe_ood = 0

        emergence = result.get("emergence_step")
        defect_on = result.get("defect_onset")
        lead = result.get("lead_time")

        em_str = str(emergence) if emergence is not None else "---"
        do_str = str(defect_on) if defect_on is not None else "---"
        lead_str = str(lead) if lead is not None else "---"

        print(f"  {tag:>30s}  {val_loss:8.4f}  {probe_in:8.3f}  "
              f"{probe_ood:9.3f}  {em_str:>9s}  {do_str:>9s}  {lead_str:>6s}")

    # Save sweep summary
    summary_path = Path(cfg.log_dir) / "sweep_summary.json"
    summary = {}
    for tag, result in all_results.items():
        summary[tag] = {
            "emergence_step": result.get("emergence_step"),
            "defect_onset": result.get("defect_onset"),
            "lead_time": result.get("lead_time"),
            "final_val_loss": result["metrics"][-1]["val_loss"] if result.get("metrics") else None,
            "final_probe_ood": result["metrics"][-1]["probe_ood_acc"] if result.get("metrics") else None,
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Sweep summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="TinyStories + Probe Grokking Experiment")
    parser.add_argument("--wd", type=float, default=None,
                       help="Single weight decay value (skip sweep)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Override total training steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--p-probe", type=float, default=None,
                       help="Probe injection probability")
    parser.add_argument("--control", type=float, default=None,
                       help="Capability control lambda (0=off, 0.3, 1.0)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing results")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
