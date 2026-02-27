import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Estimate eval noise floor for p_ood via bootstrap resampling.

At a fixed checkpoint (no training), evaluate p_ood on N_BOOT bootstrap
samples of the probe_eval_ood dataset.  Writes eval_noise.json with:
  - std_p_ood:  standard deviation of bootstrap distribution
  - mean_p_ood: mean of bootstrap distribution
  - n_boot:     number of bootstrap samples
  - ckpt_step:  checkpoint step used
  - n_eval:     size of eval dataset

The noise floor lets detect_oscillations.py classify switch-pair deltas
as significant (delta >= k*std) vs noise-level.

Usage:
    python estimate_eval_noise.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s271/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe
from attractor_analysis import load_checkpoint


N_BOOT = 20  # bootstrap iterations


def bootstrap_eval(model, probe_dataset, device, n_boot=N_BOOT, seed=0):
    """Run n_boot bootstrap evaluations of p_ood.

    Each iteration samples len(dataset) indices with replacement from
    the eval set, evaluates p_ood on that subset.

    Returns list of n_boot p_ood values.
    """
    model.eval()
    n = len(probe_dataset)
    rng = np.random.RandomState(seed)

    results = []
    for i in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        subset = Subset(probe_dataset, indices.tolist())
        p = evaluate_probe(model, subset, device)
        results.append(p)
        print(f"  boot {i+1:2d}/{n_boot}: p_ood = {p:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Estimate p_ood eval noise via bootstrap"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ckpt-step", type=int, default=None,
                        help="Checkpoint step to use (default: last available)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap iterations (default: {N_BOOT})")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = get_device()

    # Find checkpoint
    if args.ckpt_step is not None:
        ckpt_step = args.ckpt_step
    else:
        # Use last checkpoint
        ckpt_files = sorted(run_dir.glob("ckpt_*.pt"))
        if not ckpt_files:
            print("ERROR: No checkpoints found", file=sys.stderr)
            sys.exit(1)
        ckpt_step = int(ckpt_files[-1].stem.split("_")[1])

    print(f"Noise estimation: {args.n_boot} bootstrap samples @ step {ckpt_step}")
    print(f"Device: {device}")

    # Load model
    cfg = Config(
        seed=42,  # codeword seed always 42
        p_probe=0.10, batch_size=64,
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )

    cw_path = run_dir / "codewords.json"
    data = build_datasets(
        cfg, codewords_path=str(cw_path) if cw_path.exists() else None,
    )
    probe_ood = data["probe_eval_ood"]
    vocab_size = len(data["tokenizer"])

    ckpt = load_checkpoint(run_dir, ckpt_step, device=device)
    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    del ckpt

    # Also get deterministic full-set eval for reference
    with torch.no_grad():
        full_p_ood = evaluate_probe(model, probe_ood, device)
    print(f"Full-set p_ood (deterministic): {full_p_ood:.4f}")

    # Bootstrap
    print(f"\nBootstrap ({args.n_boot} iterations):")
    boot_values = bootstrap_eval(model, probe_ood, device,
                                  n_boot=args.n_boot)

    mean_p = np.mean(boot_values)
    std_p = np.std(boot_values, ddof=1)  # sample std

    print(f"\nResults:")
    print(f"  mean(p_ood) = {mean_p:.4f}")
    print(f"  std(p_ood)  = {std_p:.4f}")
    print(f"  2*std       = {2*std_p:.4f}")
    print(f"  3*std       = {3*std_p:.4f}")
    print(f"  full-set    = {full_p_ood:.4f}")

    # Binomial SE for comparison
    binom_se = np.sqrt(full_p_ood * (1 - full_p_ood) / len(probe_ood))
    print(f"  binomial SE = {binom_se:.4f} (theoretical)")

    # Save
    result = {
        "std_p_ood": round(std_p, 6),
        "mean_p_ood": round(mean_p, 4),
        "full_p_ood": round(full_p_ood, 4),
        "n_boot": args.n_boot,
        "n_eval": len(probe_ood),
        "ckpt_step": ckpt_step,
        "binomial_se": round(binom_se, 6),
        "boot_values": [round(v, 4) for v in boot_values],
    }

    out_path = run_dir / "eval_noise.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
