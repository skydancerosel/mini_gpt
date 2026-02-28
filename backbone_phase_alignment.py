#!/usr/bin/env python3
"""
Quick check: compute phase-specific backbone directions and measure alignment.

  v_b^(0-4k)  from checkpoints ≤ 4000
  v_b^(4k-10k) from checkpoints ≥ 4000
  |⟨v_b^(0-4k), v_b^(4k-10k)⟩|
"""

import re
from pathlib import Path

import numpy as np
import torch

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)


def flatten_trunk(state_dict):
    parts = []
    for key in sorted(state_dict.keys()):
        if TRUNK_PATTERN.match(key):
            parts.append(state_dict[key].cpu().numpy().astype(np.float64).ravel())
    return np.concatenate(parts)


def load_trunk(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vec = flatten_trunk(ckpt["model_state_dict"])
    del ckpt
    return vec


def compute_vb(run_dir, ref_step, step_range):
    """Compute row-normalized PC1 from checkpoints in step_range, relative to ref_step."""
    run_dir = Path(run_dir)
    theta0 = load_trunk(run_dir / f"ckpt_{ref_step:06d}.pt")

    steps = sorted(step_range)
    steps = [s for s in steps if s != ref_step]
    X = []
    for s in steps:
        theta = load_trunk(run_dir / f"ckpt_{s:06d}.pt")
        X.append(theta - theta0)

    X = np.stack(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_rn = X / np.maximum(norms, 1e-12)
    _, S, Vt = np.linalg.svd(X_rn, full_matrices=False)
    total_var = np.sum(S ** 2)
    pc1_pct = S[0] ** 2 / total_var * 100

    v = Vt[0]
    # sign fix: positive projection at last step
    if np.dot(X[-1], v) < 0:
        v = -v
    return v, pc1_pct


def main():
    seeds = {
        42: "runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY",
        271: "runs/pilot_wd0.5_lr0.001_lp2.0_s271",
    }

    # All checkpoint steps
    all_steps = [1] + list(range(200, 10001, 200))
    early_steps = [s for s in all_steps if s <= 4000]
    late_steps = [s for s in all_steps if s >= 4000]

    print("=" * 60)
    print("  Phase-specific backbone alignment")
    print("=" * 60)

    for seed, run_dir in seeds.items():
        if not Path(run_dir).exists():
            print(f"\n  [SKIP] seed {seed}")
            continue

        print(f"\n  Seed {seed}")
        print(f"  Loading early phase (steps ≤ 4000, ref=step 1)...")
        v_early, pc1_early = compute_vb(run_dir, ref_step=1, step_range=early_steps)

        print(f"  Loading late phase (steps ≥ 4000, ref=step 4000)...")
        v_late, pc1_late = compute_vb(run_dir, ref_step=4000, step_range=late_steps)

        cos = abs(float(np.dot(v_early, v_late)))

        print(f"\n  PC1 early (0-4k):  {pc1_early:.1f}%")
        print(f"  PC1 late (4k-10k): {pc1_late:.1f}%")
        print(f"  |⟨v_b^(0-4k), v_b^(4k-10k)⟩| = {cos:.4f}")

        if cos > 0.95:
            print(f"  → Same direction (cos≈1): single backbone throughout")
        elif cos > 0.7:
            print(f"  → Moderately aligned: backbone rotates after 4k")
        else:
            print(f"  → Substantially different directions after 4k")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
