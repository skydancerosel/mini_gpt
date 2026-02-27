#!/usr/bin/env python3
"""Quick single-run backbone analysis for Run A (AdamW) while other runs train."""

import json
import sys
from pathlib import Path
import numpy as np
import torch

# Import analysis functions from sgd_control_analysis
from sgd_control_analysis import (
    load_trajectory, load_metrics, estimate_backbone,
    estimate_backbone_block, pc1_rotation, update_alignment,
    residual_coupling, WINDOW_STEPS, EVAL_ALIGN_STEPS,
    WINDOW_START, WINDOW_END, ANCHOR_STEP,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    run_dir = Path("runs/sgd_control/adamw_s42")
    out_dir = Path("runs/sgd_control/analysis")
    out_dir.mkdir(exist_ok=True)

    if not (run_dir / "pilot_metrics.json").exists():
        print("ERROR: Run A not complete yet")
        return

    metrics = load_metrics(run_dir)
    print(f"Run A: {len(metrics)} eval points, "
          f"final step={metrics[-1]['step']}")

    # Load window trajectory
    print(f"\nLoading window trajectory [{WINDOW_START}, {WINDOW_END}]...")
    trunk_win, blocks_win, loaded_steps = load_trajectory(
        run_dir, WINDOW_STEPS, n_blocks=8)
    print(f"  Loaded {len(loaded_steps)} checkpoints, "
          f"trunk shape: {trunk_win.shape}")

    # 1) Backbone estimation
    print("\nEstimating backbone (drift-matrix PCA)...")
    backbone = estimate_backbone(trunk_win, loaded_steps)
    v_b = backbone["v_b"]
    Vt = backbone["Vt"]
    var_frac = backbone["var_frac"]

    print(f"  PC1 = {backbone['pc1_frac']*100:.1f}%")
    print(f"  k95 = {backbone['k95']}, k99 = {backbone['k99']}")
    print(f"  Top-10 var fractions: "
          f"{[f'{v:.3f}' for v in var_frac[:10].tolist()]}")

    # Per-block PCA
    print("\nPer-block PCA:")
    for b in range(8):
        bpca = estimate_backbone_block(blocks_win[b], loaded_steps)
        print(f"  Block {b}: PC1 = {bpca['pc1_frac']*100:.1f}%, "
              f"k95={bpca['k95']}, k99={bpca['k99']}")

    # 2) PC1 rotation
    print("\nPC1 rotation (window_size=5):")
    rot = pc1_rotation(trunk_win, loaded_steps, window_size=5)
    print(f"  Mean |cos| = {rot['mean_cos']:.4f}")
    print(f"  Min  |cos| = {rot['min_cos']:.4f}")

    # 3) Update alignment
    print(f"\nUpdate alignment (v_b held fixed, eval steps: {EVAL_ALIGN_STEPS}):")
    align = update_alignment(run_dir, v_b, EVAL_ALIGN_STEPS)
    for r in align:
        print(f"  step {r['step']}: |cos| = {r['abs_cos']:.4f}, "
              f"signed = {r['signed_cos']:+.4f}")

    # 4) Residual coupling
    print("\nResidual/switch coupling:")
    res = residual_coupling(trunk_win, Vt, loaded_steps, metrics)
    print(f"  Mean PC2-6 capture = {res['mean_pc26_capture']:.4f}")
    print(f"  corr(||r||, p_ood) = {res['corr_rnorm_pood']:.4f}")

    # Save v_b for downstream analyses (Steps 9, 10)
    torch.save({
        "v_b": v_b,
        "Vt": Vt[:10],
        "var_frac": var_frac[:10],
        "pc1_frac": backbone["pc1_frac"],
        "k95": backbone["k95"],
        "k99": backbone["k99"],
    }, out_dir / "backbone_adamw.pt")
    print(f"\nSaved backbone to {out_dir / 'backbone_adamw.pt'}")

    # ── Plots (Run A only) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scree plot
    ax = axes[0]
    n_show = min(20, len(var_frac))
    ax.bar(range(1, n_show + 1),
           [v * 100 for v in var_frac[:n_show].tolist()],
           color="#1f77b4")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance fraction (%)")
    ax.set_title(f"AdamW drift-matrix PCA (PC1={backbone['pc1_frac']*100:.1f}%)")
    ax.grid(True, alpha=0.3, axis="y")

    # Training curves
    ax = axes[1]
    steps = [m["step"] for m in metrics]
    ax.plot(steps, [m["probe_ood_acc"] for m in metrics],
            "b-o", markersize=3, label="p_ood")
    ax.plot(steps, [m["probe_in_acc"] for m in metrics],
            "g-o", markersize=3, label="p_in")
    ax.axvspan(WINDOW_START, WINDOW_END, alpha=0.07, color="gray")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("AdamW: Probe dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Update alignment timeline
    ax = axes[2]
    if align:
        a_steps = [r["step"] for r in align]
        a_signed = [r["signed_cos"] for r in align]
        a_abs = [r["abs_cos"] for r in align]
        ax.plot(a_steps, a_abs, "b-o", markersize=5, label="|cos|")
        ax.plot(a_steps, a_signed, "r-o", markersize=5, label="signed cos")
        ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.axvspan(WINDOW_START, WINDOW_END, alpha=0.07, color="gray")
    ax.set_xlabel("Step t")
    ax.set_ylabel("cos(u_t, v_b)")
    ax.set_title("AdamW: Update-backbone alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "run_a_backbone_analysis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
