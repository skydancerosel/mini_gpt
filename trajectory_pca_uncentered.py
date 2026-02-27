#!/usr/bin/env python3
"""
Uncentered trajectory PCA — per transformer block.

Unlike the centered consecutive-diff PCA in directional_probing.py, this
script computes PCA on the *cumulative drift from initialization*:

    X[t, :] = θ_block(t) − θ_block(0)

The SVD is **uncentered** — the mean is NOT subtracted — so that PC1
captures the dominant drift direction (θ_init → θ_final).  This preserves
the ability to reconstruct functional checkpoints from a low-rank
approximation of the trajectory.

Uses the same block definition as directional_probing.py:
  blocks.0 … blocks.7   (each = ln1 + attn + ln2 + mlp)

Outputs:
  fig_trajectory_pca_uncentered.png   Explained-variance plot (per block)
  trajectory_pca_uncentered.json      Full numerical results

Usage:
  python trajectory_pca_uncentered.py \\
      --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s271/

  # Custom stride & PC count:
  python trajectory_pca_uncentered.py \\
      --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s271/ \\
      --stride 200 --max-pcs 15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attractor_analysis import load_checkpoint
from directional_probing import flatten_block


# =========================================================================
# Core: uncentered trajectory PCA per block
# =========================================================================

def trajectory_pca_uncentered(run_dir, n_blocks, step_stride=400,
                              max_pcs=10):
    """
    Uncentered PCA on cumulative parameter drift per transformer block.

    For each block b, builds the matrix:
        X[t, :] = θ_b(step_t) − θ_b(step_0)      shape (T-1, D_block)

    SVD is computed **without centering**, so PC1 aligns with the dominant
    training drift direction.

    Returns list of dicts (one per block).
    """
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    # Subsample by stride
    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[0] not in steps:
        steps.insert(0, all_steps[0])
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    steps = sorted(set(steps))

    if len(steps) < 3:
        steps = all_steps

    print(f"  Loading {len(steps)} checkpoints "
          f"(stride={step_stride}, [{steps[0]}..{steps[-1]}])")

    # ── Load initial checkpoint (θ_0) per block ──────────────────────
    ckpt0 = load_checkpoint(run_dir, steps[0])
    sd0 = ckpt0["model_state_dict"]
    init_params = {}
    for b in range(n_blocks):
        init_params[b] = flatten_block(sd0, b)
    del ckpt0, sd0

    # ── Accumulate drifts from init per block ────────────────────────
    block_drifts = {b: [] for b in range(n_blocks)}

    for i, step in enumerate(steps[1:], 1):
        ckpt = load_checkpoint(run_dir, step)
        sd = ckpt["model_state_dict"]

        for b in range(n_blocks):
            cur = flatten_block(sd, b)
            block_drifts[b].append(cur - init_params[b])

        del ckpt, sd
        if i % 5 == 0:
            print(f"    {i}/{len(steps) - 1} checkpoints loaded")

    print(f"    {len(steps) - 1}/{len(steps) - 1} checkpoints loaded")

    # ── PCA per block (UNCENTERED) ───────────────────────────────────
    pca_results = []
    for b in range(n_blocks):
        drifts = block_drifts[b]
        if len(drifts) < 2:
            pca_results.append(None)
            print(f"    Block {b}: skipped (< 2 snapshots)")
            continue

        X = torch.stack(drifts).numpy()  # (T-1, D_block)
        # NOTE: NO centering — this is the key difference from
        # directional_probing.py's centered consecutive-diff PCA.

        n_comp = min(max_pcs, *X.shape)
        _, S, _ = np.linalg.svd(X, full_matrices=False)

        var = S ** 2 / max((X.shape[0] - 1), 1)
        total = var.sum()
        ratio = var / max(total, 1e-12)
        cumul = np.cumsum(ratio)

        k_95 = int((cumul < 0.95).sum()) + 1
        k_99 = int((cumul < 0.99).sum()) + 1

        result = {
            "block": b,
            "n_snapshots": len(drifts),
            "dim": X.shape[1],
            "explained_ratio": ratio[:n_comp].tolist(),
            "cumulative": cumul[:n_comp].tolist(),
            "singular_values": S[:n_comp].tolist(),
            "k_star_95": k_95,
            "k_star_99": k_99,
            "total_variance": float(total),
        }
        pca_results.append(result)

        top1 = ratio[0] * 100
        top3 = cumul[min(2, len(cumul) - 1)] * 100
        print(f"    Block {b}: dim={X.shape[1]:>10,}, "
              f"PC1={top1:.1f}%, top-3={top3:.1f}%, "
              f"k*95={k_95}, k*99={k_99}")

    return pca_results


# =========================================================================
# Rolling PC1 rotation analysis
# =========================================================================

def rolling_pc1_rotation(run_dir, n_blocks, window=10, step_stride=200):
    """
    Compute rolling PC1 direction in sliding windows and measure rotation.

    For each block, over a sliding window of `window` consecutive drift
    vectors (from init), compute the uncentered PC1 (first right singular
    vector).  Then measure cos(angle) between consecutive PC1 vectors.

    Sign ambiguity: SVD can flip the sign of singular vectors, so we use
    |cos| = |v_i · v_{i+1}|.

    Returns:
        dict mapping block index → {
            "steps": list of step values (end of each window),
            "cos_pc1": list of |cos(angle)| between consecutive PC1 vectors,
        }
    """
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    # Use dense stride for rolling windows
    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[0] not in steps:
        steps.insert(0, all_steps[0])
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    steps = sorted(set(steps))

    print(f"  Loading {len(steps)} checkpoints "
          f"(stride={step_stride}, [{steps[0]}..{steps[-1]}])")

    # ── Load initial checkpoint (θ_0) per block ──────────────────────
    ckpt0 = load_checkpoint(run_dir, steps[0])
    sd0 = ckpt0["model_state_dict"]
    init_params = {}
    for b in range(n_blocks):
        init_params[b] = flatten_block(sd0, b)
    del ckpt0, sd0

    # ── Accumulate all drifts from init ──────────────────────────────
    drift_steps = []  # steps corresponding to drifts (excludes step 0)
    block_drifts = {b: [] for b in range(n_blocks)}

    for i, step in enumerate(steps[1:], 1):
        ckpt = load_checkpoint(run_dir, step)
        sd = ckpt["model_state_dict"]
        drift_steps.append(step)

        for b in range(n_blocks):
            cur = flatten_block(sd, b)
            block_drifts[b].append(cur - init_params[b])

        del ckpt, sd
        if i % 10 == 0:
            print(f"    {i}/{len(steps) - 1} checkpoints loaded")

    n_drifts = len(drift_steps)
    print(f"    {n_drifts}/{n_drifts} drifts collected, "
          f"window={window} → {n_drifts - window + 1} windows")

    if n_drifts < window:
        print(f"    ERROR: need at least {window} drifts, got {n_drifts}")
        return {}

    # ── Rolling PC1 per block ────────────────────────────────────────
    results = {}
    for b in range(n_blocks):
        drifts = block_drifts[b]
        n_windows = n_drifts - window + 1

        pc1_vectors = []
        window_steps = []

        for w in range(n_windows):
            X = torch.stack(drifts[w:w + window]).numpy()
            # Uncentered SVD
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            pc1 = Vt[0]  # first right singular vector (D_block,)
            pc1_vectors.append(pc1)
            window_steps.append(drift_steps[w + window - 1])  # end of window

        # Cosine between consecutive PC1 vectors
        cos_values = []
        cos_steps = []
        for i in range(len(pc1_vectors) - 1):
            dot = np.dot(pc1_vectors[i], pc1_vectors[i + 1])
            cos_values.append(float(abs(dot)))  # |cos| for sign ambiguity
            cos_steps.append(window_steps[i + 1])

        results[b] = {
            "steps": cos_steps,
            "cos_pc1": cos_values,
        }

        min_cos = min(cos_values) if cos_values else 1.0
        mean_cos = np.mean(cos_values) if cos_values else 1.0
        print(f"    Block {b}: {len(cos_values)} pairs, "
              f"mean |cos|={mean_cos:.4f}, min |cos|={min_cos:.4f}")

    return results


# =========================================================================
# Backbone vs residual decomposition
# =========================================================================

def _load_drifts_and_context(run_dir, n_blocks, manifest_path, step_stride):
    """Shared checkpoint loading for backbone_residual_decomposition."""
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    # Dense stride
    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[0] not in steps:
        steps.insert(0, all_steps[0])
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    steps = sorted(set(steps))

    print(f"  Loading {len(steps)} checkpoints "
          f"(stride={step_stride}, [{steps[0]}..{steps[-1]}])")

    # ── Load metrics for p_ood ───────────────────────────────────────
    metrics_path = Path(run_dir) / "pilot_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    step_to_pood = {m["step"]: m["probe_ood_acc"] for m in metrics}

    # ── Load manifest for switching directions ───────────────────────
    switch_pair = None
    if manifest_path is None:
        manifest_path = Path(run_dir) / "oscillation_manifest.json"
    if Path(manifest_path).exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if "representative" in manifest:
            pp = manifest["representative"].get("priority_pairs", [])
            if pp:
                switch_pair = (pp[0]["peak"], pp[0]["trough"])
        elif "switch_pairs" in manifest:
            sp = manifest["switch_pairs"]
            if sp:
                switch_pair = (sp[0]["peak"], sp[0]["trough"])
        if switch_pair:
            print(f"  Switching direction: peak={switch_pair[0]} → "
                  f"trough={switch_pair[1]}")

    # ── Load initial checkpoint ──────────────────────────────────────
    ckpt0 = load_checkpoint(run_dir, steps[0])
    sd0 = ckpt0["model_state_dict"]
    init_params = {}
    for b in range(n_blocks):
        init_params[b] = flatten_block(sd0, b)
    del ckpt0, sd0

    # ── Accumulate all drifts ────────────────────────────────────────
    drift_steps = []
    block_drifts = {b: [] for b in range(n_blocks)}

    for i, step in enumerate(steps[1:], 1):
        ckpt = load_checkpoint(run_dir, step)
        sd = ckpt["model_state_dict"]
        drift_steps.append(step)

        for b in range(n_blocks):
            cur = flatten_block(sd, b)
            block_drifts[b].append(cur - init_params[b])

        del ckpt, sd
        if i % 10 == 0:
            print(f"    {i}/{len(steps) - 1} checkpoints loaded")

    print(f"    {len(steps) - 1}/{len(steps) - 1} checkpoints loaded")

    # ── Compute switching directions per block ───────────────────────
    block_switch_dir = {}
    if switch_pair:
        peak_step, trough_step = switch_pair
        sd_peak = load_checkpoint(run_dir, peak_step)["model_state_dict"]
        sd_trough = load_checkpoint(run_dir, trough_step)["model_state_dict"]
        for b in range(n_blocks):
            d = flatten_block(sd_peak, b) - flatten_block(sd_trough, b)
            block_switch_dir[b] = d / d.norm()
        del sd_peak, sd_trough
        print(f"  Switching directions computed (unit-norm per block)")

    p_ood = [step_to_pood.get(s, float("nan")) for s in drift_steps]

    return drift_steps, block_drifts, block_switch_dir, switch_pair, p_ood


def _decompose_blocks(block_drifts, n_blocks, block_switch_dir,
                      row_normalize=False):
    """
    Core decomposition logic for both raw and row-normalized modes.

    If row_normalize=True, each drift vector is normalized to unit norm
    BEFORE computing SVD, so the backbone direction is determined by
    angular distribution rather than magnitude.
    """
    tag = "row-norm" if row_normalize else "raw"
    block_results = {}

    for b in range(n_blocks):
        drifts = block_drifts[b]
        X = torch.stack(drifts).numpy()  # (T, D)

        # Optionally row-normalize for SVD
        if row_normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_svd = X / norms
        else:
            X_svd = X

        # Uncentered SVD → PC1
        _, S, Vt = np.linalg.svd(X_svd, full_matrices=False)
        v_backbone = Vt[0]  # (D,)
        v_backbone_t = torch.from_numpy(v_backbone).float()

        # Explained variance of PC1 in the SVD matrix
        var = S ** 2 / max(X_svd.shape[0] - 1, 1)
        total_var = var.sum()
        pc1_var_ratio = float(var[0] / max(total_var, 1e-12))

        a_t = []
        r_norm = []
        drift_norm = []
        r_proj_switch = []

        for delta in drifts:
            delta_np = delta.numpy()
            a = float(np.dot(delta_np, v_backbone))
            a_t.append(a)
            r = delta - a * v_backbone_t
            r_norm.append(float(r.norm()))
            drift_norm.append(float(delta.norm()))

            if b in block_switch_dir:
                proj = float(torch.dot(r, block_switch_dir[b]))
                r_proj_switch.append(proj)

        pct_backbone = np.mean(
            [a**2 / max(d**2, 1e-30)
             for a, d in zip(a_t, drift_norm)]
        ) * 100

        block_result = {
            "a_t": a_t,
            "r_norm": r_norm,
            "drift_norm": drift_norm,
            "pc1_var_ratio": pc1_var_ratio,
        }
        if r_proj_switch:
            block_result["r_proj_switch"] = r_proj_switch

        block_results[b] = block_result

        print(f"    Block {b} ({tag}): PC1={pc1_var_ratio*100:.1f}%, "
              f"backbone {pct_backbone:.1f}% of ||Δθ||², "
              f"max a_t={max(a_t):.1f}, max ||r_t||={max(r_norm):.1f}")

    return block_results


def backbone_residual_decomposition(run_dir, n_blocks, manifest_path=None,
                                    step_stride=200):
    """
    Decompose the training trajectory into backbone + residual per block.

    For each block b:
      1. Compute v_backbone = PC1 of uncentered (θ_t − θ_0) over ALL ckpts
      2. For each checkpoint t:
           delta_t   = θ_b(t) − θ_b(0)
           a_t       = <delta_t, v_backbone>         (backbone projection)
           r_t       = delta_t − a_t * v_backbone    (residual)
           ||r_t||   = norm of residual
      3. Optionally project r_t onto switching direction(s) from manifest

    Returns (raw_results, rownorm_results) — two dicts with identical
    structure, the second using row-normalized deltas for SVD.
    """
    drift_steps, block_drifts, block_switch_dir, switch_pair, p_ood = \
        _load_drifts_and_context(run_dir, n_blocks, manifest_path, step_stride)

    print("\n  --- Raw (uncentered) ---")
    raw_blocks = _decompose_blocks(
        block_drifts, n_blocks, block_switch_dir, row_normalize=False)

    print("\n  --- Row-normalized (uncentered) ---")
    rn_blocks = _decompose_blocks(
        block_drifts, n_blocks, block_switch_dir, row_normalize=True)

    def _build_result(block_results):
        r = {"steps": drift_steps, "p_ood": p_ood,
             "blocks": block_results}
        if switch_pair:
            r["switch_pair"] = list(switch_pair)
        return r

    return _build_result(raw_blocks), _build_result(rn_blocks)


def plot_backbone_residual(decomp, out_dir, suffix="", title_extra=""):
    """
    Three-row figure:
      Row 1: a_t (backbone) per block  +  p_ood overlay
      Row 2: ||r_t|| (residual) per block  +  p_ood overlay
      Row 3: <r_t, v_switch> per block  +  p_ood overlay  (if available)

    suffix: appended to filename, e.g. "_rownorm"
    title_extra: appended to title, e.g. " (row-normalized)"
    """
    steps = decomp["steps"]
    p_ood = decomp["p_ood"]
    blocks = decomp["blocks"]
    has_switch = any("r_proj_switch" in v for v in blocks.values())

    n_rows = 3 if has_switch else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows),
                             sharex=True)

    n = len(blocks)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    # ── Row 1: a_t (backbone coefficient) ────────────────────────────
    ax = axes[0]
    for b in sorted(blocks.keys()):
        ax.plot(steps, blocks[b]["a_t"], "-", color=colors[b],
                linewidth=1.2, label=f"Block {b}")
    ax.set_ylabel("$a_t$ (backbone projection)", fontsize=11)
    ax.set_title("Backbone vs Residual Decomposition"
                 f"{title_extra}: "
                 r"$\theta_t = \theta_0 + a_t\,v_{backbone} + r_t$",
                 fontsize=13)
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, p_ood, "k--", linewidth=2, alpha=0.6, label="$p_{ood}$")
    ax2.set_ylabel("$p_{ood}$", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    # ── Row 2: ||r_t|| (residual norm) ───────────────────────────────
    ax = axes[1]
    for b in sorted(blocks.keys()):
        ax.plot(steps, blocks[b]["r_norm"], "-", color=colors[b],
                linewidth=1.2, label=f"Block {b}")
    ax.set_ylabel("$\\|r_t\\|$ (residual norm)", fontsize=11)
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, p_ood, "k--", linewidth=2, alpha=0.6, label="$p_{ood}$")
    ax2.set_ylabel("$p_{ood}$", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    # ── Row 3: <r_t, v_switch> (residual projected onto switching) ───
    if has_switch:
        ax = axes[2]
        for b in sorted(blocks.keys()):
            if "r_proj_switch" in blocks[b]:
                ax.plot(steps, blocks[b]["r_proj_switch"], "-",
                        color=colors[b], linewidth=1.2,
                        label=f"Block {b}")
        sp = decomp.get("switch_pair", ["?", "?"])
        ax.set_ylabel(
            f"$\\langle r_t,\\, v_{{switch}}\\rangle$\n"
            f"(peak {sp[0]}→trough {sp[1]})",
            fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", ls="-", alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(steps, p_ood, "k--", linewidth=2, alpha=0.6,
                 label="$p_{ood}$")
        ax2.set_ylabel("$p_{ood}$", fontsize=11)
        ax2.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Training step", fontsize=11)

    fig.tight_layout()
    path = Path(out_dir) / f"fig_backbone_residual{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Plotting
# =========================================================================

def plot_trajectory_pca_uncentered(pca_results, out_dir):
    """Plot explained variance ratios per block (uncentered PCA)."""
    valid = [r for r in pca_results if r is not None]
    if not valid:
        print("  No valid PCA results to plot.")
        return

    n = len(valid)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: individual explained variance ratio
    ax1 = axes[0]
    for r, c in zip(valid, colors):
        k = len(r["explained_ratio"])
        ax1.plot(range(1, k + 1), r["explained_ratio"], "-o", color=c,
                 markersize=4, label=f'Block {r["block"]}')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Per-block Trajectory PCA (uncentered)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: cumulative explained variance
    ax2 = axes[1]
    for r, c in zip(valid, colors):
        k = len(r["cumulative"])
        ax2.plot(range(1, k + 1), r["cumulative"], "-o", color=c,
                 markersize=4, label=f'Block {r["block"]}')
    ax2.axhline(0.95, ls="--", color="gray", alpha=0.5, label="95%")
    ax2.axhline(0.99, ls=":", color="gray", alpha=0.5, label="99%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Trajectory Dimensionality per Block (uncentered)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "fig_trajectory_pca_uncentered.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rolling_pc1_rotation(rotation_results, out_dir):
    """Plot |cos(angle)| between consecutive rolling PC1 vectors."""
    if not rotation_results:
        print("  No rotation results to plot.")
        return

    n = len(rotation_results)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    fig, ax = plt.subplots(figsize=(12, 5))

    for b in sorted(rotation_results.keys()):
        r = rotation_results[b]
        ax.plot(r["steps"], r["cos_pc1"], "-o", color=colors[b],
                markersize=3, linewidth=1.2, label=f"Block {b}")

    ax.set_xlabel("Training step (end of window)")
    ax.set_ylabel("|cos(PC1_t, PC1_{t+1})|")
    ax.set_title("Rolling PC1 Rotation (window=10, uncentered)")
    ax.set_ylim(0.9, 1.005)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "fig_pc1_rotation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Uncentered trajectory PCA per transformer block"
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory with checkpoints")
    parser.add_argument("--n-blocks", type=int, default=8,
                        help="Number of transformer blocks (default: 8)")
    parser.add_argument("--stride", type=int, default=400,
                        help="Checkpoint stride (default: 400)")
    parser.add_argument("--max-pcs", type=int, default=10,
                        help="Max principal components to report (default: 10)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <run-dir>/analysis)")
    parser.add_argument("--window", type=int, default=10,
                        help="Rolling window size for PC1 rotation (default: 10)")
    parser.add_argument("--rolling-stride", type=int, default=200,
                        help="Checkpoint stride for rolling PC1 (default: 200)")
    parser.add_argument("--skip-global", action="store_true",
                        help="Skip the global PCA (only run rolling PC1)")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip rolling PC1 rotation analysis")
    parser.add_argument("--skip-decomp", action="store_true",
                        help="Skip backbone/residual decomposition")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to oscillation manifest (for switching dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir  : {run_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Blocks   : {args.n_blocks}")
    print(f"Stride   : {args.stride}")
    print(f"Max PCs  : {args.max_pcs}")

    output = {}

    # ── Global PCA ────────────────────────────────────────────────────
    if not args.skip_global:
        print("\n" + "=" * 60)
        print("Uncentered trajectory PCA per block")
        print("  X[t] = theta_block(t) - theta_block(0), NO centering")
        print("=" * 60)

        pca_results = trajectory_pca_uncentered(
            run_dir, args.n_blocks,
            step_stride=args.stride,
            max_pcs=args.max_pcs,
        )
        output["pca"] = pca_results
        plot_trajectory_pca_uncentered(pca_results, out_dir)

    # ── Rolling PC1 rotation ──────────────────────────────────────────
    if not args.skip_rolling:
        print("\n" + "=" * 60)
        print(f"Rolling PC1 rotation (window={args.window}, "
              f"stride={args.rolling_stride})")
        print("=" * 60)

        rotation = rolling_pc1_rotation(
            run_dir, args.n_blocks,
            window=args.window,
            step_stride=args.rolling_stride,
        )
        # Convert int keys to str for JSON serialization
        output["rolling_pc1"] = {str(b): v for b, v in rotation.items()}
        plot_rolling_pc1_rotation(rotation, out_dir)

    # ── Backbone / residual decomposition ─────────────────────────────
    if not args.skip_decomp:
        print("\n" + "=" * 60)
        print("Backbone vs Residual Decomposition")
        print("  θ_t = θ_0 + a_t · v_backbone + r_t")
        print("  (raw + row-normalized)")
        print("=" * 60)

        decomp_raw, decomp_rn = backbone_residual_decomposition(
            run_dir, args.n_blocks,
            manifest_path=args.manifest,
            step_stride=args.rolling_stride,
        )

        # Convert block int keys to str for JSON
        def _jsonify(d):
            out = dict(d)
            out["blocks"] = {str(b): v for b, v in d["blocks"].items()}
            return out

        output["backbone_residual"] = _jsonify(decomp_raw)
        output["backbone_residual_rownorm"] = _jsonify(decomp_rn)

        plot_backbone_residual(decomp_raw, out_dir)
        plot_backbone_residual(decomp_rn, out_dir,
                               suffix="_rownorm",
                               title_extra=" (row-normalized)")

    # ── Save JSON ─────────────────────────────────────────────────────
    json_path = out_dir / "trajectory_pca_uncentered.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
