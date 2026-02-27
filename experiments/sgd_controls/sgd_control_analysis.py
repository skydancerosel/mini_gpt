#!/usr/bin/env python3
"""
Early-Window Backbone Estimation for SGD Control Experiment.

Protocol:
  1. Checkpoint sampling:
     - Every 50 steps in [600, 2000] (backbone estimation window)
     - Every 100 steps in (2000, 4000] (out-of-window evaluation)

  2. Backbone estimation (per optimizer):
     - Build drift matrix X from window [600, 2000]:
       X[i] = θ(t_i) − θ(600), row-normalized
     - Uncentered PCA → backbone direction v_b = first right singular vector
     - Fixed early window avoids "direction to final weights" artifact

  3. Out-of-window evaluation:
     - PC1 variance fraction & k95/k99 within window
     - PC1 rotation stability within window
     - Update alignment C(t) = cos(θ(t+200)−θ(t), v_b)
       at t ∈ {1000, 1600, 2200, 3000, 4000} (held-out from estimation)
     - Residual/switch coupling (PC2-6 capture, corr(‖r‖, p_ood))

Decision criteria (pre-registered):
  Optimizer-induced → AdamW PC1 70-80%, SGD no-mom <50-60%, SGD+mom intermediate
  Intrinsic         → All three show strong PC1 and stability

Usage:
  python sgd_control_analysis.py
  python sgd_control_analysis.py --all-steps
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# Trunk parameter extraction (mirrors attractor_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight"
    r"|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)

BLOCK_KEYS = [
    "attn.qkv.weight",
    "attn.out_proj.weight",
    "mlp.w_up.weight",
    "mlp.w_down.weight",
]


def flatten_trunk(state_dict):
    """Flatten all trunk params into 1-D float32 tensor."""
    parts = []
    for key in sorted(state_dict.keys()):
        if TRUNK_PATTERN.match(key):
            parts.append(state_dict[key].cpu().reshape(-1).float())
    return torch.cat(parts)


def flatten_block(state_dict, block_idx):
    """Flatten params for a single block into 1-D float32 tensor."""
    parts = []
    for suffix in BLOCK_KEYS:
        key = f"blocks.{block_idx}.{suffix}"
        if key in state_dict:
            parts.append(state_dict[key].cpu().reshape(-1).float())
    return torch.cat(parts)


def load_checkpoint(run_dir, step):
    p = Path(run_dir) / f"ckpt_{step:06d}.pt"
    return torch.load(p, map_location="cpu", weights_only=True)


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint step schedule
# ═══════════════════════════════════════════════════════════════════════════

# Backbone estimation window
WINDOW_START = 600
WINDOW_END = 2000
ANCHOR_STEP = 600  # θ(600) is the reference point for drift matrix

# Window steps: every 50 in [600, 2000]
WINDOW_STEPS = list(range(WINDOW_START, WINDOW_END + 1, 50))

# Out-of-window evaluation steps for update alignment.
# Each C(t) needs θ(t) and θ(t+200), so last feasible t = 3800 (→4000).
EVAL_ALIGN_STEPS = [1000, 1600, 2200, 3000, 3800]

# All checkpoint steps needed
ALL_CKPT_STEPS = sorted(set(
    WINDOW_STEPS
    + list(range(2100, 4001, 100))  # every 100 in (2000, 4000]
))


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory loading
# ═══════════════════════════════════════════════════════════════════════════

def load_trajectory(run_dir, steps, n_blocks=8):
    """Load checkpoint trajectory at specified steps.

    Returns:
        trunk_matrix:   [T, D_trunk]  raw params
        block_matrices: dict {block_idx: [T, D_block]}
    """
    run_dir = Path(run_dir)
    trunk_rows = []
    block_rows = {b: [] for b in range(n_blocks)}
    loaded_steps = []

    for step in steps:
        ckpt_path = run_dir / f"ckpt_{step:06d}.pt"
        if not ckpt_path.exists():
            print(f"    WARNING: {ckpt_path} not found, skipping")
            continue
        ckpt = load_checkpoint(run_dir, step)
        sd = ckpt["model_state_dict"]
        trunk_rows.append(flatten_trunk(sd))
        for b in range(n_blocks):
            block_rows[b].append(flatten_block(sd, b))
        loaded_steps.append(step)
        del ckpt, sd

    trunk_matrix = torch.stack(trunk_rows)
    block_matrices = {b: torch.stack(block_rows[b]) for b in range(n_blocks)}
    return trunk_matrix, block_matrices, loaded_steps


def load_metrics(run_dir):
    with open(Path(run_dir) / "pilot_metrics.json") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# 1) Backbone estimation: drift-matrix PCA on early window
# ═══════════════════════════════════════════════════════════════════════════

def estimate_backbone(trunk_matrix, window_steps, anchor_idx=0):
    """Estimate backbone from drift matrix in early window.

    Drift matrix: X[i] = θ(t_i) − θ(anchor), then row-normalize.
    Uncentered PCA → v_b = first right singular vector.

    Args:
        trunk_matrix: [T, D] raw trunk params for window steps
        window_steps: list of steps (for labeling)
        anchor_idx:   index of anchor step in trunk_matrix (default: 0 = first)

    Returns dict with:
        v_b:       [D] backbone direction (unit vector)
        Vt:        [K, D] all PCs
        sv:        singular values
        var_frac:  variance fractions
        pc1_frac:  PC1 variance fraction
        k95, k99:  effective dimensionality
    """
    theta_anchor = trunk_matrix[anchor_idx]  # [D]

    # Build drift matrix (exclude anchor itself)
    drift_rows = []
    drift_steps = []
    for i, step in enumerate(window_steps):
        if i == anchor_idx:
            continue
        drift_rows.append(trunk_matrix[i] - theta_anchor)
        drift_steps.append(step)

    X = torch.stack(drift_rows)  # [T-1, D]

    # Row-normalize
    norms = X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X_normed = X / norms

    # Uncentered PCA (no mean subtraction)
    U, S, Vt = torch.linalg.svd(X_normed, full_matrices=False)

    var = S ** 2
    total_var = var.sum()
    var_frac = var / total_var

    cumvar = var_frac.cumsum(0)
    k95 = int((cumvar >= 0.95).nonzero()[0][0].item()) + 1
    k99 = int((cumvar >= 0.99).nonzero()[0][0].item()) + 1

    return {
        "v_b": Vt[0],           # backbone direction
        "Vt": Vt,               # all PCs
        "sv": S,
        "var_frac": var_frac,
        "pc1_frac": float(var_frac[0]),
        "k95": k95,
        "k99": k99,
        "drift_steps": drift_steps,
    }


def estimate_backbone_block(block_matrix, window_steps, anchor_idx=0):
    """Same as estimate_backbone but for a single block."""
    theta_anchor = block_matrix[anchor_idx]
    drift_rows = []
    for i in range(len(window_steps)):
        if i == anchor_idx:
            continue
        drift_rows.append(block_matrix[i] - theta_anchor)
    X = torch.stack(drift_rows)
    norms = X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X_normed = X / norms
    _, S, Vt = torch.linalg.svd(X_normed, full_matrices=False)
    var = S ** 2
    var_frac = var / var.sum()
    cumvar = var_frac.cumsum(0)
    k95 = int((cumvar >= 0.95).nonzero()[0][0].item()) + 1
    k99 = int((cumvar >= 0.99).nonzero()[0][0].item()) + 1
    return {"pc1_frac": float(var_frac[0]), "k95": k95, "k99": k99}


# ═══════════════════════════════════════════════════════════════════════════
# 2) PC1 rotation (within estimation window)
# ═══════════════════════════════════════════════════════════════════════════

def pc1_rotation(trunk_matrix, window_steps, anchor_idx=0, window_size=5):
    """Measure PC1 stability across sliding windows within estimation window.

    Uses the same drift-matrix construction (X[i] = θ(t_i) − θ(anchor))
    in each sub-window.
    """
    theta_anchor = trunk_matrix[anchor_idx]

    # Build drift vectors (exclude anchor)
    drifts = []
    for i in range(len(window_steps)):
        if i == anchor_idx:
            continue
        drifts.append(trunk_matrix[i] - theta_anchor)

    if len(drifts) < window_size + 1:
        return {"cos_values": [], "mean_cos": float("nan"),
                "min_cos": float("nan")}

    # Row-normalize all drifts
    D = torch.stack(drifts)
    norms = D.norm(dim=1, keepdim=True).clamp(min=1e-12)
    D_normed = D / norms

    T = D_normed.shape[0]
    pc1s = []
    for i in range(T - window_size + 1):
        window = D_normed[i:i + window_size]
        _, _, Vt = torch.linalg.svd(window, full_matrices=False)
        pc1s.append(Vt[0])

    cos_values = []
    for i in range(len(pc1s) - 1):
        cos_val = abs(float(pc1s[i] @ pc1s[i + 1]))
        cos_values.append(cos_val)

    return {
        "cos_values": cos_values,
        "mean_cos": float(np.mean(cos_values)) if cos_values else float("nan"),
        "min_cos": float(np.min(cos_values)) if cos_values else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3) Update-direction alignment (out-of-window evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def update_alignment(run_dir, v_b, eval_steps, delta=200):
    """Compute C(t) = cos(θ(t+δ) − θ(t), v_b) at specified eval steps.

    v_b is held fixed from the early-window estimation.

    Args:
        run_dir: path to run directory
        v_b: [D] backbone direction (unit vector)
        eval_steps: list of steps t at which to evaluate
        delta: step offset for update direction (default: 200)

    Returns:
        results: list of {step, signed_cos, abs_cos}
    """
    v_b_unit = v_b / (v_b.norm() + 1e-12)
    results = []

    for t in eval_steps:
        t2 = t + delta
        p1 = Path(run_dir) / f"ckpt_{t:06d}.pt"
        p2 = Path(run_dir) / f"ckpt_{t2:06d}.pt"

        if not p1.exists() or not p2.exists():
            print(f"    WARNING: Missing checkpoint for alignment at "
                  f"t={t} or t+{delta}={t2}, skipping")
            continue

        theta_t = flatten_trunk(
            torch.load(p1, map_location="cpu", weights_only=True)
            ["model_state_dict"])
        theta_t2 = flatten_trunk(
            torch.load(p2, map_location="cpu", weights_only=True)
            ["model_state_dict"])

        u = theta_t2 - theta_t
        u_norm = u.norm()
        if u_norm < 1e-12:
            c = 0.0
        else:
            c = float((u @ v_b_unit) / u_norm)

        results.append({
            "step": t,
            "signed_cos": c,
            "abs_cos": abs(c),
        })
        del theta_t, theta_t2

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4) Residual/switch coupling
# ═══════════════════════════════════════════════════════════════════════════

def residual_coupling(trunk_matrix, Vt, steps, metrics):
    """Residual analysis: PC2-6 capture and correlation with p_ood.

    Uses the window drift matrix (row-normalized) for projection.
    """
    # Row-normalize raw params (not drift — full snapshots for residual)
    norms = trunk_matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X_normed = trunk_matrix / norms

    pc1 = Vt[0]  # backbone direction

    # PC1 projection
    c1 = (X_normed @ pc1).unsqueeze(1) * pc1.unsqueeze(0)  # [T, D]
    residuals = X_normed - c1
    r_norms = residuals.norm(dim=1)

    # PC2-6 capture
    n_pcs = min(Vt.shape[0], 6)
    if n_pcs >= 2:
        pc26 = Vt[1:n_pcs]
        proj_coeff = residuals @ pc26.T
        capture = (proj_coeff ** 2).sum(dim=1) / (r_norms ** 2 + 1e-12)
    else:
        capture = torch.zeros(X_normed.shape[0])

    # Match steps to metrics for p_ood
    step_to_pood = {}
    for m in metrics:
        step_to_pood[m["step"]] = m["probe_ood_acc"]

    p_ood = [step_to_pood.get(s, float("nan")) for s in steps]
    valid = [(r, p) for r, p in zip(r_norms.tolist(), p_ood)
             if not np.isnan(p)]

    if len(valid) >= 3:
        r_arr, p_arr = zip(*valid)
        corr = float(np.corrcoef(r_arr, p_arr)[0, 1])
    else:
        corr = float("nan")

    return {
        "residual_norms": r_norms.tolist(),
        "pc26_capture": capture.tolist(),
        "mean_pc26_capture": float(capture.mean()),
        "corr_rnorm_pood": corr,
        "steps": steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

RUN_LABELS = {"adamw": "AdamW (A)", "sgd_nomom": "SGD no-mom (B)",
              "sgd_mom": "SGD+mom (C)"}
RUN_COLORS = {"adamw": "#1f77b4", "sgd_nomom": "#d62728",
              "sgd_mom": "#2ca02c"}


def plot_training_curves(all_metrics, out_dir):
    """Figure 1: training curves for all runs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for name, metrics in all_metrics.items():
        steps = [m["step"] for m in metrics]
        c = RUN_COLORS[name]
        lbl = RUN_LABELS[name]
        axes[0].plot(steps, [m["train_loss"] for m in metrics],
                     color=c, linewidth=1.2, label=lbl)
        axes[1].plot(steps, [m["probe_ood_acc"] for m in metrics],
                     color=c, linewidth=1.2, label=lbl)
        axes[2].plot(steps, [m["val_loss"] for m in metrics],
                     color=c, linewidth=1.2, label=lbl)

    axes[0].set_ylabel("Train loss")
    axes[1].set_ylabel("p_ood (exact-match)")
    axes[2].set_ylabel("Val loss")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    # Shade estimation window
    for ax in axes:
        ax.axvspan(WINDOW_START, WINDOW_END, alpha=0.07, color="gray")

    fig.suptitle("SGD Control: Training Curves", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "fig1_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_pca_summary(all_backbone, all_block_pca, out_dir, n_blocks=8):
    """Figure 2: PC1 fraction and effective dimensionality."""
    names = list(all_backbone.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Trunk PC1 fraction (from drift-matrix PCA)
    ax = axes[0]
    x = np.arange(len(names))
    vals = [all_backbone[n]["pc1_frac"] * 100 for n in names]
    colors = [RUN_COLORS[n] for n in names]
    bars = ax.bar(x, vals, color=colors, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel("PC1 variance fraction (%)")
    ax.set_title(f"Trunk PC1 (drift matrix, window [{WINDOW_START},{WINDOW_END}])")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontsize=9)

    # Panel B: k95, k99
    ax = axes[1]
    width = 0.35
    k95 = [all_backbone[n]["k95"] for n in names]
    k99 = [all_backbone[n]["k99"] for n in names]
    ax.bar(x - width / 2, k95, width, color=colors, alpha=0.7, label="k95")
    ax.bar(x + width / 2, k99, width, color=colors, alpha=1.0, label="k99")
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel("Number of PCs")
    ax.set_title("Effective dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Per-block PC1 fraction
    ax = axes[2]
    x_blocks = np.arange(n_blocks)
    bw = 0.25
    for i, name in enumerate(names):
        vals = [all_block_pca[name][b]["pc1_frac"] * 100
                for b in range(n_blocks)]
        ax.bar(x_blocks + i * bw, vals, bw,
               color=RUN_COLORS[name], label=RUN_LABELS[name], alpha=0.8)
    ax.set_xticks(x_blocks + bw)
    ax.set_xticklabels([f"L{b}" for b in range(n_blocks)], fontsize=9)
    ax.set_ylabel("PC1 variance fraction (%)")
    ax.set_title("Per-block PC1 (drift matrix)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SGD Control: Uncentered Drift-Matrix PCA", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "fig2_pca_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_pc1_rotation(all_rotation, out_dir):
    """Figure 3: PC1 rotation stability within estimation window."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rot in all_rotation.items():
        cos_vals = rot["cos_values"]
        if not cos_vals:
            continue
        x = list(range(len(cos_vals)))
        ax.plot(x, cos_vals, "o-", color=RUN_COLORS[name],
                linewidth=1.2, markersize=4, label=RUN_LABELS[name])

    ax.set_xlabel("Window index (within estimation window)")
    ax.set_ylabel("|cos(PC1$_t$, PC1$_{t+1}$)|")
    ax.set_title(f"PC1 Rotation Stability (window [{WINDOW_START},{WINDOW_END}])")
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(out_dir) / "fig3_pc1_rotation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_update_alignment(all_alignment, out_dir):
    """Figure 4: out-of-window update alignment C(t) = cos(Δθ, v_b)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name, results in all_alignment.items():
        if not results:
            continue
        steps = [r["step"] for r in results]
        absc = [r["abs_cos"] for r in results]
        signed = [r["signed_cos"] for r in results]
        c = RUN_COLORS[name]
        lbl = RUN_LABELS[name]
        axes[0].plot(steps, absc, "o-", color=c, linewidth=1.5,
                     markersize=5, label=lbl)
        axes[1].plot(steps, signed, "o-", color=c, linewidth=1.5,
                     markersize=5, label=lbl)

    # Shade estimation window
    for ax in axes:
        ax.axvspan(WINDOW_START, WINDOW_END, alpha=0.07, color="gray",
                   label="estimation window" if ax == axes[0] else None)

    axes[0].set_ylabel("|cos($\\hat{u}$, $v_b$)|")
    axes[0].set_title("Update alignment magnitude (v_b from early window)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("signed cos($\\hat{u}$, $v_b$)")
    axes[1].set_xlabel("Step t (update = θ(t+200) − θ(t))")
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_title("Signed update alignment")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("SGD Control: Update-Direction Alignment", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "fig4_update_alignment.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_residual_coupling(all_residual, all_metrics, out_dir):
    """Figure 5: Residual/switch coupling."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: PC2-6 capture over steps
    ax = axes[0]
    for name, res in all_residual.items():
        cap = res["pc26_capture"]
        steps = res["steps"]
        ax.plot(steps[:len(cap)], cap, "o-", color=RUN_COLORS[name],
                linewidth=1.2, markersize=4, label=RUN_LABELS[name])
    ax.axvspan(WINDOW_START, WINDOW_END, alpha=0.07, color="gray")
    ax.set_xlabel("Step")
    ax.set_ylabel("PC2-6 capture fraction")
    ax.set_title("Residualized switch capture")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: ||r(t)|| vs p_ood scatter
    ax = axes[1]
    for name, res in all_residual.items():
        r_norms = res["residual_norms"]
        steps = res["steps"]
        step_to_pood = {}
        for m in all_metrics[name]:
            step_to_pood[m["step"]] = m["probe_ood_acc"]
        p_ood = [step_to_pood.get(s, float("nan")) for s in steps]
        valid_r = [r for r, p in zip(r_norms, p_ood) if not np.isnan(p)]
        valid_p = [p for p in p_ood if not np.isnan(p)]
        ax.scatter(valid_r, valid_p, color=RUN_COLORS[name],
                   alpha=0.6, s=30, label=RUN_LABELS[name])
    ax.set_xlabel("||r(t)|| (residual norm)")
    ax.set_ylabel("p_ood")
    ax.set_title("Residual norm vs probe accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("SGD Control: Residual/Switch Coupling", fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "fig5_residual_coupling.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Decision table
# ═══════════════════════════════════════════════════════════════════════════

def print_decision_table(all_backbone, all_rotation, all_alignment,
                         all_residual):
    """Print summary with pre-registered criteria."""
    print("\n" + "=" * 78)
    print("  SGD CONTROL: DECISION TABLE")
    print("  (Backbone estimated from drift matrix, "
          f"window [{WINDOW_START},{WINDOW_END}])")
    print("=" * 78)

    header = (f"  {'Metric':<30s}  {'AdamW':>10s}  "
              f"{'SGD no-mom':>10s}  {'SGD+mom':>10s}")
    print(header)
    print(f"  {'-'*74}")

    names = ["adamw", "sgd_nomom", "sgd_mom"]

    def row(label, values, fmt=".1f"):
        vals = "  ".join(f"{v:>10{fmt}}" for v in values)
        print(f"  {label:<30s}  {vals}")

    # PCA (drift-matrix)
    row("PC1 var fraction (%)",
        [all_backbone[n]["pc1_frac"] * 100 for n in names])
    row("k95",
        [all_backbone[n]["k95"] for n in names], fmt="d")
    row("k99",
        [all_backbone[n]["k99"] for n in names], fmt="d")

    # PC1 rotation (within window)
    row("PC1 rotation mean |cos|",
        [all_rotation[n]["mean_cos"] for n in names], fmt=".4f")
    row("PC1 rotation min |cos|",
        [all_rotation[n]["min_cos"] for n in names], fmt=".4f")

    # Update alignment (out-of-window, held fixed v_b)
    for n in names:
        results = all_alignment[n]
        if results:
            mean_abs = float(np.mean([r["abs_cos"] for r in results]))
            mean_sgn = float(np.mean([r["signed_cos"] for r in results]))
        else:
            mean_abs = mean_sgn = float("nan")
        all_alignment[n + "_mean_abs"] = mean_abs
        all_alignment[n + "_mean_sgn"] = mean_sgn

    row("Update align mean |cos|",
        [all_alignment[n + "_mean_abs"] for n in names], fmt=".4f")
    row("Update align mean signed",
        [all_alignment[n + "_mean_sgn"] for n in names], fmt=".4f")

    # Per-step alignment
    print(f"\n  Update alignment per step (v_b held fixed):")
    print(f"  {'Step':<8s}", end="")
    for n in names:
        print(f"  {RUN_LABELS[n]:>14s}", end="")
    print()
    all_steps = set()
    for n in names:
        for r in all_alignment[n]:
            all_steps.add(r["step"])
    for step in sorted(all_steps):
        print(f"  {step:<8d}", end="")
        for n in names:
            vals = [r for r in all_alignment[n] if r["step"] == step]
            if vals:
                v = vals[0]["signed_cos"]
                print(f"  {v:>+14.4f}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()

    # Residual coupling
    print()
    row("PC2-6 mean capture",
        [all_residual[n]["mean_pc26_capture"] for n in names], fmt=".4f")
    row("corr(||r||, p_ood)",
        [all_residual[n]["corr_rnorm_pood"] for n in names], fmt=".4f")

    print(f"\n  {'-'*74}")

    # Verdict
    adamw_pc1 = all_backbone["adamw"]["pc1_frac"] * 100
    sgd_pc1 = all_backbone["sgd_nomom"]["pc1_frac"] * 100
    mom_pc1 = all_backbone["sgd_mom"]["pc1_frac"] * 100

    print("\n  Pre-registered criteria:")
    print(f"    Optimizer-induced: AdamW PC1 ~70-80%, "
          f"SGD no-mom <50-60%, SGD+mom intermediate")
    print(f"    Intrinsic:        All three show strong PC1 and stability")

    if sgd_pc1 < adamw_pc1 * 0.75:
        verdict = "OPTIMIZER-INDUCED"
        detail = (f"SGD no-mom PC1 ({sgd_pc1:.1f}%) dropped >{25:.0f}% "
                  f"vs AdamW ({adamw_pc1:.1f}%)")
    elif abs(sgd_pc1 - adamw_pc1) < adamw_pc1 * 0.15:
        verdict = "INTRINSIC"
        detail = (f"SGD no-mom PC1 ({sgd_pc1:.1f}%) within 15% "
                  f"of AdamW ({adamw_pc1:.1f}%)")
    else:
        verdict = "AMBIGUOUS"
        detail = (f"SGD no-mom PC1 ({sgd_pc1:.1f}%) vs "
                  f"AdamW ({adamw_pc1:.1f}%) — moderate difference")

    bridge_ok = (mom_pc1 > sgd_pc1) if verdict == "OPTIMIZER-INDUCED" else True

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"      {detail}")
    if verdict == "OPTIMIZER-INDUCED":
        bridge_str = ("yes" if bridge_ok
                      else "no (SGD+mom did not recover)")
        print(f"      Bridge (SGD+mom recovery): {bridge_str} "
              f"(PC1={mom_pc1:.1f}%)")
    print("=" * 78)

    return verdict


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Early-window backbone analysis for SGD control")
    parser.add_argument("--base-dir", type=str,
                        default="runs/sgd_control")
    parser.add_argument("--window", type=int, default=5,
                        help="Sub-window size for PC1 rotation")
    parser.add_argument("--n-blocks", type=int, default=8)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = base_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    run_dirs = {
        "adamw": base_dir / "adamw_s42",
        "sgd_nomom": base_dir / "sgd_nomom_s42",
        "sgd_mom": base_dir / "sgd_mom_s42",
    }

    # Verify runs exist
    for name, rd in run_dirs.items():
        if not (rd / "pilot_metrics.json").exists():
            print(f"ERROR: {rd} not found. Run sgd_control.py first.")
            return

    print(f"Backbone estimation window: [{WINDOW_START}, {WINDOW_END}]")
    print(f"Anchor step: {ANCHOR_STEP}")
    print(f"Window steps ({len(WINDOW_STEPS)}): "
          f"{WINDOW_STEPS[0]}..{WINDOW_STEPS[-1]} (every 50)")
    print(f"Alignment eval steps: {EVAL_ALIGN_STEPS}")
    print(f"PC1 rotation sub-window: {args.window}")

    # ── Analyze each run ──────────────────────────────────────────────
    all_backbone = {}
    all_block_pca = {}
    all_rotation = {}
    all_alignment = {}
    all_residual = {}
    all_metrics = {}

    for name, rd in run_dirs.items():
        print(f"\n{'─'*60}")
        print(f"  Analyzing: {RUN_LABELS[name]} ({rd})")
        print(f"{'─'*60}")

        metrics = load_metrics(rd)
        all_metrics[name] = metrics

        # Load window trajectory
        print("  Loading window trajectory...")
        trunk_win, blocks_win, loaded_steps = load_trajectory(
            rd, WINDOW_STEPS, n_blocks=args.n_blocks)
        print(f"    Loaded {len(loaded_steps)} checkpoints, "
              f"trunk shape: {trunk_win.shape}")

        # 1) Backbone estimation (drift-matrix PCA)
        print("  Estimating backbone (drift-matrix PCA)...")
        backbone = estimate_backbone(trunk_win, loaded_steps)
        all_backbone[name] = {
            "pc1_frac": backbone["pc1_frac"],
            "k95": backbone["k95"],
            "k99": backbone["k99"],
            "var_frac": backbone["var_frac"][:10].tolist(),
        }
        print(f"    PC1 = {backbone['pc1_frac']*100:.1f}%,  "
              f"k95 = {backbone['k95']},  k99 = {backbone['k99']}")

        # Per-block PCA
        print("  Per-block PCA...")
        all_block_pca[name] = {}
        for b in range(args.n_blocks):
            bpca = estimate_backbone_block(blocks_win[b], loaded_steps)
            all_block_pca[name][b] = bpca
            print(f"    Block {b}: PC1 = {bpca['pc1_frac']*100:.1f}%")

        # 2) PC1 rotation (within window)
        print("  Computing PC1 rotation...")
        rot = pc1_rotation(trunk_win, loaded_steps,
                           window_size=args.window)
        all_rotation[name] = rot
        print(f"    Mean |cos| = {rot['mean_cos']:.4f},  "
              f"min |cos| = {rot['min_cos']:.4f}")

        # 3) Update alignment (out-of-window, v_b held fixed)
        print("  Computing update alignment (v_b held fixed)...")
        align = update_alignment(rd, backbone["v_b"], EVAL_ALIGN_STEPS)
        all_alignment[name] = align
        for r in align:
            print(f"    step {r['step']}: "
                  f"|cos| = {r['abs_cos']:.4f},  "
                  f"signed = {r['signed_cos']:+.4f}")

        # 4) Residual/switch coupling (on window trajectory)
        print("  Computing residual coupling...")
        res = residual_coupling(trunk_win, backbone["Vt"],
                                loaded_steps, metrics)
        all_residual[name] = res
        print(f"    Mean PC2-6 capture = {res['mean_pc26_capture']:.4f},  "
              f"corr(||r||, p_ood) = {res['corr_rnorm_pood']:.4f}")

        del trunk_win, blocks_win, backbone

    # ── Generate figures ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Generating figures...")
    print(f"{'─'*60}")

    plot_training_curves(all_metrics, out_dir)
    plot_pca_summary(all_backbone, all_block_pca, out_dir,
                     n_blocks=args.n_blocks)
    plot_pc1_rotation(all_rotation, out_dir)
    plot_update_alignment(all_alignment, out_dir)
    plot_residual_coupling(all_residual, all_metrics, out_dir)

    # ── Decision table ────────────────────────────────────────────────
    verdict = print_decision_table(
        all_backbone, all_rotation, all_alignment, all_residual)

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "protocol": {
            "window": [WINDOW_START, WINDOW_END],
            "anchor_step": ANCHOR_STEP,
            "window_steps": WINDOW_STEPS,
            "eval_align_steps": EVAL_ALIGN_STEPS,
            "pc1_rotation_subwindow": args.window,
        },
        "backbone_pca": all_backbone,
        "block_pca": {name: {str(b): v for b, v in bdict.items()}
                      for name, bdict in all_block_pca.items()},
        "pc1_rotation": {name: {"mean_cos": r["mean_cos"],
                                "min_cos": r["min_cos"],
                                "cos_values": r["cos_values"]}
                         for name, r in all_rotation.items()},
        "update_alignment": {
            name: results_list
            for name, results_list in all_alignment.items()
            if isinstance(results_list, list)
        },
        "residual_coupling": {
            name: {k: v for k, v in res.items() if k != "steps"}
            for name, res in all_residual.items()
        },
        "verdict": verdict,
    }
    results_path = out_dir / "sgd_control_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {results_path}")
    print(f"  All figures in {out_dir}/")


if __name__ == "__main__":
    main()
