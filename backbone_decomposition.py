#!/usr/bin/env python3
"""
Backbone decomposition: compute a(t), r(t), speeds, and plots for 10k runs.

Uses trunk-only parameters (attn + MLP weights, excluding embeddings/LN)
consistent with the layerwise analysis in beta_summary.py / attractor_analysis.py.

For each seed:
  1. Load θ₀ (step 1) as reference, trunk params only
  2. Compute Δθ_t = θ_t − θ₀ for all checkpoints
  3. Row-normalized uncentered SVD → stable rank-1 backbone direction v_b
  4. a(t) = ⟨Δθ_t, v_b⟩,  r(t) = ‖Δθ_t − a(t)·v_b‖
  5. Finite-difference speeds ȧ(t), ṙ(t) at 200-step resolution
  6. Merge with pilot_metrics.json, output CSV + plots

Usage:
    python backbone_decomposition.py
    python backbone_decomposition.py --seeds 42 --run-dirs runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY
"""

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════════
# Trunk-only parameter extraction (matches attractor_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)


def flatten_trunk(state_dict):
    """Flatten trunk-only params (attn + MLP weights) into 1-D float64 numpy array.

    Excludes: tok_emb, lm_head (tied), pos_emb, attn.bias (causal mask), all LN.
    Includes: blocks.*.attn.qkv.weight, attn.out_proj.weight, mlp.w_up/w_down.weight.
    """
    parts = []
    for key in sorted(state_dict.keys()):
        if TRUNK_PATTERN.match(key):
            parts.append(state_dict[key].cpu().numpy().astype(np.float64).ravel())
    if not parts:
        raise ValueError("No parameters matched TRUNK_PATTERN — check state_dict keys")
    return np.concatenate(parts)


def load_trunk(path):
    """Load checkpoint and return flattened trunk parameter vector."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vec = flatten_trunk(ckpt["model_state_dict"])
    del ckpt
    return vec


# ═══════════════════════════════════════════════════════════════════════════
# Core decomposition
# ═══════════════════════════════════════════════════════════════════════════

def compute_backbone(run_dir, seed, out_dir):
    """Full backbone decomposition for one seed (trunk params only)."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    ckpt_files = sorted(run_dir.glob("ckpt_*.pt"))
    steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)
    print(f"  Seed {seed}: found {len(steps)} checkpoints, range [{steps[0]}, {steps[-1]}]")

    # Use step 1 as θ₀
    ref_step = steps[0]
    print(f"  Loading θ₀ (step {ref_step}), trunk only...")
    theta0 = load_trunk(run_dir / f"ckpt_{ref_step:06d}.pt")
    n_trunk = len(theta0)
    print(f"  Trunk parameter dimension: {n_trunk:,}")

    # Compute Δθ_t for all checkpoints
    other_steps = [s for s in steps if s != ref_step]
    print(f"  Computing Δθ for {len(other_steps)} checkpoints...")
    deltas = {}
    for i, s in enumerate(other_steps):
        theta_t = load_trunk(run_dir / f"ckpt_{s:06d}.pt")
        deltas[s] = theta_t - theta0
        if (i + 1) % 10 == 0:
            print(f"    loaded {i + 1}/{len(other_steps)}")

    # Form matrix X (rows = Δθ_t)
    X = np.stack([deltas[s] for s in other_steps], axis=0)  # (T, D)

    # Row-normalize (matches beta_summary.py backbone_geometry)
    print("  Row-normalizing + SVD for backbone direction...")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_rn = X / norms

    # Uncentered SVD on row-normalized matrix
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(X_rn, n_components=10, random_state=seed)
    except ImportError:
        U, S, Vt = np.linalg.svd(X_rn, full_matrices=False)

    v_b = Vt[0]  # top right singular vector (unit norm by construction)

    # Explained variance (of row-normalized matrix)
    total_var = np.sum(S ** 2)
    pc1_frac = S[0] ** 2 / total_var
    print(f"  PC1 explains {pc1_frac * 100:.1f}% of row-normalized variance")
    for i in range(min(5, len(S))):
        print(f"    PC{i+1}: {S[i]**2/total_var*100:.2f}%")

    # Also compute explained variance on raw (non-normalized) matrix
    _, S_raw, Vt_raw = np.linalg.svd(X, full_matrices=False)
    total_var_raw = np.sum(S_raw ** 2)
    pc1_frac_raw = S_raw[0] ** 2 / total_var_raw
    print(f"  PC1 raw (unnormalized): {pc1_frac_raw * 100:.1f}%")

    # Sign fix: ensure a(last_step) > 0
    last_step = other_steps[-1]
    if np.dot(deltas[last_step], v_b) < 0:
        v_b = -v_b
        print("  (flipped v_b sign)")

    # ── Compute time series ──
    print("  Computing a(t), r(t) time series...")
    rows = []
    r_vecs = {}

    # Reference step: all zeros
    rows.append({
        "step": ref_step,
        "a_t": 0.0,
        "norm_r_t": 0.0,
        "norm_delta_t": 0.0,
        "backbone_frac": 0.0,
    })
    r_vecs[ref_step] = np.zeros(n_trunk)

    for s in other_steps:
        d = deltas[s]
        a_t = float(np.dot(d, v_b))
        r_vec = d - a_t * v_b
        norm_r = float(np.linalg.norm(r_vec))
        norm_d = float(np.linalg.norm(d))
        bfrac = (a_t ** 2) / (norm_d ** 2) if norm_d > 0 else 0.0

        rows.append({
            "step": s,
            "a_t": a_t,
            "norm_r_t": norm_r,
            "norm_delta_t": norm_d,
            "backbone_frac": float(bfrac),
        })
        r_vecs[s] = r_vec

    rows.sort(key=lambda r: r["step"])

    # ── Finite-difference speeds ──
    print("  Computing speeds ȧ(t), ṙ(t)...")
    for i, row in enumerate(rows):
        s = row["step"]
        if i == 0:
            row["a_dot"] = 0.0
            row["r_dot"] = 0.0
            continue

        prev = rows[i - 1]
        dt = s - prev["step"]
        if dt <= 0:
            row["a_dot"] = 0.0
            row["r_dot"] = 0.0
            continue

        row["a_dot"] = (row["a_t"] - prev["a_t"]) / dt
        row["r_dot"] = float(np.linalg.norm(
            r_vecs[s] - r_vecs[prev["step"]])) / dt

    # Free heavy memory
    del deltas, r_vecs, X, X_rn
    gc.collect()

    # ── Merge with pilot_metrics.json ──
    metrics_path = run_dir / "pilot_metrics.json"
    metrics_by_step = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for m in json.load(f):
                metrics_by_step[m["step"]] = m

    for row in rows:
        s = row["step"]
        if s in metrics_by_step:
            m = metrics_by_step[s]
            row["probe_ood_acc"] = m.get("probe_ood_acc")
            row["probe_in_acc"] = m.get("probe_in_acc")
            row["val_loss"] = m.get("val_loss")
            row["train_loss"] = m.get("train_loss")
            row["lr"] = m.get("lr")
            row["cur_lambda"] = m.get("cur_lambda")

    # ── Save CSV ──
    csv_path = out_dir / f"backbone_timeseries_seed{seed}.csv"
    keys = ["step", "a_t", "norm_r_t", "norm_delta_t", "backbone_frac",
            "a_dot", "r_dot", "probe_ood_acc", "probe_in_acc",
            "val_loss", "train_loss", "lr", "cur_lambda"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = [str(row.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")
    print(f"  Saved: {csv_path}")

    # ── Save singular values ──
    sv_path = out_dir / f"singular_values_seed{seed}.json"
    n_sv = min(20, len(S))
    with open(sv_path, "w") as f:
        json.dump({
            "method": "row-normalized uncentered SVD on trunk params",
            "trunk_dim": n_trunk,
            "n_checkpoints": len(other_steps),
            "singular_values_rownorm": S[:n_sv].tolist(),
            "explained_var_rownorm": [(s**2 / total_var) for s in S[:n_sv]],
            "pc1_pct_rownorm": float(pc1_frac * 100),
            "singular_values_raw": S_raw[:n_sv].tolist(),
            "explained_var_raw": [(s**2 / total_var_raw) for s in S_raw[:n_sv]],
            "pc1_pct_raw": float(pc1_frac_raw * 100),
        }, f, indent=2)

    # ── Plots ──
    if HAS_MPL:
        make_plots(rows, seed, pc1_frac, out_dir)

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(rows, seed, pc1_frac, out_dir):
    """Generate all plots for one seed."""
    steps = [r["step"] for r in rows]
    a_t = [r["a_t"] for r in rows]
    norm_r = [r["norm_r_t"] for r in rows]
    norm_d = [r["norm_delta_t"] for r in rows]
    bfrac = [r["backbone_frac"] for r in rows]
    a_dot = [r["a_dot"] for r in rows]
    r_dot = [r["r_dot"] for r in rows]
    p_ood = [r.get("probe_ood_acc") for r in rows]
    val_loss = [r.get("val_loss") for r in rows]
    lr = [r.get("lr") for r in rows]

    # ── Main 6-panel figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Trunk Backbone Decomposition — Seed {seed}  "
        f"(PC1 = {pc1_frac*100:.1f}%, row-normalized)",
        fontsize=14, fontweight="bold")

    # 1. a(t) vs step
    ax = axes[0, 0]
    ax.plot(steps, a_t, "b-o", ms=3)
    ax.set_xlabel("Step"); ax.set_ylabel("a(t)")
    ax.set_title("Backbone projection a(t)")
    ax.grid(True, alpha=0.3)

    # 2. ||r(t)|| vs step
    ax = axes[0, 1]
    ax.plot(steps, norm_r, "r-o", ms=3)
    ax.set_xlabel("Step"); ax.set_ylabel("‖r(t)‖")
    ax.set_title("Residual norm ‖r(t)‖")
    ax.grid(True, alpha=0.3)

    # 3. p_ood overlaid with ||r(t)||
    ax = axes[0, 2]
    ax_r = ax.twinx()
    valid = [(s, p) for s, p in zip(steps, p_ood) if p is not None]
    if valid:
        s_p, p_vals = zip(*valid)
        ax.plot(s_p, p_vals, "g-o", ms=3, label="p_ood")
    ax_r.plot(steps, norm_r, "r-s", ms=2, alpha=0.7, label="‖r(t)‖")
    ax.set_xlabel("Step")
    ax.set_ylabel("probe_ood_acc", color="green")
    ax_r.set_ylabel("‖r(t)‖", color="red")
    ax.set_title("p_ood vs ‖r(t)‖")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Scatter: p_ood vs ||r(t)||
    ax = axes[1, 0]
    valid = [(r_val, p, s) for r_val, p, s in zip(norm_r, p_ood, steps) if p is not None]
    if valid:
        r_vals, p_vals, s_vals = zip(*valid)
        sc = ax.scatter(r_vals, p_vals, c=s_vals, cmap="viridis", s=25)
        plt.colorbar(sc, ax=ax, label="step")
        ax.set_xlabel("‖r(t)‖"); ax.set_ylabel("probe_ood_acc")
        ax.set_title("p_ood vs ‖r(t)‖ (color=step)")
        ax.grid(True, alpha=0.3)

    # 5. Speeds: ȧ(t) and ṙ(t)
    ax = axes[1, 1]
    ax.plot(steps[1:], a_dot[1:], "b-o", ms=3, label="ȧ(t)")
    ax.plot(steps[1:], r_dot[1:], "r-s", ms=3, label="ṙ(t)")
    ax.set_xlabel("Step"); ax.set_ylabel("Speed (per step)")
    ax.set_title("Speeds ȧ(t), ṙ(t)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. ṙ(t) vs lr
    ax = axes[1, 2]
    valid_lr = [(l, rd, s) for l, rd, s in zip(lr[1:], r_dot[1:], steps[1:]) if l is not None]
    if valid_lr:
        l_vals, rd_vals, s_vals = zip(*valid_lr)
        sc = ax.scatter(l_vals, rd_vals, c=s_vals, cmap="viridis", s=25)
        plt.colorbar(sc, ax=ax, label="step")
        ax.set_xlabel("Learning rate"); ax.set_ylabel("ṙ(t)")
        ax.set_title("ṙ(t) vs lr (color=step)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"backbone_decomposition_seed{seed}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Backbone fraction + norms ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, bfrac, "k-o", ms=3)
    ax1.set_xlabel("Step"); ax1.set_ylabel("a(t)² / ‖Δθ‖²")
    ax1.set_title(f"Backbone fraction (seed {seed})")
    ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, norm_d, "k-o", ms=3, label="‖Δθ‖")
    ax2.plot(steps, [abs(a) for a in a_t], "b--", ms=2, label="|a(t)|")
    ax2.plot(steps, norm_r, "r--", ms=2, label="‖r(t)‖")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Norm")
    ax2.set_title(f"Trunk decomposition norms (seed {seed})")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"backbone_norms_seed{seed}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Log-log scaling fits ──
    valid_steps = [s for s in steps if s > 1]
    valid_a = [abs(a) for s, a in zip(steps, a_t) if s > 1]
    valid_r = [r for s, r in zip(steps, norm_r) if s > 1]

    if valid_steps and all(v > 0 for v in valid_a) and all(v > 0 for v in valid_r):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        log_s = np.log10(valid_steps)
        log_a = np.log10(valid_a)
        log_r = np.log10(valid_r)

        ax1.plot(log_s, log_a, "b-o", ms=3)
        coeffs_a = np.polyfit(log_s, log_a, 1)
        ax1.plot(log_s, np.polyval(coeffs_a, log_s), "b--", alpha=0.5,
                 label=f"slope={coeffs_a[0]:.2f}")
        ax1.set_xlabel("log₁₀(step)"); ax1.set_ylabel("log₁₀(|a(t)|)")
        ax1.set_title(f"‖Δ_∥(t)‖ scaling (seed {seed})")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(log_s, log_r, "r-o", ms=3)
        coeffs_r = np.polyfit(log_s, log_r, 1)
        ax2.plot(log_s, np.polyval(coeffs_r, log_s), "r--", alpha=0.5,
                 label=f"slope={coeffs_r[0]:.2f}")
        ax2.set_xlabel("log₁₀(step)"); ax2.set_ylabel("log₁₀(‖r(t)‖)")
        ax2.set_title(f"‖Δ_⊥(t)‖ scaling (seed {seed})")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / f"backbone_loglog_seed{seed}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")
        print(f"  Log-log slopes: a(t) ~ t^{coeffs_a[0]:.2f}, ‖r(t)‖ ~ t^{coeffs_r[0]:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backbone decomposition a(t), r(t) on trunk params")
    parser.add_argument("--seeds", type=str, default="42,271")
    parser.add_argument("--run-dirs", type=str, default=None,
                        help="Comma-separated run directories (must match --seeds order)")
    parser.add_argument("--out-dir", type=str, default="analysis/backbone_decomposition")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    default_dirs = {
        42: "runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY",
        271: "runs/pilot_wd0.5_lr0.001_lp2.0_s271",
    }

    if args.run_dirs:
        run_dirs = args.run_dirs.split(",")
        assert len(run_dirs) == len(seeds), "Must provide one dir per seed"
        seed_dirs = dict(zip(seeds, run_dirs))
    else:
        seed_dirs = {s: default_dirs.get(s, f"runs/pilot_wd0.5_lr0.001_lp2.0_s{s}")
                     for s in seeds}

    out_dir = Path(args.out_dir)

    print("=" * 60)
    print("  Trunk Backbone Decomposition: a(t), r(t)")
    print("  (attn + MLP weights only, row-normalized SVD)")
    print("=" * 60)

    all_results = {}
    for seed in seeds:
        rd = seed_dirs[seed]
        if not Path(rd).exists():
            print(f"\n  [SKIP] seed {seed}: {rd} not found")
            continue
        print(f"\n  Processing seed {seed}: {rd}")
        rows = compute_backbone(rd, seed, out_dir)
        all_results[seed] = rows

    # ── Cross-seed comparison ──
    if HAS_MPL and len(all_results) >= 2:
        print("\n  Generating cross-seed comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cross-Seed Trunk Backbone Comparison (row-normalized)",
                     fontsize=14, fontweight="bold")
        colors = {42: "C0", 271: "C1"}

        for seed, rows in all_results.items():
            c = colors.get(seed, "gray")
            st = [r["step"] for r in rows]
            at = [r["a_t"] for r in rows]
            nr = [r["norm_r_t"] for r in rows]
            bf = [r["backbone_frac"] for r in rows]
            po = [r.get("probe_ood_acc") for r in rows]

            axes[0, 0].plot(st, at, "-o", color=c, ms=3, label=f"seed {seed}")
            axes[0, 1].plot(st, nr, "-o", color=c, ms=3, label=f"seed {seed}")
            axes[1, 0].plot(st, bf, "-o", color=c, ms=3, label=f"seed {seed}")

            valid = [(s, p) for s, p in zip(st, po) if p is not None]
            if valid:
                sp, pv = zip(*valid)
                axes[1, 1].plot(sp, pv, "-o", color=c, ms=3, label=f"seed {seed}")

        axes[0, 0].set_title("a(t)"); axes[0, 0].set_xlabel("Step")
        axes[0, 1].set_title("‖r(t)‖"); axes[0, 1].set_xlabel("Step")
        axes[1, 0].set_title("Backbone fraction"); axes[1, 0].set_xlabel("Step")
        axes[1, 1].set_title("probe_ood_acc"); axes[1, 1].set_xlabel("Step")
        for ax in axes.flat:
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "backbone_cross_seed.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    print(f"\n{'='*60}")
    print(f"  Done. Output in {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
