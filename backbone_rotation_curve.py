#!/usr/bin/env python3
"""
1) Sliding-window rotation curve: ρ(t) = |⟨v_b^(w_t), v_b^(w_{t+1})⟩|
2) Alignment to phase backbones: A_E(t), A_L(t)

Uses W=10 checkpoints (2000 steps) sliding windows on trunk params.
"""

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

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)

OUT_DIR = Path("analysis/backbone_decomposition")


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


def compute_vb_from_deltas(deltas):
    """Row-normalized PC1 from a list of delta vectors. Returns (v, pc1_pct)."""
    X = np.stack(deltas)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_rn = X / np.maximum(norms, 1e-12)
    _, S, Vt = np.linalg.svd(X_rn, full_matrices=False)
    total = np.sum(S ** 2)
    pc1_pct = S[0] ** 2 / total * 100
    v = Vt[0]
    # sign fix: positive at last delta
    if np.dot(deltas[-1], v) < 0:
        v = -v
    return v, pc1_pct


def run_seed(seed, run_dir, W=10):
    run_dir = Path(run_dir)
    all_steps = [1] + list(range(200, 10001, 200))

    print(f"\n  Loading {len(all_steps)} checkpoints for seed {seed}...")
    thetas = {}
    for i, s in enumerate(all_steps):
        thetas[s] = load_trunk(run_dir / f"ckpt_{s:06d}.pt")
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(all_steps)}")

    theta0 = thetas[all_steps[0]]
    # Deltas relative to step 1
    deltas_from_0 = {s: thetas[s] - theta0 for s in all_steps if s != all_steps[0]}

    # ── Phase backbones ──
    early_steps = [s for s in all_steps if 1 < s <= 4000]
    late_steps_raw = [s for s in all_steps if s >= 4000]
    theta4k = thetas[4000]
    deltas_from_4k = {s: thetas[s] - theta4k for s in late_steps_raw if s != 4000}

    print("  Computing phase backbones v_E, v_L...")
    v_E, pc1_E = compute_vb_from_deltas([deltas_from_0[s] for s in early_steps])
    v_L, pc1_L = compute_vb_from_deltas([deltas_from_4k[s] for s in sorted(deltas_from_4k.keys())])
    print(f"    v_E (0-4k): PC1={pc1_E:.1f}%")
    print(f"    v_L (4k-10k): PC1={pc1_L:.1f}%")
    print(f"    |⟨v_E, v_L⟩| = {abs(np.dot(v_E, v_L)):.4f}")

    # ── Sliding windows ──
    # Each window: W consecutive checkpoints, drift relative to first in window
    other_steps = [s for s in all_steps if s != all_steps[0]]  # exclude step 1 ref
    n = len(other_steps)
    n_windows = n - W + 1

    print(f"  Computing {n_windows} sliding windows (W={W}, {W*200} steps each)...")
    window_vbs = []
    window_centers = []
    window_pc1s = []

    for i in range(n_windows):
        win_steps = other_steps[i:i + W]
        # Drift relative to first checkpoint in window
        ref = thetas[win_steps[0]]
        win_deltas = [thetas[s] - ref for s in win_steps[1:]]
        v, pc1 = compute_vb_from_deltas(win_deltas)
        window_vbs.append(v)
        center = (win_steps[0] + win_steps[-1]) / 2
        window_centers.append(center)
        window_pc1s.append(pc1)

    # ── 1) Rotation curve ρ(t) ──
    rho = []
    rho_centers = []
    for i in range(len(window_vbs) - 1):
        cos = abs(float(np.dot(window_vbs[i], window_vbs[i + 1])))
        rho.append(cos)
        rho_centers.append((window_centers[i] + window_centers[i + 1]) / 2)

    # ── 2) Alignment to phase backbones ──
    A_E = [abs(float(np.dot(v, v_E))) for v in window_vbs]
    A_L = [abs(float(np.dot(v, v_L))) for v in window_vbs]

    # Print summary
    print(f"\n  Rotation curve ρ(t):")
    print(f"    min ρ = {min(rho):.4f} at step ≈ {rho_centers[np.argmin(rho)]:.0f}")
    print(f"    max ρ = {max(rho):.4f}")
    print(f"    mean ρ = {np.mean(rho):.4f}")

    # Find dip region
    dip_idx = np.argmin(rho)
    dip_step = rho_centers[dip_idx]
    print(f"    Dip centered at step ≈ {dip_step:.0f}")

    print(f"\n  Phase alignment:")
    print(f"    {'center':>6s}  {'A_E':>6s}  {'A_L':>6s}  {'PC1%':>5s}")
    for i in range(0, len(window_centers), max(1, len(window_centers) // 12)):
        print(f"    {window_centers[i]:6.0f}  {A_E[i]:6.3f}  {A_L[i]:6.3f}  {window_pc1s[i]:5.1f}")

    # ── Plots ──
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Backbone Rotation — Seed {seed}", fontsize=14, fontweight="bold")

        # Panel 1: ρ(t)
        ax = axes[0]
        ax.plot(rho_centers, rho, "k-o", ms=4)
        ax.axvline(4000, color="red", ls="--", alpha=0.7, label="λ switch (4k)")
        ax.set_xlabel("Step (window midpoint)")
        ax.set_ylabel("ρ(t) = |⟨v_b^(w_t), v_b^(w_{t+1})⟩|")
        ax.set_title("Rotation curve ρ(t)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: A_E, A_L
        ax = axes[1]
        ax.plot(window_centers, A_E, "C0-o", ms=4, label="A_E = |⟨v_w, v_E⟩|")
        ax.plot(window_centers, A_L, "C1-s", ms=4, label="A_L = |⟨v_w, v_L⟩|")
        ax.axvline(4000, color="red", ls="--", alpha=0.7, label="λ switch (4k)")
        ax.set_xlabel("Step (window midpoint)")
        ax.set_ylabel("Alignment")
        ax.set_title("Alignment to phase backbones")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Window PC1%
        ax = axes[2]
        ax.plot(window_centers, window_pc1s, "k-o", ms=4)
        ax.axvline(4000, color="red", ls="--", alpha=0.7, label="λ switch (4k)")
        ax.set_xlabel("Step (window midpoint)")
        ax.set_ylabel("Window PC1 %")
        ax.set_title("Local backbone strength")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = OUT_DIR / f"rotation_curve_seed{seed}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {fig_path}")

    return {
        "rho_centers": rho_centers, "rho": rho,
        "window_centers": window_centers,
        "A_E": A_E, "A_L": A_L, "pc1s": window_pc1s,
    }


def main():
    seeds = {
        42: "runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY",
        271: "runs/pilot_wd0.5_lr0.001_lp2.0_s271",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Backbone Rotation Curve + Phase Alignment")
    print("  W=10 checkpoints (2000 steps per window)")
    print("=" * 60)

    all_results = {}
    for seed, run_dir in seeds.items():
        if not Path(run_dir).exists():
            print(f"\n  [SKIP] seed {seed}")
            continue
        result = run_seed(seed, run_dir)
        all_results[seed] = result

    # Cross-seed overlay
    if HAS_MPL and len(all_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Cross-Seed Rotation & Alignment", fontsize=14, fontweight="bold")
        colors = {42: "C0", 271: "C1"}

        for seed, r in all_results.items():
            c = colors.get(seed, "gray")
            axes[0].plot(r["rho_centers"], r["rho"], f"-o", color=c, ms=3,
                         label=f"seed {seed}")
            axes[1].plot(r["window_centers"], r["A_E"], f"-o", color=c, ms=3,
                         label=f"s{seed} A_E")
            axes[1].plot(r["window_centers"], r["A_L"], f"--s", color=c, ms=3,
                         alpha=0.7, label=f"s{seed} A_L")

        for ax in axes:
            ax.axvline(4000, color="red", ls="--", alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.05)

        axes[0].set_title("ρ(t)")
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("ρ")
        axes[1].set_title("A_E (solid) and A_L (dashed)")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Alignment")

        plt.tight_layout()
        fig_path = OUT_DIR / "rotation_curve_cross_seed.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {fig_path}")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
