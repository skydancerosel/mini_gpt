#!/usr/bin/env python3
"""
Plot switch-pair delta vs step with noise-calibrated significance band.

x-axis: step (peak step of each switch pair)
y-axis: delta = p_ood(peak) - p_ood(trough)
shaded band: ±3*std(p_ood) from bootstrap noise estimate
dots: all detected switch pairs, colored by seed

Requires eval_noise.json from estimate_eval_noise.py.

Usage:
    python plot_delta_significance.py
    python plot_delta_significance.py --seeds 42,271
    python plot_delta_significance.py --seeds 42,271,137
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_seed_data(run_dir):
    """Load metrics, manifest, and noise for one seed."""
    run_dir = Path(run_dir)

    metrics_path = run_dir / "pilot_metrics.json"
    manifest_path = run_dir / "oscillation_manifest.json"
    noise_path = run_dir / "eval_noise.json"

    if not metrics_path.exists() or not manifest_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    step_pood = {m["step"]: m["probe_ood_acc"] for m in metrics}

    noise_std = None
    if noise_path.exists():
        with open(noise_path) as f:
            noise = json.load(f)
        noise_std = noise["std_p_ood"]

    # Build switch pair deltas
    pairs = []
    for sp in manifest["switch_pairs"]:
        p, t = sp["peak"], sp["trough"]
        pp = step_pood.get(p, 0)
        pt = step_pood.get(t, 0)
        pairs.append({
            "peak_step": p,
            "trough_step": t,
            "p_peak": pp,
            "p_trough": pt,
            "delta": pp - pt,
        })

    return {
        "step_pood": step_pood,
        "pairs": pairs,
        "noise_std": noise_std,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot delta significance across seeds"
    )
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: auto-detect)")
    parser.add_argument("--out", type=str, default="fig_delta_significance.png")
    args = parser.parse_args()

    base = Path("runs")

    # Auto-detect seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = []
        for d in sorted(base.glob("pilot_wd0.5_lr0.001_lp2.0_s*")):
            name = d.name
            try:
                s = int(name.split("_s")[-1])
                if (d / "oscillation_manifest.json").exists():
                    seeds.append(s)
            except ValueError:
                continue
        print(f"Auto-detected seeds: {seeds}")

    # Load data
    seed_data = {}
    noise_stds = []
    for seed in seeds:
        run_dir = base / f"pilot_wd0.5_lr0.001_lp2.0_s{seed}"
        data = load_seed_data(run_dir)
        if data:
            seed_data[seed] = data
            if data["noise_std"] is not None:
                noise_stds.append(data["noise_std"])
                print(f"  seed {seed}: {len(data['pairs'])} pairs, "
                      f"std={data['noise_std']:.4f}")
            else:
                print(f"  seed {seed}: {len(data['pairs'])} pairs, "
                      f"no noise estimate")

    if not seed_data:
        print("ERROR: No seed data found")
        return

    # Use mean noise_std across seeds (should be similar)
    if noise_stds:
        avg_std = np.mean(noise_stds)
    else:
        # Fallback: binomial SE with p=0.3, N=2000
        avg_std = np.sqrt(0.3 * 0.7 / 2000)
        print(f"  No noise estimates found, using binomial SE: {avg_std:.4f}")

    print(f"\n  Average std(p_ood) = {avg_std:.4f}")
    print(f"  3*std = {3*avg_std:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Shaded significance bands
    ax.axhspan(-3 * avg_std, 3 * avg_std, alpha=0.12, color="gray",
               label=f"$\\pm 3\\sigma$ (std={avg_std:.3f})")
    ax.axhspan(-2 * avg_std, 2 * avg_std, alpha=0.10, color="gray")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.axhline(3 * avg_std, color="red", ls="--", lw=1, alpha=0.5,
               label=f"$3\\sigma = {3*avg_std:.3f}$")
    ax.axhline(2 * avg_std, color="orange", ls=":", lw=1, alpha=0.5,
               label=f"$2\\sigma = {2*avg_std:.3f}$")

    # Color map for seeds
    colors = {42: "#1f77b4", 137: "#ff7f0e", 271: "#2ca02c",
              314: "#d62728", 0: "#9467bd"}
    markers = {42: "o", 137: "s", 271: "D", 314: "^", 0: "v"}

    for seed, data in sorted(seed_data.items()):
        c = colors.get(seed, "#333333")
        m = markers.get(seed, "o")
        steps = [p["peak_step"] for p in data["pairs"]]
        deltas = [p["delta"] for p in data["pairs"]]

        ax.scatter(steps, deltas, c=c, marker=m, s=70, edgecolors="black",
                   linewidth=0.5, zorder=5, label=f"seed {seed}")

        # Annotate large deltas
        for p in data["pairs"]:
            if p["delta"] > 0.15:
                ax.annotate(f'{p["delta"]:.2f}',
                            (p["peak_step"], p["delta"]),
                            textcoords="offset points", xytext=(5, 8),
                            fontsize=7, color=c, alpha=0.8)

    ax.set_xlabel("Step (peak checkpoint)", fontsize=12)
    ax.set_ylabel("$\\delta$ = p_ood(peak) $-$ p_ood(trough)", fontsize=12)
    ax.set_title("Switch-pair significance: $\\delta$ vs training step",
                 fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, None)

    # Tier labels on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    tier_y = {
        f"Tier C (<{2*avg_std:.2f})": avg_std,
        f"Tier B ({2*avg_std:.2f}–{3*avg_std:.2f})": 2.5 * avg_std,
        f"Tier A (>{3*avg_std:.2f})": max(ax.get_ylim()[1] * 0.8, 4 * avg_std),
    }
    ax2.set_yticks(list(tier_y.values()))
    ax2.set_yticklabels(list(tier_y.keys()), fontsize=8)
    ax2.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
