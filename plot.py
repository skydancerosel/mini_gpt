#!/usr/bin/env python3
"""
Visualization for TinyStories + Probe grokking experiment.

Generates figures from saved results:
  fig1 — Training curves: val loss + probe accuracy over time
  fig2 — Commutator defect vs probe_ood accuracy (emergence detection)
  fig3 — PCA explained variance / effective dimensionality
  fig4 — Lead time analysis across weight decays
  fig5 — Weight decay comparison (val loss, probe_in, probe_ood)

Usage:
    python plot.py                    # plot all runs in runs/
    python plot.py --run wd1e-3_s42   # plot single run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("plots")

COLORS = {
    "wd0.0": "#d62728",
    "wd0.0001": "#ff7f0e",
    "wd0.0003": "#2ca02c",
    "wd0.001": "#1f77b4",
    "wd0.003": "#9467bd",
}


def load_results(run_dir):
    """Load results.pt and metrics.json from a run directory."""
    results_path = run_dir / "results.pt"
    if not results_path.exists():
        return None
    return torch.load(results_path, map_location="cpu", weights_only=False)


def get_color(tag):
    for key, color in COLORS.items():
        if key in tag:
            return color
    return "#333333"


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Training curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(all_results, out_dir):
    """Val loss + probe accuracies over training for each run."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for tag, result in sorted(all_results.items()):
        metrics = result.get("metrics", [])
        if not metrics:
            continue
        steps = [m["step"] for m in metrics]
        color = get_color(tag)

        axes[0].plot(steps, [m["val_loss"] for m in metrics],
                     label=tag, color=color, linewidth=1.5)
        axes[1].plot(steps, [m["probe_in_acc"] for m in metrics],
                     label=tag, color=color, linewidth=1.5)
        axes[2].plot(steps, [m["probe_ood_acc"] for m in metrics],
                     label=tag, color=color, linewidth=1.5)

    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("LM Validation Loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Probe In-Distribution Accuracy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Probe OOD Accuracy")

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("TinyStories + Key Retrieval Probe: Training Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig1_training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Defect vs Probe OOD (emergence detection)
# ═══════════════════════════════════════════════════════════════════════════

def plot_defect_vs_emergence(all_results, out_dir):
    """Overlay commutator defect with probe_ood accuracy."""
    n_runs = len(all_results)
    if n_runs == 0:
        return

    fig, axes = plt.subplots(1, min(n_runs, 5), figsize=(6 * min(n_runs, 5), 5),
                             squeeze=False)

    for i, (tag, result) in enumerate(sorted(all_results.items())):
        if i >= 5:
            break
        ax = axes[0, i]
        ax2 = ax.twinx()

        metrics = result.get("metrics", [])
        defect_log = result.get("defect_log", [])

        if metrics:
            steps_m = [m["step"] for m in metrics]
            ood_acc = [m["probe_ood_acc"] for m in metrics]
            ax2.plot(steps_m, ood_acc, color="#e74c3c", linewidth=2,
                     linestyle="--", label="Probe OOD acc")

        if defect_log:
            steps_d = [d["step"] for d in defect_log]
            defects = [d["defect_median"] for d in defect_log]
            ax.plot(steps_d, defects, color="#1a5276", linewidth=2,
                    label="Defect")

        emergence = result.get("emergence_step")
        defect_onset = result.get("defect_onset")
        if emergence is not None:
            ax.axvline(x=emergence, color="#e74c3c", linestyle=":",
                       alpha=0.7, label=f"Emergence ({emergence})")
        if defect_onset is not None:
            ax.axvline(x=defect_onset, color="#1a5276", linestyle=":",
                       alpha=0.7, label=f"Defect onset ({defect_onset})")

        ax.set_yscale("log")
        ax.set_ylabel("Defect", color="#1a5276")
        ax2.set_ylabel("Probe OOD acc", color="#e74c3c")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step")
        ax.set_title(tag, fontsize=10)
        ax.grid(alpha=0.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle("Commutator Defect vs Probe OOD Accuracy\n"
                 "(Does defect spike predict emergence?)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_defect_vs_emergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig2_defect_vs_emergence.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: PCA dimensionality
# ═══════════════════════════════════════════════════════════════════════════

def plot_pca_analysis(all_results, out_dir):
    """PCA explained variance and effective dimensionality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for tag, result in sorted(all_results.items()):
        pca_log = result.get("pca_log", [])
        if not pca_log:
            continue
        color = get_color(tag)

        steps = [p["step"] for p in pca_log]
        k95 = [p["k_star_95"] for p in pca_log]
        k99 = [p["k_star_99"] for p in pca_log]
        pc1 = [p["explained_variance_ratio"][0] for p in pca_log]

        axes[0].plot(steps, k95, label=f"{tag} (k*95)", color=color,
                     linewidth=1.5, linestyle="-")
        axes[0].plot(steps, k99, label=f"{tag} (k*99)", color=color,
                     linewidth=1.5, linestyle="--", alpha=0.7)

        axes[1].plot(steps, pc1, label=tag, color=color, linewidth=1.5)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Effective dimensionality (k*)")
    axes[0].set_title("PCA: Effective Dimensionality of Weight Updates")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("PC1 explained variance ratio")
    axes[1].set_title("PCA: Top Component Dominance")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig3_pca_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Lead time analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_lead_time(all_results, out_dir):
    """Bar chart of lead times across weight decays."""
    tags = []
    leads = []
    for tag, result in sorted(all_results.items()):
        em = result.get("emergence_step")
        do = result.get("defect_onset")
        lead = result.get("lead_time")
        tags.append(tag)
        leads.append(lead if lead is not None else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tags))
    colors = [get_color(t) for t in tags]
    bars = ax.bar(x, leads, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)

    for bar, lead in zip(bars, leads):
        if lead != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    str(lead), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Lead time (steps)")
    ax.set_title("Defect Onset → Probe OOD Emergence: Lead Time\n"
                 "(positive = defect precedes emergence)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_lead_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig4_lead_time.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Weight decay comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_wd_comparison(all_results, out_dir):
    """Final metrics comparison across weight decays."""
    tags = []
    val_losses = []
    probe_ins = []
    probe_oods = []

    for tag, result in sorted(all_results.items()):
        metrics = result.get("metrics", [])
        if not metrics:
            continue
        last = metrics[-1]
        tags.append(tag)
        val_losses.append(last.get("val_loss", 0))
        probe_ins.append(last.get("probe_in_acc", 0))
        probe_oods.append(last.get("probe_ood_acc", 0))

    if not tags:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(tags))
    colors = [get_color(t) for t in tags]

    axes[0].bar(x, val_losses, color=colors, alpha=0.85)
    axes[0].set_ylabel("Val loss")
    axes[0].set_title("Final Validation Loss")

    axes[1].bar(x, probe_ins, color=colors, alpha=0.85)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Final Probe In-Dist Accuracy")

    axes[2].bar(x, probe_oods, color=colors, alpha=0.85)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Final Probe OOD Accuracy")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=7, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Weight Decay Sweep: Final Metrics Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_wd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig5_wd_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--run", type=str, default=None,
                       help="Plot single run (directory name)")
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Directory containing run outputs")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    runs_dir = Path(args.runs_dir)

    if args.run:
        run_dirs = [runs_dir / args.run]
    else:
        run_dirs = sorted(runs_dir.glob("*"))
        run_dirs = [d for d in run_dirs if d.is_dir()]

    all_results = {}
    for rd in run_dirs:
        result = load_results(rd)
        if result is not None:
            all_results[rd.name] = result
            print(f"  Loaded {rd.name}")

    if not all_results:
        print("No results found. Run main.py first.")
        return

    print(f"\nGenerating figures for {len(all_results)} runs...")
    plot_training_curves(all_results, OUT_DIR)
    plot_defect_vs_emergence(all_results, OUT_DIR)
    plot_pca_analysis(all_results, OUT_DIR)
    plot_lead_time(all_results, OUT_DIR)
    plot_wd_comparison(all_results, OUT_DIR)
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
