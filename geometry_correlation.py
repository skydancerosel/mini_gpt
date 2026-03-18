#!/usr/bin/env python3
"""
Geometry–LM Performance Correlation + Predictability Model.

Phase 2: Correlates causal geometry with LM/probe performance across runs.
Phase 4: Linear model predicting reheating gain G/D from local geometry.

Usage:
    python geometry_correlation.py --phase 2
    python geometry_correlation.py --phase 4
    python geometry_correlation.py --phase all
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = Path("runs/beta2_ablation")
RESULTS_DIR = Path("results")


# ── Phase 2: Correlation Aggregation ──────────────────────────────────────


def load_run_metrics(run_dir):
    """Load pilot_metrics.json from a run directory."""
    path = run_dir / "pilot_metrics.json"
    if not path.exists():
        return None
    return json.load(open(path))


def load_causal_geometry(run_dir):
    """Load causal_geometry.json from a run directory."""
    path = run_dir / "causal_geometry.json"
    if not path.exists():
        return None
    return json.load(open(path))


def compute_lm_metrics(metrics):
    """Extract LM performance metrics from pilot_metrics.json."""
    if not metrics:
        return {}

    val_losses = [(m["step"], m["val_loss"]) for m in metrics]
    best_val_loss = min(v for _, v in val_losses)
    final_val_loss = val_losses[-1][1]
    ppl_final = math.exp(final_val_loss)

    # time_to_val1.8 (interpolated)
    time_to_val1_8 = None
    for i in range(1, len(val_losses)):
        s0, v0 = val_losses[i - 1]
        s1, v1 = val_losses[i]
        if v0 >= 1.8 > v1:
            # Linear interpolation
            frac = (1.8 - v0) / (v1 - v0) if v1 != v0 else 0
            time_to_val1_8 = s0 + frac * (s1 - s0)
            break

    # time_to_val1.6
    time_to_val1_6 = None
    for i in range(1, len(val_losses)):
        s0, v0 = val_losses[i - 1]
        s1, v1 = val_losses[i]
        if v0 >= 1.6 > v1:
            frac = (1.6 - v0) / (v1 - v0) if v1 != v0 else 0
            time_to_val1_6 = s0 + frac * (s1 - s0)
            break

    # Generalization gap (train_loss - val_loss at final step)
    gen_gap = metrics[-1].get("train_loss", 0) - final_val_loss

    return {
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "ppl_final": ppl_final,
        "time_to_val1.8": time_to_val1_8,
        "time_to_val1.6": time_to_val1_6,
        "generalization_gap_final": gen_gap,
    }


def compute_probe_metrics(metrics):
    """Extract probe performance metrics from pilot_metrics.json."""
    if not metrics:
        return {}

    pood_vals = [(m["step"], m["probe_ood_acc"]) for m in metrics]
    best_p_ood = max(v for _, v in pood_vals)
    step_best = [s for s, v in pood_vals if v == best_p_ood][0]
    final_p_ood = pood_vals[-1][1]

    # AUC (trapezoidal)
    auc = 0.0
    for i in range(1, len(pood_vals)):
        s0, v0 = pood_vals[i - 1]
        s1, v1 = pood_vals[i]
        auc += (s1 - s0) * (v0 + v1) / 2
    # Normalize by total step range
    total_range = pood_vals[-1][0] - pood_vals[0][0]
    auc_normalized = auc / total_range if total_range > 0 else 0

    # Oscillation amplitude: std of first 2 local extrema differences
    # Simplified: std of p_ood over steps 2000-6000
    mid_vals = [v for s, v in pood_vals if 2000 <= s <= 6000]
    osc_amp = float(np.std(mid_vals)) if mid_vals else 0.0

    return {
        "best_p_ood": best_p_ood,
        "step_best_p_ood": step_best,
        "AUC_p_ood": auc_normalized,
        "final_p_ood": final_p_ood,
        "oscillation_amplitude_first2": osc_amp,
    }


def pearson_corr(x, y):
    """Pearson correlation, handling NaN/None."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    a, b = zip(*pairs)
    a, b = np.array(a), np.array(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_phase2(run_dirs):
    """Phase 2: Correlation aggregation across runs."""
    print("\n" + "=" * 60)
    print("  Phase 2: Correlation Aggregation")
    print("=" * 60)

    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            print(f"  [SKIP] {run_dir}")
            continue

        metrics = load_run_metrics(run_dir)
        geometry = load_causal_geometry(run_dir)
        if metrics is None or geometry is None:
            print(f"  [SKIP] Missing metrics or geometry: {run_dir.name}")
            continue

        lm = compute_lm_metrics(metrics)
        probe = compute_probe_metrics(metrics)
        summaries = geometry.get("summaries", {})

        row = {"run": run_dir.name}
        row.update(lm)
        row.update(probe)
        row.update(summaries)
        rows.append(row)
        print(f"  Loaded: {run_dir.name}")

    if len(rows) < 2:
        print(f"\n  Only {len(rows)} runs found. Need >= 2 for correlations.")
        # Still save what we have
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "geometry_lm_table.csv", "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        print(f"  Saved table with {len(rows)} rows")
        return rows

    # Correlation pairs from spec
    corr_pairs = [
        ("mean_PC1_0_2k", "time_to_val1.8"),
        ("mean_align_u_0_2k", "time_to_val1.8"),
        ("mean_PC1_0_10k", "final_val_loss"),
        ("best_p_ood", "mean_PC1_0_2k"),
        ("AUC_p_ood", "mean_align_u_0_2k"),
    ]

    correlations = {}
    print(f"\n  Correlations across {len(rows)} runs:")
    print(f"    {'Metric A':>25s}  {'Metric B':>25s}  {'r':>8s}")
    print(f"    {'-'*65}")

    for a_key, b_key in corr_pairs:
        a_vals = [r.get(a_key) for r in rows]
        b_vals = [r.get(b_key) for r in rows]
        r = pearson_corr(a_vals, b_vals)
        correlations[f"corr({a_key}, {b_key})"] = r
        r_str = f"{r:.4f}" if r is not None else "N/A"
        print(f"    {a_key:>25s}  {b_key:>25s}  {r_str:>8s}")

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "geometry_lm_correlations.json", "w") as f:
        json.dump({"n_runs": len(rows), "correlations": correlations}, f, indent=2)

    with open(RESULTS_DIR / "geometry_lm_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n  Saved: results/geometry_lm_correlations.json")
    print(f"  Saved: results/geometry_lm_table.csv")

    # Scatter plots
    if HAS_MPL and len(rows) >= 2:
        n_pairs = len(corr_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5))
        if n_pairs == 1:
            axes = [axes]
        fig.suptitle("Geometry–Performance Correlations", fontsize=13, fontweight="bold")

        for ax, (a_key, b_key) in zip(axes, corr_pairs):
            a_vals = [r.get(a_key) for r in rows]
            b_vals = [r.get(b_key) for r in rows]
            labels = [r["run"].split("b2")[1].split("_")[0] if "b2" in r["run"] else r["run"]
                      for r in rows]

            valid = [(a, b, l) for a, b, l in zip(a_vals, b_vals, labels)
                     if a is not None and b is not None]
            if not valid:
                ax.set_title(f"{a_key} vs {b_key}\n(no data)")
                continue

            xs, ys, ls = zip(*valid)
            ax.scatter(xs, ys, s=60, zorder=5)
            for x, y, l in zip(xs, ys, ls):
                ax.annotate(f"β2={l}", (x, y), fontsize=7, ha="center", va="bottom")

            r = pearson_corr(list(xs), list(ys))
            r_str = f"r={r:.3f}" if r is not None else ""
            ax.set_xlabel(a_key, fontsize=9)
            ax.set_ylabel(b_key, fontsize=9)
            ax.set_title(f"{a_key}\nvs {b_key}\n{r_str}", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = RESULTS_DIR / "geometry_lm_scatter_plots.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    return rows


# ── Phase 4: Predictability Model ────────────────────────────────────────


def run_phase4(run_dirs):
    """Phase 4: Linear model predicting reheating G/D from local geometry."""
    print("\n" + "=" * 60)
    print("  Phase 4: Predictability Model")
    print("=" * 60)

    # Load reheat summary
    csv_path = Path("runs/reheat_sweep/reheat_summary.csv")
    if not csv_path.exists():
        print(f"  [ERROR] Reheat summary not found: {csv_path}")
        return

    reheat_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reheat_rows.append(row)

    print(f"  Loaded {len(reheat_rows)} reheat conditions")

    # Load geometry for each parent run
    geometry_cache = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        geo = load_causal_geometry(run_dir)
        if geo:
            geometry_cache[run_dir.name] = geo

    # Match reheat conditions to geometry at checkpoint step
    feature_names = ["PC1_roll", "align_u", "drift_speed", "r_speed", "kappa", "a_speed"]
    X_rows = []
    G_vals = []
    D_vals = []
    labels = []

    for rh in reheat_rows:
        beta2 = float(rh["beta2"])
        ckpt_step = int(rh["ckpt_step"])
        G = float(rh.get("G", 0))
        D = float(rh.get("D", 0))

        # Find matching geometry
        b2_str = f"{beta2:.2f}" if beta2 < 1.0 else str(beta2)
        run_name = f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"

        geo = geometry_cache.get(run_name)
        if geo is None:
            continue

        # Find window closest to ckpt_step
        windows = geo.get("windows", [])
        best_win = None
        best_dist = float("inf")
        for w in windows:
            d = abs(w["step"] - ckpt_step)
            if d < best_dist:
                best_dist = d
                best_win = w

        if best_win is None or best_dist > 400:
            continue

        features = [best_win.get(f) for f in feature_names]
        if any(f is None for f in features):
            continue

        X_rows.append(features)
        G_vals.append(G)
        D_vals.append(D)
        labels.append(f"β2={beta2} ckpt={ckpt_step} lr={rh['lr']} λ={rh['lam']}")

    if len(X_rows) < 3:
        print(f"  Only {len(X_rows)} valid data points. Need >= 3 for regression.")
        return

    X = np.array(X_rows)
    G = np.array(G_vals)
    D = np.array(D_vals)

    print(f"\n  Fitting linear model with {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Features: {feature_names}")

    results = {}

    for target_name, target in [("G", G), ("D", D)]:
        # Standardize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-12] = 1.0
        X_norm = (X - X_mean) / X_std

        # Add intercept
        X_aug = np.column_stack([np.ones(X_norm.shape[0]), X_norm])

        # OLS fit
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_aug, target, rcond=None)
        except np.linalg.LinAlgError:
            print(f"  [ERROR] Least squares failed for {target_name}")
            continue

        y_pred = X_aug @ beta
        ss_res = np.sum((target - y_pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        print(f"\n  {target_name} model:")
        print(f"    R² = {r2:.4f}")
        print(f"    Intercept = {beta[0]:.6f}")
        for i, fname in enumerate(feature_names):
            print(f"    {fname:>15s}: β = {beta[i+1]:+.6f}")

        results[target_name] = {
            "R2": r2,
            "intercept": float(beta[0]),
            "coefficients": {fname: float(beta[i + 1]) for i, fname in enumerate(feature_names)},
            "n_samples": X.shape[0],
            "feature_means": {fname: float(X_mean[i]) for i, fname in enumerate(feature_names)},
            "feature_stds": {fname: float(X_std[i]) for i, fname in enumerate(feature_names)},
        }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "reheat_predictability.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Scatter plot: predicted vs actual
    if HAS_MPL and results:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        fig.suptitle("Reheating Predictability", fontsize=13, fontweight="bold")

        for ax, (target_name, target) in zip(axes, [("G", G), ("D", D)]):
            if target_name not in results:
                continue
            res = results[target_name]
            X_norm = (X - np.array([res["feature_means"][f] for f in feature_names])) / \
                     np.array([res["feature_stds"][f] for f in feature_names])
            X_aug = np.column_stack([np.ones(X_norm.shape[0]), X_norm])
            beta_vec = np.array([res["intercept"]] + [res["coefficients"][f] for f in feature_names])
            y_pred = X_aug @ beta_vec

            ax.scatter(target, y_pred, s=40, alpha=0.7)
            lims = [min(target.min(), y_pred.min()), max(target.max(), y_pred.max())]
            ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
            ax.set_xlabel(f"Actual {target_name}")
            ax.set_ylabel(f"Predicted {target_name}")
            ax.set_title(f"{target_name}: R²={res['R2']:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = RESULTS_DIR / "reheat_predictability_scatter.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Geometry-LM Correlation + Predictability")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["2", "4", "all"],
                        help="Which phase to run (2, 4, or all)")
    parser.add_argument("--base-dir", type=str, default="runs/beta2_ablation",
                        help="Base directory with β2 runs")
    parser.add_argument("--beta2s", type=str, default="0.99,0.95,0.80",
                        help="Comma-separated β2 values to include")
    args = parser.parse_args()

    beta2_list = [float(x) for x in args.beta2s.split(",")]
    base_dir = Path(args.base_dir)

    # Discover run directories
    run_dirs = []
    for b2 in beta2_list:
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        d = base_dir / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
        run_dirs.append(d)

    if args.phase in ("2", "all"):
        run_phase2(run_dirs)

    if args.phase in ("4", "all"):
        run_phase4(run_dirs)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
