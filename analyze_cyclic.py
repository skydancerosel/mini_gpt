#!/usr/bin/env python3
"""
Analyze cyclic reheating vs baseline runs.

Loads baseline and cyclic pilot_metrics.json for each β2,
computes comparison metrics, generates plots and summary table.

Usage:
    python analyze_cyclic.py
    python analyze_cyclic.py --base-dir runs/beta2_ablation --cycle-tag cyclic_K800_H200
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_metrics(path):
    """Load pilot_metrics.json, return list of dicts."""
    with open(path) as f:
        return json.load(f)


def compute_summary(metrics):
    """Compute summary stats from a metrics list."""
    steps = [m["step"] for m in metrics]
    p_ood = [m["probe_ood_acc"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]

    # AUC via trapezoidal rule
    auc = float(np.trapz(p_ood, steps))

    return {
        "final_val_loss": val_loss[-1],
        "final_p_ood": p_ood[-1],
        "best_p_ood": max(p_ood),
        "best_p_ood_step": steps[int(np.argmax(p_ood))],
        "p_ood_auc": auc,
        "n_steps": steps[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze cyclic vs baseline")
    parser.add_argument("--base-dir", type=str, default="runs/beta2_ablation")
    parser.add_argument("--beta2s", type=str, default="0.99,0.95,0.90,0.80")
    parser.add_argument("--cycle-tag", type=str, default="cyclic_K800_H200",
                        help="Cyclic run subdirectory name")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: base-dir/summary/cyclic)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    beta2_list = [float(x) for x in args.beta2s.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "summary" / "cyclic"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for b2 in beta2_list:
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        run_dir = base_dir / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"

        baseline_path = run_dir / "pilot_metrics.json"
        cyclic_path = run_dir / args.cycle_tag / "pilot_metrics.json"

        if not baseline_path.exists():
            print(f"  [SKIP] β2={b2}: no baseline metrics")
            continue
        if not cyclic_path.exists():
            print(f"  [SKIP] β2={b2}: no cyclic metrics")
            continue

        baseline = load_metrics(baseline_path)
        cyclic = load_metrics(cyclic_path)

        bs = compute_summary(baseline)
        cs = compute_summary(cyclic)

        row = {
            "beta2": b2,
            "baseline_final_val_loss": bs["final_val_loss"],
            "cyclic_final_val_loss": cs["final_val_loss"],
            "delta_val_loss": cs["final_val_loss"] - bs["final_val_loss"],
            "baseline_final_p_ood": bs["final_p_ood"],
            "cyclic_final_p_ood": cs["final_p_ood"],
            "delta_final_p_ood": cs["final_p_ood"] - bs["final_p_ood"],
            "baseline_best_p_ood": bs["best_p_ood"],
            "cyclic_best_p_ood": cs["best_p_ood"],
            "delta_best_p_ood": cs["best_p_ood"] - bs["best_p_ood"],
            "baseline_p_ood_auc": bs["p_ood_auc"],
            "cyclic_p_ood_auc": cs["p_ood_auc"],
            "delta_p_ood_auc": cs["p_ood_auc"] - bs["p_ood_auc"],
        }
        results.append(row)

        print(f"  β2={b2}: Δ(final_p_ood)={row['delta_final_p_ood']:+.4f}, "
              f"Δ(val_loss)={row['delta_val_loss']:+.4f}")

        # Per-β2 plot
        if HAS_MPL:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            b_steps = [m["step"] for m in baseline]
            b_pood = [m["probe_ood_acc"] for m in baseline]
            c_steps = [m["step"] for m in cyclic]
            c_pood = [m["probe_ood_acc"] for m in cyclic]

            ax1.plot(b_steps, b_pood, "b-o", ms=3, label="baseline")
            ax1.plot(c_steps, c_pood, "r-s", ms=3, label="cyclic")

            # Shade hot windows
            is_hot = [m.get("is_hot", False) for m in cyclic]
            for i, (s, h) in enumerate(zip(c_steps, is_hot)):
                if h:
                    ax1.axvspan(max(0, s - 100), s + 100, alpha=0.1, color="red")

            ax1.set_xlabel("Step")
            ax1.set_ylabel("probe_ood_acc")
            ax1.set_title(f"β2={b2}: p_ood")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            b_vl = [m["val_loss"] for m in baseline]
            c_vl = [m["val_loss"] for m in cyclic]
            ax2.plot(b_steps, b_vl, "b-o", ms=3, label="baseline")
            ax2.plot(c_steps, c_vl, "r-s", ms=3, label="cyclic")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("val_loss")
            ax2.set_title(f"β2={b2}: val_loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(out_dir / f"cyclic_b2{b2_str}.png", dpi=150)
            plt.close()

    if not results:
        print("\n  No results to summarize.")
        return

    # Summary table
    print(f"\n{'='*80}")
    print(f"  CYCLIC REHEATING COMPARISON ({args.cycle_tag})")
    print(f"{'='*80}")
    print(f"  {'β2':>5s}  {'base_p_ood':>10s}  {'cyc_p_ood':>10s}  {'Δ_p_ood':>8s}  "
          f"{'base_vl':>8s}  {'cyc_vl':>8s}  {'Δ_vl':>8s}  {'verdict':>8s}")
    print(f"  {'-'*75}")

    for r in results:
        # Success: Δ(final_p_ood) >= +0.05 AND Δ(val_loss) <= +0.05
        ok_p = r["delta_final_p_ood"] >= 0.05
        ok_v = r["delta_val_loss"] <= 0.05
        verdict = "PASS" if (ok_p and ok_v) else "FAIL"

        print(f"  {r['beta2']:5.2f}  {r['baseline_final_p_ood']:10.4f}  "
              f"{r['cyclic_final_p_ood']:10.4f}  {r['delta_final_p_ood']:+8.4f}  "
              f"{r['baseline_final_val_loss']:8.4f}  {r['cyclic_final_val_loss']:8.4f}  "
              f"{r['delta_val_loss']:+8.4f}  {verdict:>8s}")

    # Save JSON
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    if results:
        keys = results[0].keys()
        with open(out_dir / "comparison.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"\n  Saved to {out_dir}/")

    # Overlay plot: all β2 on one figure
    if HAS_MPL and len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = {"0.99": "C0", "0.95": "C1", "0.90": "C2", "0.80": "C3"}

        for b2 in beta2_list:
            b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
            run_dir = base_dir / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
            bp = run_dir / "pilot_metrics.json"
            cp = run_dir / args.cycle_tag / "pilot_metrics.json"
            if not bp.exists() or not cp.exists():
                continue

            bm = load_metrics(bp)
            cm = load_metrics(cp)
            c = colors.get(b2_str, "gray")

            ax1.plot([m["step"] for m in bm], [m["probe_ood_acc"] for m in bm],
                     color=c, ls="--", alpha=0.5, label=f"β2={b2} base")
            ax1.plot([m["step"] for m in cm], [m["probe_ood_acc"] for m in cm],
                     color=c, ls="-", lw=2, label=f"β2={b2} cyclic")

            ax2.plot([m["step"] for m in bm], [m["val_loss"] for m in bm],
                     color=c, ls="--", alpha=0.5, label=f"β2={b2} base")
            ax2.plot([m["step"] for m in cm], [m["val_loss"] for m in cm],
                     color=c, ls="-", lw=2, label=f"β2={b2} cyclic")

        ax1.set_xlabel("Step")
        ax1.set_ylabel("probe_ood_acc")
        ax1.set_title("p_ood: Baseline (dashed) vs Cyclic (solid)")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Step")
        ax2.set_ylabel("val_loss")
        ax2.set_title("val_loss: Baseline (dashed) vs Cyclic (solid)")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "cyclic_overlay.png", dpi=150)
        plt.close()
        print(f"  Overlay plot: {out_dir}/cyclic_overlay.png")


if __name__ == "__main__":
    main()
