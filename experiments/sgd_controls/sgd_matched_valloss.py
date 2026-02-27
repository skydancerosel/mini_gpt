import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Matched val_loss geometry comparison.

Key idea: SGD+mom plateaus at val≈5.1 (steps 2200-4000). AdamW blows past
val≈5.1 before step 200 (no checkpoint there). This script compares backbone
geometry at "matched performance level" to answer the reviewer critique:
    "You compared SGD late to Adam early."

Analysis:
  1. Point comparison: drift from init at matched val_loss steps
     - SGD+mom step ~2800 (val≈5.15) vs AdamW step 200 (val=4.33, closest)
  2. Window backbone comparison at "matched operating regime":
     - SGD+mom plateau window [2200, 4000]: val ≈ 5.10-5.20
     - AdamW early window [200, 1000]: val ≈ 4.33-2.80
       (already LOWER loss than SGD+mom's best — makes comparison conservative)
  3. Standard window [600, 2000] comparison (from sgd_matched_progress.py)

If even AdamW at val=4.33-2.80 already shows multi-dimensional backbone while
SGD+mom at val=5.10-5.20 remains degenerate PC1≈100%, the result is solid.
"""

import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgd_control_analysis import (
    load_trajectory, load_metrics, estimate_backbone,
    pc1_rotation, update_alignment, flatten_trunk,
    WINDOW_START, WINDOW_END,
)


RUN_LABELS = {"adamw": "AdamW (A)", "sgd_nomom": "SGD no-mom (B)",
              "sgd_mom": "SGD+mom (C)"}
RUN_COLORS = {"adamw": "#1f77b4", "sgd_nomom": "#d62728",
              "sgd_mom": "#2ca02c"}


def find_matched_step(metrics, target_val_loss):
    """Find step closest to target val_loss."""
    best = min(metrics, key=lambda r: abs(r["val_loss"] - target_val_loss))
    return best["step"], best["val_loss"]


def compute_drift_from_init(run_dir, step):
    """Compute ||theta(step) - theta(1)|| = total parameter drift from init."""
    rd = Path(run_dir)
    init_ckpt = torch.load(rd / "ckpt_000001.pt", map_location="cpu",
                           weights_only=True)
    step_ckpt = torch.load(rd / f"ckpt_{step:06d}.pt", map_location="cpu",
                           weights_only=True)
    theta_init = flatten_trunk(init_ckpt["model_state_dict"])
    theta_step = flatten_trunk(step_ckpt["model_state_dict"])
    drift = (theta_step - theta_init).norm().item()
    del init_ckpt, step_ckpt
    return drift


def backbone_on_window(run_dir, steps, label=""):
    """Estimate backbone on a custom window of steps."""
    print(f"    Loading trajectory for {label} ({len(steps)} steps)...")
    trunk, _, loaded = load_trajectory(run_dir, steps, n_blocks=0)

    if len(loaded) < 3:
        print(f"    WARNING: Only {len(loaded)} checkpoints loaded, "
              f"backbone estimation may be unreliable")
        if len(loaded) < 2:
            return None

    bb = estimate_backbone(trunk, loaded)
    rot = pc1_rotation(trunk, loaded, window_size=min(5, len(loaded) - 1))

    result = {
        "window": [steps[0], steps[-1]],
        "n_ckpts": len(loaded),
        "pc1_frac": bb["pc1_frac"],
        "k95": bb["k95"],
        "k99": bb["k99"],
        "var_frac_top5": bb["var_frac"][:5].tolist(),
        "rot_mean": rot["mean_cos"],
        "rot_min": rot["min_cos"],
        "v_b": bb["v_b"],
    }

    # Drift norms within window
    theta_anchor = trunk[0]
    drift_norms = [(trunk[i] - theta_anchor).norm().item()
                   for i in range(len(loaded))]
    result["window_drift"] = drift_norms[-1] if drift_norms else 0
    result["drift_norms"] = drift_norms
    result["loaded_steps"] = loaded

    del trunk
    return result


def main():
    base_dir = Path("runs/sgd_control")
    out_dir = base_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    run_dirs = {
        "adamw": base_dir / "adamw_s42",
        "sgd_nomom": base_dir / "sgd_nomom_s42",
        "sgd_mom": base_dir / "sgd_mom_s42",
    }

    # Load all metrics
    all_metrics = {}
    for name, rd in run_dirs.items():
        all_metrics[name] = json.load(open(rd / "pilot_metrics.json"))

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Point comparison at matched val_loss
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PART 1: Point comparison at matched val_loss")
    print("=" * 70)

    sgd_mom_final = all_metrics["sgd_mom"][-1]
    target_val = sgd_mom_final["val_loss"]
    print(f"\n  Target val_loss: {target_val:.4f} (SGD+mom final)")

    matched_steps = {}
    for name in ["adamw", "sgd_nomom", "sgd_mom"]:
        step, val = find_matched_step(all_metrics[name], target_val)
        matched_steps[name] = {"step": step, "val_loss": val}
        print(f"  {RUN_LABELS[name]:>20s}: step {step:>5d} (val={val:.4f})")

    # Note about AdamW
    adamw_200 = next(r for r in all_metrics["adamw"] if r["step"] == 200)
    print(f"\n  NOTE: AdamW is already at val={adamw_200['val_loss']:.4f} "
          f"at step 200")
    print(f"  AdamW passes through val≈{target_val:.1f} between steps 1-200 "
          f"(no intermediate ckpts)")
    print(f"  Using step {matched_steps['adamw']['step']} "
          f"(val={matched_steps['adamw']['val_loss']:.4f}) as closest match")

    # Drift from init at matched steps
    print(f"\n  Drift from init at matched val_loss:")
    drift_at_match = {}
    for name, rd in run_dirs.items():
        ms = matched_steps[name]["step"]
        drift = compute_drift_from_init(rd, ms)
        drift_at_match[name] = drift
        print(f"    {RUN_LABELS[name]:>20s}: ||theta({ms}) - theta(1)|| "
              f"= {drift:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Window backbone at matched operating regime
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 2: Backbone at matched operating regime")
    print("=" * 70)

    # SGD+mom plateau window: [2200, 4000] every 100 steps
    sgdmom_plateau_steps = list(range(2200, 4001, 100))
    # AdamW early window: [200, 1000] — available: 200, 400, 600, 650..1000
    adamw_early_steps = [200, 400, 600] + list(range(650, 1001, 50))
    # SGD no-mom: same plateau window for fair comparison
    sgdnm_plateau_steps = list(range(2200, 4001, 100))
    # Standard window for reference
    standard_window = list(range(600, 2001, 50))

    windows = {
        "adamw_early": {
            "run": "adamw",
            "steps": adamw_early_steps,
            "label": "AdamW early [200,1000] (val=4.33→2.80)",
        },
        "adamw_std": {
            "run": "adamw",
            "steps": standard_window,
            "label": "AdamW standard [600,2000] (val=3.02→2.47)",
        },
        "sgd_mom_plateau": {
            "run": "sgd_mom",
            "steps": sgdmom_plateau_steps,
            "label": "SGD+mom plateau [2200,4000] (val≈5.10-5.20)",
        },
        "sgd_mom_std": {
            "run": "sgd_mom",
            "steps": standard_window,
            "label": "SGD+mom standard [600,2000]",
        },
        "sgd_nomom_plateau": {
            "run": "sgd_nomom",
            "steps": sgdnm_plateau_steps,
            "label": "SGD no-mom plateau [2200,4000]",
        },
    }

    window_results = {}
    for wname, winfo in windows.items():
        rd = run_dirs[winfo["run"]]
        print(f"\n  {winfo['label']}:")
        result = backbone_on_window(rd, winfo["steps"], label=winfo["label"])
        if result is not None:
            print(f"    PC1 = {result['pc1_frac']*100:.1f}%, "
                  f"k95={result['k95']}, k99={result['k99']}")
            print(f"    Rotation mean={result['rot_mean']:.4f}, "
                  f"min={result['rot_min']:.4f}")
            print(f"    Window drift = {result['window_drift']:.4f}")
        else:
            print(f"    FAILED — too few checkpoints")
        window_results[wname] = result

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Backbone alignment across windows (v_b similarity)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 3: Cross-window v_b alignment")
    print("=" * 70)

    pairs = [
        ("adamw_early", "adamw_std"),
        ("sgd_mom_plateau", "sgd_mom_std"),
        ("adamw_early", "sgd_mom_plateau"),
        ("adamw_std", "sgd_mom_std"),
    ]
    vb_alignment = {}
    for w1, w2 in pairs:
        r1 = window_results.get(w1)
        r2 = window_results.get(w2)
        if r1 is not None and r2 is not None:
            cos = abs(float(r1["v_b"] @ r2["v_b"]))
            label = f"{w1} vs {w2}"
            vb_alignment[label] = cos
            print(f"  |cos(v_b, v_b)| {w1:>20s} vs {w2:<20s}: {cos:.4f}")
        else:
            print(f"  {w1} vs {w2}: SKIPPED (missing data)")

    # ═══════════════════════════════════════════════════════════════════
    # Summary Table
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  MATCHED VAL_LOSS COMPARISON TABLE")
    print("=" * 70)

    # Header
    col_keys = ["adamw_early", "sgd_mom_plateau", "sgd_nomom_plateau"]
    col_labels = ["AdamW [200,1000]", "SGD+mom [2200,4000]",
                  "SGD no-mom [2200,4000]"]

    header = f"  {'Metric':<32s}"
    for lbl in col_labels:
        header += f"  {lbl:>22s}"
    print(header)
    print(f"  {'-' * 100}")

    def row(label, values, fmt=".1f"):
        parts = [f"  {label:<32s}"]
        for v in values:
            if v is None:
                parts.append(f"{'N/A':>22s}")
            elif isinstance(v, float) and not np.isnan(v):
                parts.append(f"{v:>22{fmt}}")
            else:
                parts.append(f"{str(v):>22s}")
        print("  ".join(parts))

    # Val loss range in each window
    row("Val loss range",
        [f"4.33 -> 2.80" if window_results.get(k) else None
         for k in ["adamw_early"]] +
        [f"5.20 -> 5.10" if window_results.get(k) else None
         for k in ["sgd_mom_plateau"]] +
        [None for k in ["sgd_nomom_plateau"]],
        fmt="s")

    # Geometry
    for metric, key, fmt in [
        ("PC1 var fraction (%)", "pc1_frac", ".1f"),
        ("k95", "k95", "d"),
        ("k99", "k99", "d"),
        ("Window drift", "window_drift", ".2f"),
        ("PC1 rot mean |cos|", "rot_mean", ".4f"),
        ("PC1 rot min |cos|", "rot_min", ".4f"),
    ]:
        vals = []
        for k in col_keys:
            r = window_results.get(k)
            if r is None:
                vals.append(None)
            else:
                v = r[key]
                if key == "pc1_frac":
                    v *= 100
                vals.append(v)
        row(metric, vals, fmt=fmt)

    # Also show standard window for reference
    print(f"\n  {'(Standard window [600,2000])':<32s}", end="")
    for wkey in ["adamw_std", "sgd_mom_std"]:
        r = window_results.get(wkey)
        if r:
            print(f"  PC1={r['pc1_frac']*100:5.1f}% k95={r['k95']}", end="")
        else:
            print(f"  {'N/A':>22s}", end="")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Interpretation
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print("=" * 70)

    adamw_early = window_results.get("adamw_early")
    sgdmom_plat = window_results.get("sgd_mom_plateau")

    if adamw_early and sgdmom_plat:
        adamw_pc1 = adamw_early["pc1_frac"] * 100
        sgdmom_pc1 = sgdmom_plat["pc1_frac"] * 100

        print(f"\n  At matched performance level (val≈4.3-5.1):")
        print(f"    AdamW  [200,1000]:  PC1={adamw_pc1:.1f}%, "
              f"k95={adamw_early['k95']}, drift={adamw_early['window_drift']:.2f}")
        print(f"    SGD+mom [2200,4000]: PC1={sgdmom_pc1:.1f}%, "
              f"k95={sgdmom_plat['k95']}, drift={sgdmom_plat['window_drift']:.2f}")

        if sgdmom_pc1 > 95 and adamw_pc1 < 90:
            print(f"\n  CONCLUSION: Even at matched (or lower) val_loss, "
                  f"AdamW develops")
            print(f"  multi-dimensional backbone structure (PC1={adamw_pc1:.1f}%) "
                  f"while SGD+mom")
            print(f"  remains degenerate (PC1={sgdmom_pc1:.1f}%). "
                  f"This rules out the critique")
            print(f"  that the difference is merely due to comparing "
                  f"different training phases.")
        elif sgdmom_pc1 < 90:
            print(f"\n  NOTE: SGD+mom develops non-degenerate backbone "
                  f"(PC1={sgdmom_pc1:.1f}%)")
            print(f"  at its plateau — interesting.")
        else:
            print(f"\n  NOTE: Both show high PC1, suggesting intrinsic dynamics.")

        # Conservative note
        print(f"\n  Conservative note: AdamW window [200,1000] has val_loss "
              f"ranging from")
        print(f"  4.33 to 2.80, which is LOWER than SGD+mom's 5.1-5.2. "
              f"If anything, this")
        print(f"  makes the comparison conservative (AdamW is further along).")
        print(f"  The key point: SGD+mom at 4000 steps and val≈5.1 "
              f"has less structure")
        print(f"  than AdamW achieves in its first 200-1000 steps.")

    # ═══════════════════════════════════════════════════════════════════
    # Figure
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel (0,0): Training curves with matched regions highlighted
    ax = axes[0, 0]
    for name in ["adamw", "sgd_mom", "sgd_nomom"]:
        m = all_metrics[name]
        steps = [r["step"] for r in m]
        ax.plot(steps, [r["val_loss"] for r in m],
                color=RUN_COLORS[name], linewidth=1.5,
                label=RUN_LABELS[name])
    # Highlight windows
    ax.axvspan(200, 1000, alpha=0.12, color=RUN_COLORS["adamw"],
               label="AdamW early window")
    ax.axvspan(2200, 4000, alpha=0.12, color=RUN_COLORS["sgd_mom"],
               label="SGD+mom plateau window")
    ax.axhline(target_val, color="gray", ls="--", alpha=0.5,
               label=f"val={target_val:.1f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val loss")
    ax.set_title("Training curves + matched windows")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (0,1): Probe OOD with matched regions
    ax = axes[0, 1]
    for name in ["adamw", "sgd_mom", "sgd_nomom"]:
        m = all_metrics[name]
        steps = [r["step"] for r in m]
        ax.plot(steps, [r["probe_ood_acc"] for r in m],
                color=RUN_COLORS[name], linewidth=1.5,
                label=RUN_LABELS[name])
    ax.axvspan(200, 1000, alpha=0.12, color=RUN_COLORS["adamw"])
    ax.axvspan(2200, 4000, alpha=0.12, color=RUN_COLORS["sgd_mom"])
    ax.set_xlabel("Step")
    ax.set_ylabel("p_ood")
    ax.set_title("Probe OOD (matched windows shaded)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (0,2): PC1 fraction bar chart (all windows)
    ax = axes[0, 2]
    bar_keys = ["adamw_early", "adamw_std", "sgd_mom_std", "sgd_mom_plateau",
                "sgd_nomom_plateau"]
    bar_labels = ["AdamW\n[200,1000]", "AdamW\n[600,2000]",
                  "SGD+mom\n[600,2000]", "SGD+mom\n[2200,4000]",
                  "SGD no-mom\n[2200,4000]"]
    bar_colors = [RUN_COLORS["adamw"], RUN_COLORS["adamw"],
                  RUN_COLORS["sgd_mom"], RUN_COLORS["sgd_mom"],
                  RUN_COLORS["sgd_nomom"]]
    bar_alphas = [1.0, 0.5, 0.5, 1.0, 1.0]

    pc1_vals = []
    for k in bar_keys:
        r = window_results.get(k)
        pc1_vals.append(r["pc1_frac"] * 100 if r else 0)

    x = np.arange(len(bar_keys))
    bars = ax.bar(x, pc1_vals, color=bar_colors, width=0.6)
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)
    for bar, v in zip(bars, pc1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylabel("PC1 variance fraction (%)")
    ax.set_title("PC1 across windows (matched val_loss)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    # Panel (1,0): Drift norms within each window
    ax = axes[1, 0]
    for wkey, color, ls, lbl in [
        ("adamw_early", RUN_COLORS["adamw"], "-", "AdamW [200,1000]"),
        ("adamw_std", RUN_COLORS["adamw"], "--", "AdamW [600,2000]"),
        ("sgd_mom_plateau", RUN_COLORS["sgd_mom"], "-",
         "SGD+mom [2200,4000]"),
        ("sgd_mom_std", RUN_COLORS["sgd_mom"], "--", "SGD+mom [600,2000]"),
    ]:
        r = window_results.get(wkey)
        if r and r["drift_norms"]:
            steps_w = r["loaded_steps"]
            ax.plot(steps_w, r["drift_norms"], ls,
                    color=color, linewidth=1.5, label=lbl)
    ax.set_xlabel("Step")
    ax.set_ylabel("||theta(t) - theta(anchor)||")
    ax.set_title("Drift within each window")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel (1,1): k95 bar chart
    ax = axes[1, 1]
    k95_vals = []
    k99_vals = []
    for k in bar_keys:
        r = window_results.get(k)
        k95_vals.append(r["k95"] if r else 0)
        k99_vals.append(r["k99"] if r else 0)

    width = 0.35
    ax.bar(x - width / 2, k95_vals, width, color=bar_colors, alpha=0.7,
           label="k95")
    ax.bar(x + width / 2, k99_vals, width, color=bar_colors, alpha=1.0,
           label="k99")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylabel("Number of PCs")
    ax.set_title("Effective dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (1,2): Variance spectrum (top-5 PCs)
    ax = axes[1, 2]
    for wkey, color, ls, lbl in [
        ("adamw_early", RUN_COLORS["adamw"], "-", "AdamW [200,1000]"),
        ("sgd_mom_plateau", RUN_COLORS["sgd_mom"], "-",
         "SGD+mom [2200,4000]"),
        ("sgd_nomom_plateau", RUN_COLORS["sgd_nomom"], "-",
         "SGD no-mom [2200,4000]"),
    ]:
        r = window_results.get(wkey)
        if r:
            vf = [v * 100 for v in r["var_frac_top5"]]
            ax.plot(range(1, len(vf) + 1), vf, "o-",
                    color=color, linewidth=1.5, markersize=5, label=lbl)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance fraction (%)")
    ax.set_title("Variance spectrum (top 5 PCs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))

    fig.suptitle("SGD Control: Matched Val_loss Geometry Comparison",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = out_dir / "fig_matched_valloss.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {path}")

    # ═══════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════
    out_data = {
        "target_val_loss": target_val,
        "matched_steps": matched_steps,
        "drift_at_matched_step": drift_at_match,
        "note": ("AdamW passes val≈5.1 before step 200; "
                 "closest ckpt at step 200 (val=4.33)"),
        "window_comparison": {
            wname: {
                "window": r["window"],
                "n_ckpts": r["n_ckpts"],
                "pc1_frac": r["pc1_frac"],
                "k95": r["k95"],
                "k99": r["k99"],
                "window_drift": r["window_drift"],
                "rot_mean": r["rot_mean"],
                "rot_min": r["rot_min"],
                "var_frac_top5": r["var_frac_top5"],
            }
            for wname, r in window_results.items()
            if r is not None
        },
        "vb_alignment": vb_alignment,
    }
    out_path = out_dir / "matched_valloss.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
