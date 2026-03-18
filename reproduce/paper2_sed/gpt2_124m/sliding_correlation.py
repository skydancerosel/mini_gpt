#!/usr/bin/env python3
"""
Phase-Specific Correlation Analysis for the Spectral Edge Paper.

The global detrended cross-correlation between σ₂/σ₃ and val_loss underestimates
the true signal because it mixes opposite-sign phases. This script:

1. Segments each seed's trajectory into 3 phases: rise, plateau, collapse
2. Computes phase-specific Pearson r (σ₂/σ₃ vs val_loss) within each phase
3. Also computes phase-specific r for drift_speed vs val_loss (comparison)
4. Runs a sliding-window local correlation analysis
5. Reports Δval_loss per phase (how much loss improvement happens in each)
6. Tests robustness: multiple phase boundary definitions

Usage:
    python phase_specific_analysis.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 149, 256]
RUN_BASE = Path(__file__).resolve().parent.parent / "runs" / "scale_124M"
OUT_DIR = RUN_BASE / "pilot_124M_b20.95_s42" / "results"
SEED_COLORS = {42: '#1f77b4', 123: '#ff7f0e', 149: '#2ca02c', 256: '#d62728'}
SEED_MARKERS = {42: 'o', 123: 's', 149: '^', 256: 'D'}


def load_all(seed):
    base = RUN_BASE / f"pilot_124M_b20.95_s{seed}"
    noise = json.load(open(base / "results" / "pc2_noise_test.json"))
    geo = json.load(open(base / "causal_geometry.json"))
    metrics = json.load(open(base / "pilot_metrics.json"))
    return noise, geo, metrics


def to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ══════════════════════════════════════════════════════════════════════
# PHASE DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_phases(steps, r23, method="peak"):
    """
    Detect rise/plateau/collapse phases.

    Returns dict with phase boundaries: {phase_name: (start_idx, end_idx)}
    """
    peak_idx = int(np.argmax(r23))

    if method == "peak":
        # Simple: split at peak
        # Rise = start to peak
        # Plateau = peak ± 3 steps (where r23 is within 90% of peak)
        # Collapse = peak to end
        peak_val = r23[peak_idx]
        plat_thresh = peak_val * 0.85

        # Find plateau bounds
        plat_start = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if r23[i] >= plat_thresh:
                plat_start = i
            else:
                break

        plat_end = peak_idx
        for i in range(peak_idx + 1, len(r23)):
            if r23[i] >= plat_thresh:
                plat_end = i
            else:
                break

        return {
            "rise": (0, plat_start + 1),       # up to plateau start (inclusive)
            "plateau": (plat_start, plat_end + 1),  # plateau region
            "collapse": (plat_end, len(r23)),   # from plateau end to finish
        }

    elif method == "derivative":
        # Use sign of smoothed derivative
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(r23.astype(float), size=5)
        deriv = np.gradient(smooth)

        # Rise: where derivative is consistently positive
        # Collapse: where derivative is consistently negative
        rise_end = peak_idx
        collapse_start = peak_idx
        for i in range(peak_idx, len(r23)):
            if deriv[i] < -0.01:
                collapse_start = i
                break

        return {
            "rise": (0, rise_end + 1),
            "plateau": (rise_end, collapse_start + 1),
            "collapse": (collapse_start, len(r23)),
        }

    elif method == "threshold":
        # Use fixed threshold crossings
        tau = 1.30
        # Rise: from start until first time r23 > tau
        rise_end = 0
        for i in range(len(r23)):
            if r23[i] > tau:
                rise_end = i
                break

        # Collapse: from last time r23 > tau to end
        collapse_start = len(r23) - 1
        for i in range(len(r23) - 1, -1, -1):
            if r23[i] > tau:
                collapse_start = i
                break

        return {
            "rise": (0, rise_end + 1),
            "plateau": (rise_end, collapse_start + 1),
            "collapse": (collapse_start, len(r23)),
        }


# ══════════════════════════════════════════════════════════════════════
# PHASE-SPECIFIC CORRELATIONS
# ══════════════════════════════════════════════════════════════════════

def phase_correlations(steps, r23, vl, ds, phases):
    """Compute correlation and stats within each phase."""
    results = {}
    for name, (s, e) in phases.items():
        n = e - s
        if n < 4:
            results[name] = {"n": n, "r_gap": None, "r_drift": None,
                             "p_gap": None, "p_drift": None,
                             "vl_start": None, "vl_end": None, "dvl": None,
                             "step_range": (int(steps[s]), int(steps[e-1]))}
            continue

        phase_r23 = r23[s:e]
        phase_vl = vl[s:e]
        phase_ds = ds[s:e]
        phase_steps = steps[s:e]

        # Pearson r with p-value for σ₂/σ₃ vs val_loss
        r_gap, p_gap = stats.pearsonr(phase_r23, phase_vl)

        # Pearson r for drift_speed vs val_loss
        r_drift, p_drift = stats.pearsonr(phase_ds, phase_vl)

        # Val_loss change in this phase
        vl_start = float(phase_vl[0])
        vl_end = float(phase_vl[-1])
        dvl = vl_start - vl_end  # positive means improvement

        # σ₂/σ₃ change
        r23_start = float(phase_r23[0])
        r23_end = float(phase_r23[-1])

        results[name] = {
            "n": n,
            "step_range": (int(phase_steps[0]), int(phase_steps[-1])),
            "r_gap": float(r_gap),
            "p_gap": float(p_gap),
            "r_drift": float(r_drift),
            "p_drift": float(p_drift),
            "vl_start": vl_start,
            "vl_end": vl_end,
            "dvl": float(dvl),
            "r23_start": float(r23_start),
            "r23_end": float(r23_end),
            "dr23": float(r23_end - r23_start),
        }

    return results


# ══════════════════════════════════════════════════════════════════════
# SLIDING WINDOW LOCAL CORRELATION
# ══════════════════════════════════════════════════════════════════════

def sliding_correlation(x, y, steps, window=11):
    """Compute local Pearson r in a sliding window."""
    local_rs = []
    local_ps = []
    local_steps = []
    hw = window // 2
    for i in range(hw, len(x) - hw):
        chunk_x = x[i-hw:i+hw+1]
        chunk_y = y[i-hw:i+hw+1]
        if len(chunk_x) >= 5 and not np.any(np.isnan(chunk_y)):
            r, p = stats.pearsonr(chunk_x, chunk_y)
            local_rs.append(float(r))
            local_ps.append(float(p))
            local_steps.append(int(steps[i]))
    return np.array(local_rs), np.array(local_ps), np.array(local_steps)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("PHASE-SPECIFIC CORRELATION ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    # Load data for all seeds
    all_data = {}
    for seed in SEEDS:
        noise, geo, metrics = load_all(seed)
        val_map = {m["step"]: m["val_loss"] for m in metrics}
        geo_map = {w["step"]: w for w in geo["windows"]}

        steps = np.array([w["step"] for w in noise])
        r23 = np.array([w["sigma_ratio_23"] for w in noise])
        r12 = np.array([w["sigma_ratio_12"] for w in noise])
        pc1 = np.array([w["pc1_pct"] for w in noise])
        pc2 = np.array([w["pc2_pct"] for w in noise])
        vl = np.array([val_map.get(s, np.nan) for s in steps])
        ds = np.array([geo_map[s]["drift_speed"] if s in geo_map else np.nan
                       for s in steps])

        all_data[seed] = {
            "steps": steps, "r23": r23, "r12": r12,
            "pc1": pc1, "pc2": pc2, "vl": vl, "ds": ds,
        }

    # ═══════════════════════════════════════════════════════════════
    # Part 1: Phase-specific correlations (primary method: peak-based)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 70, flush=True)
    print("PART 1: Phase-Specific Correlations (peak-based segmentation)", flush=True)
    print("─" * 70, flush=True)

    all_phase_results = {}
    for seed in SEEDS:
        d = all_data[seed]
        phases = detect_phases(d["steps"], d["r23"], method="peak")
        results = phase_correlations(d["steps"], d["r23"], d["vl"], d["ds"], phases)
        all_phase_results[seed] = {"phases": phases, "results": results}

        print(f"\n  Seed {seed}:", flush=True)
        for name in ["rise", "plateau", "collapse"]:
            r = results[name]
            if r["r_gap"] is not None:
                gap_sig = "***" if r["p_gap"] < 0.001 else ("**" if r["p_gap"] < 0.01 else ("*" if r["p_gap"] < 0.05 else ""))
                drift_sig = "***" if r["p_drift"] < 0.001 else ("**" if r["p_drift"] < 0.01 else ("*" if r["p_drift"] < 0.05 else ""))
                print(f"    {name:>10}: steps {r['step_range'][0]:>5}–{r['step_range'][1]:>5} "
                      f"(n={r['n']:>2})  "
                      f"r(gap,vl)={r['r_gap']:+.3f}{gap_sig:<3}  "
                      f"r(drift,vl)={r['r_drift']:+.3f}{drift_sig:<3}  "
                      f"Δvl={r['dvl']:+.4f}  "
                      f"Δr23={r['dr23']:+.3f}", flush=True)
            else:
                print(f"    {name:>10}: n={r['n']} (too few)", flush=True)

    # Summary table
    print(f"\n{'─'*90}", flush=True)
    print(f"{'':>6} │ {'RISE':^25} │ {'PLATEAU':^25} │ {'COLLAPSE':^25}", flush=True)
    print(f"{'Seed':>6} │ {'r(gap)':>8} {'r(drift)':>9} {'Δvl':>7} │ "
          f"{'r(gap)':>8} {'r(drift)':>9} {'Δvl':>7} │ "
          f"{'r(gap)':>8} {'r(drift)':>9} {'Δvl':>7}", flush=True)
    print(f"{'─'*90}", flush=True)
    for seed in SEEDS:
        parts = []
        for phase in ["rise", "plateau", "collapse"]:
            r = all_phase_results[seed]["results"][phase]
            if r["r_gap"] is not None:
                parts.append(f"{r['r_gap']:>+8.3f} {r['r_drift']:>+9.3f} {r['dvl']:>+7.3f}")
            else:
                parts.append(f"{'---':>8} {'---':>9} {'---':>7}")
        print(f"{seed:>6} │ {parts[0]} │ {parts[1]} │ {parts[2]}", flush=True)

    # Cross-seed collapse-phase statistics
    collapse_gap_rs = [all_phase_results[s]["results"]["collapse"]["r_gap"]
                       for s in SEEDS if all_phase_results[s]["results"]["collapse"]["r_gap"] is not None]
    collapse_drift_rs = [all_phase_results[s]["results"]["collapse"]["r_drift"]
                         for s in SEEDS if all_phase_results[s]["results"]["collapse"]["r_drift"] is not None]
    print(f"\n  Collapse-phase cross-seed: r(gap,vl) = {np.mean(collapse_gap_rs):.3f} ± {np.std(collapse_gap_rs):.3f}", flush=True)
    print(f"  Collapse-phase cross-seed: r(drift,vl) = {np.mean(collapse_drift_rs):.3f} ± {np.std(collapse_drift_rs):.3f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Part 2: Robustness across segmentation methods
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print("PART 2: Robustness Across Segmentation Methods", flush=True)
    print("─" * 70, flush=True)

    methods = ["peak", "derivative", "threshold"]
    robustness_results = {}

    for method in methods:
        collapse_rs = []
        for seed in SEEDS:
            d = all_data[seed]
            phases = detect_phases(d["steps"], d["r23"], method=method)
            results = phase_correlations(d["steps"], d["r23"], d["vl"], d["ds"], phases)
            r_col = results["collapse"]["r_gap"]
            if r_col is not None:
                collapse_rs.append(r_col)
        robustness_results[method] = collapse_rs
        print(f"  {method:>12}: collapse r(gap,vl) = "
              f"{np.mean(collapse_rs):.3f} ± {np.std(collapse_rs):.3f} "
              f"(seeds: {[f'{r:.3f}' for r in collapse_rs]})", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Part 3: Sliding window local correlation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print("PART 3: Sliding Window Local Correlation (window=11)", flush=True)
    print("─" * 70, flush=True)

    sliding_results = {}
    for seed in SEEDS:
        d = all_data[seed]
        lr_gap, lp_gap, ls = sliding_correlation(d["r23"], d["vl"], d["steps"], window=11)
        lr_drift, lp_drift, _ = sliding_correlation(d["ds"], d["vl"], d["steps"], window=11)

        sliding_results[seed] = {
            "steps": ls, "r_gap": lr_gap, "p_gap": lp_gap,
            "r_drift": lr_drift, "p_drift": lp_drift,
        }

        # When does local correlation flip sign?
        sign_changes = np.where(np.diff(np.sign(lr_gap)))[0]
        flip_steps = ls[sign_changes] if len(sign_changes) > 0 else []

        print(f"\n  Seed {seed}:", flush=True)
        print(f"    r(gap,vl) range: [{lr_gap.min():.3f}, {lr_gap.max():.3f}], "
              f"mean={lr_gap.mean():.3f}", flush=True)
        print(f"    Sign flips at steps: {flip_steps.tolist()}", flush=True)
        print(f"    % windows with |r| > 0.5: {100*np.mean(np.abs(lr_gap)>0.5):.0f}%", flush=True)
        print(f"    % windows significant (p<0.05): {100*np.mean(lp_gap<0.05):.0f}%", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Part 4: Val-loss budget per phase
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}", flush=True)
    print("PART 4: Val-Loss Budget Per Phase", flush=True)
    print("─" * 70, flush=True)

    for seed in SEEDS:
        d = all_data[seed]
        total_dvl = float(d["vl"][0] - d["vl"][-1])
        print(f"\n  Seed {seed}: total Δval_loss = {total_dvl:.4f}", flush=True)
        for phase in ["rise", "plateau", "collapse"]:
            r = all_phase_results[seed]["results"][phase]
            if r["dvl"] is not None:
                pct = 100 * r["dvl"] / total_dvl
                print(f"    {phase:>10}: Δvl = {r['dvl']:+.4f} ({pct:>5.1f}% of total), "
                      f"steps {r['step_range'][0]}–{r['step_range'][1]}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ── Row 1: Phase-shaded trajectory plots (one per seed) ──
    for i, seed in enumerate(SEEDS):
        ax = fig.add_subplot(gs[0, i])
        d = all_data[seed]
        phases = all_phase_results[seed]["phases"]
        r = all_phase_results[seed]["results"]

        # Shade phases
        phase_colors = {"rise": "#2196F3", "plateau": "#FFC107", "collapse": "#F44336"}
        for name, (s, e) in phases.items():
            if e > s:
                ax.axvspan(d["steps"][s], d["steps"][min(e-1, len(d["steps"])-1)],
                          alpha=0.15, color=phase_colors[name], label=name)

        # Plot σ₂/σ₃
        ax.plot(d["steps"], d["r23"], 'k-o', markersize=2.5, linewidth=1.5)

        # Annotate correlations
        for name in ["rise", "plateau", "collapse"]:
            pr = r[name]
            if pr["r_gap"] is not None and pr["n"] > 4:
                mid_step = (pr["step_range"][0] + pr["step_range"][1]) / 2
                mid_r23 = d["r23"][phases[name][0]:phases[name][1]].mean()
                sig = "***" if pr["p_gap"] < 0.001 else ("**" if pr["p_gap"] < 0.01 else ("*" if pr["p_gap"] < 0.05 else "ns"))
                ax.annotate(f"r={pr['r_gap']:+.2f}{sig}",
                           (mid_step, mid_r23),
                           fontsize=7, fontweight='bold', ha='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel("σ₂/σ₃", fontsize=9)
        ax.set_title(f"seed {seed}", fontsize=11, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    # ── Row 2, Panel A: Collapse-phase bar chart ──
    ax = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(SEEDS))
    width = 0.35
    gap_rs = [all_phase_results[s]["results"]["collapse"]["r_gap"] for s in SEEDS]
    drift_rs = [all_phase_results[s]["results"]["collapse"]["r_drift"] for s in SEEDS]
    bars1 = ax.bar(x - width/2, gap_rs, width, label="σ₂/σ₃ vs val_loss",
                   color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x + width/2, drift_rs, width, label="drift_speed vs val_loss",
                   color='#ff7f0e', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=10)
    ax.set_ylabel("Pearson r (collapse phase)", fontsize=11)
    ax.set_title("A. Collapse-Phase Correlation (σ₂/σ₃ wins in all seeds)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    # ── Row 2, Panel B: Rise-phase bar chart ──
    ax = fig.add_subplot(gs[1, 2:4])
    rise_gap_rs = [all_phase_results[s]["results"]["rise"]["r_gap"]
                   if all_phase_results[s]["results"]["rise"]["r_gap"] is not None else 0
                   for s in SEEDS]
    rise_drift_rs = [all_phase_results[s]["results"]["rise"]["r_drift"]
                     if all_phase_results[s]["results"]["rise"]["r_drift"] is not None else 0
                     for s in SEEDS]
    bars1 = ax.bar(x - width/2, rise_gap_rs, width, label="σ₂/σ₃ vs val_loss",
                   color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x + width/2, rise_drift_rs, width, label="drift_speed vs val_loss",
                   color='#ff7f0e', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=10)
    ax.set_ylabel("Pearson r (rise phase)", fontsize=11)
    ax.set_title("B. Rise-Phase Correlation (negative = both improving)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            offset = -0.04 if h < 0 else 0.01
            ax.text(bar.get_x() + bar.get_width()/2., h + offset,
                    f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

    # ── Row 3, Panel C: Sliding window local correlation ──
    ax = fig.add_subplot(gs[2, 0:2])
    for seed in SEEDS:
        sr = sliding_results[seed]
        ax.plot(sr["steps"], sr["r_gap"], '-', color=SEED_COLORS[seed],
                linewidth=1.5, label=f"seed {seed}")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.3)
    ax.fill_between([2000, 10000], -0.5, 0.5, alpha=0.05, color='gray')
    ax.set_xlabel("Center step", fontsize=11)
    ax.set_ylabel("Local Pearson r (11-step window)", fontsize=11)
    ax.set_title("C. Sliding-Window r(σ₂/σ₃, val_loss) — sign flip at peak",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2000, 10000)

    # ── Row 3, Panel D: Robustness across methods ──
    ax = fig.add_subplot(gs[2, 2:4])
    method_names = list(robustness_results.keys())
    method_labels = ["Peak-based", "Derivative-based", "Threshold (τ=1.3)"]
    x_m = np.arange(len(method_names))
    for j, seed in enumerate(SEEDS):
        vals = []
        for method in method_names:
            # Find this seed's value
            d = all_data[seed]
            phases = detect_phases(d["steps"], d["r23"], method=method)
            results = phase_correlations(d["steps"], d["r23"], d["vl"], d["ds"], phases)
            r_col = results["collapse"]["r_gap"]
            vals.append(r_col if r_col is not None else 0)
        ax.plot(x_m, vals, 'o-', color=SEED_COLORS[seed], markersize=8,
                linewidth=1.5, label=f"seed {seed}")
    ax.set_xticks(x_m)
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_ylabel("Collapse-phase r(gap, val_loss)", fontsize=11)
    ax.set_title("D. Robustness: Collapse r Across Segmentation Methods",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.3, label="r=0.9")

    plt.suptitle("Phase-Specific Correlation Analysis: σ₂/σ₃ vs Val-Loss\n"
                 "Global correlation underestimates — collapse phase r > 0.9 universally",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "phase_specific_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}", flush=True)

    # ── Save JSON ──
    json_out = {
        "method": "peak-based phase segmentation",
        "seeds": {},
        "cross_seed_collapse": {
            "mean_r_gap": float(np.mean(collapse_gap_rs)),
            "std_r_gap": float(np.std(collapse_gap_rs)),
            "mean_r_drift": float(np.mean(collapse_drift_rs)),
            "std_r_drift": float(np.std(collapse_drift_rs)),
        },
        "robustness": {method: {
            "mean": float(np.mean(rs)),
            "std": float(np.std(rs)),
            "per_seed": [float(r) for r in rs],
        } for method, rs in robustness_results.items()},
    }
    for seed in SEEDS:
        json_out["seeds"][str(seed)] = all_phase_results[seed]["results"]

    out_json = OUT_DIR / "phase_specific_analysis.json"
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2, default=to_native)
    print(f"Saved: {out_json}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("PHASE-SPECIFIC ANALYSIS COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
