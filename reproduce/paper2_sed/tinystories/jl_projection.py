#!/usr/bin/env python3
"""
Three remaining analyses for the spectral edge dynamics paper:

1. Noise floor characterization across seeds (fast — cached JSON)
2. Gradient norm comparison: drift_speed vs σ₂/σ₃ (fast — cached JSON)
3. Random projection (JL) validation test (slow — checkpoint loading)

Usage:
    python spectral_remaining_analyses.py
"""

import json
import re
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch

# ── Paths ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 149, 256]
RUN_BASE = Path(__file__).resolve().parent.parent / "runs" / "scale_124M"
OUT_DIR = RUN_BASE / "pilot_124M_b20.95_s42" / "results"
P_TOTAL = 163_150_848
SEED_COLORS = {42: '#1f77b4', 123: '#ff7f0e', 149: '#2ca02c', 256: '#d62728'}


def load_noise_data(seed):
    """Load pc2_noise_test.json for a seed."""
    p = RUN_BASE / f"pilot_124M_b20.95_s{seed}" / "results" / "pc2_noise_test.json"
    return json.load(open(p))


def load_geo_data(seed):
    """Load causal_geometry.json for a seed."""
    p = RUN_BASE / f"pilot_124M_b20.95_s{seed}" / "causal_geometry.json"
    return json.load(open(p))


def load_metrics(seed):
    """Load pilot_metrics.json for a seed."""
    p = RUN_BASE / f"pilot_124M_b20.95_s{seed}" / "pilot_metrics.json"
    return json.load(open(p))


def detrend(x, steps):
    """Cubic detrend."""
    coeffs = np.polyfit(steps, x, 3)
    return x - np.polyval(coeffs, steps)


def cross_correlate_detrended(x, y, steps, max_lag=6):
    """Detrend both series and compute cross-correlation at multiple lags."""
    x_dt = detrend(np.array(x, dtype=float), np.array(steps, dtype=float))
    y_dt = detrend(np.array(y, dtype=float), np.array(steps, dtype=float))
    results = {}
    for lag in range(-3, max_lag):
        if lag >= 0:
            x_s = x_dt[:len(x_dt)-lag] if lag > 0 else x_dt
            y_s = y_dt[lag:] if lag > 0 else y_dt
        else:
            x_s = x_dt[-lag:]
            y_s = y_dt[:len(y_dt)+lag]
        if len(x_s) > 5:
            r = float(np.corrcoef(x_s, y_s)[0, 1])
            results[lag] = {"r": r, "n": len(x_s)}
    return results


def to_native(obj):
    """JSON serialization helper for numpy types."""
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
# ANALYSIS 1: NOISE FLOOR CHARACTERIZATION ACROSS SEEDS
# ══════════════════════════════════════════════════════════════════════

def noise_floor_cross_seed():
    """Characterize the noise floor across all 4 seeds."""
    print("=" * 80, flush=True)
    print("NOISE FLOOR CHARACTERIZATION ACROSS SEEDS", flush=True)
    print("=" * 80, flush=True)

    all_data = {}
    summary = {}

    for seed in SEEDS:
        noise = load_noise_data(seed)
        all_data[seed] = noise
        steps = [w["step"] for w in noise]
        r23 = [w["sigma_ratio_23"] for w in noise]
        r12 = [w["sigma_ratio_12"] for w in noise]
        pc2 = [w["pc2_pct"] for w in noise]
        null95 = [w["null_pc2_95"] for w in noise]
        pc2_above = [w["pc2_above_null95"] for w in noise]
        split_k2 = [w["split_half_k2"] for w in noise]

        # Find peak σ₂/σ₃
        peak_idx = int(np.argmax(r23))
        peak_r23 = r23[peak_idx]
        peak_step = steps[peak_idx]

        # Find collapse step (first time r23 < 1.20 after peak)
        collapse_step = None
        for i in range(peak_idx, len(r23)):
            if r23[i] < 1.20:
                collapse_step = steps[i]
                break

        # Find noise transition step (first time pc2 < null95)
        noise_step = None
        for i, (p2, n95) in enumerate(zip(pc2, null95)):
            if not pc2_above[i]:
                noise_step = steps[i]
                break

        # Noise floor stats: bottom eigenvalues (pc3 and below as fraction)
        # Coefficient of variation of pc2_pct across training
        pc2_arr = np.array(pc2)
        noise_cv = float(np.std(pc2_arr) / np.mean(pc2_arr))

        summary[seed] = {
            "peak_r23": float(peak_r23),
            "peak_step": int(peak_step),
            "collapse_step_tau120": int(collapse_step) if collapse_step else None,
            "noise_transition_step": int(noise_step) if noise_step else None,
            "mean_pc2_pct": float(np.mean(pc2)),
            "mean_null95": float(np.mean(null95)),
            "pc2_cv": float(noise_cv),
            "mean_split_k2": float(np.mean(split_k2)),
            "n_above_null95": int(sum(pc2_above)),
            "n_total": len(pc2_above),
        }

        print(f"\nSeed {seed}:", flush=True)
        print(f"  Peak σ₂/σ₃ = {peak_r23:.3f} at step {peak_step}", flush=True)
        print(f"  Collapse (τ=1.20): step {collapse_step}", flush=True)
        print(f"  Noise transition (PC2 < null95): step {noise_step}", flush=True)
        print(f"  Mean PC2% = {np.mean(pc2):.2f}, Mean null95 = {np.mean(null95):.2f}", flush=True)
        print(f"  PC2 above null95: {sum(pc2_above)}/{len(pc2_above)} windows", flush=True)
        print(f"  Mean split-half k2 = {np.mean(split_k2):.3f}", flush=True)

    # Cross-seed statistics
    peaks = [summary[s]["peak_r23"] for s in SEEDS]
    collapse_steps = [summary[s]["collapse_step_tau120"] for s in SEEDS
                      if summary[s]["collapse_step_tau120"] is not None]
    print(f"\n{'─'*60}", flush=True)
    print(f"Cross-seed: peak σ₂/σ₃ = {np.mean(peaks):.3f} ± {np.std(peaks):.3f}", flush=True)
    if collapse_steps:
        print(f"Cross-seed: collapse step = {np.mean(collapse_steps):.0f} ± {np.std(collapse_steps):.0f}", flush=True)

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: σ₂/σ₃ trajectories
    ax = axes[0, 0]
    for seed in SEEDS:
        d = all_data[seed]
        steps = [w["step"] for w in d]
        r23 = [w["sigma_ratio_23"] for w in d]
        ax.plot(steps, r23, 'o-', color=SEED_COLORS[seed], markersize=3,
                linewidth=1.5, label=f"seed {seed}")
    ax.axhline(1.20, color='gray', linestyle='--', alpha=0.5, label="τ=1.20")
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.3, label="r=1 (no gap)")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("σ₂/σ₃ ratio", fontsize=11)
    ax.set_title("A. Spectral Gap Ratio σ₂/σ₃ (4 seeds)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: σ₁/σ₂ trajectories
    ax = axes[0, 1]
    for seed in SEEDS:
        d = all_data[seed]
        steps = [w["step"] for w in d]
        r12 = [w["sigma_ratio_12"] for w in d]
        ax.plot(steps, r12, 'o-', color=SEED_COLORS[seed], markersize=3,
                linewidth=1.5, label=f"seed {seed}")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("σ₁/σ₂ ratio", fontsize=11)
    ax.set_title("B. Top Eigenvalue Ratio σ₁/σ₂ (4 seeds)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: PC2% vs null 95th percentile
    ax = axes[1, 0]
    for seed in SEEDS:
        d = all_data[seed]
        steps = [w["step"] for w in d]
        pc2 = [w["pc2_pct"] for w in d]
        null95 = [w["null_pc2_95"] for w in d]
        ax.plot(steps, pc2, 'o-', color=SEED_COLORS[seed], markersize=3,
                linewidth=1.5, label=f"PC2% seed {seed}")
        ax.plot(steps, null95, '--', color=SEED_COLORS[seed], alpha=0.5,
                linewidth=1)
    # Add a single legend entry for null
    ax.plot([], [], '--', color='gray', alpha=0.5, label="null 95th %ile")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Variance explained (%)", fontsize=11)
    ax.set_title("C. PC2% vs Noise Null (dashed=null 95th)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel D: Split-half reliability
    ax = axes[1, 1]
    for seed in SEEDS:
        d = all_data[seed]
        steps = [w["step"] for w in d]
        k2 = [w["split_half_k2"] for w in d]
        ax.plot(steps, k2, 'o-', color=SEED_COLORS[seed], markersize=3,
                linewidth=1.5, label=f"seed {seed}")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label="0.5 (weak)")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Split-half cos(angle) for PC2", fontsize=11)
    ax.set_title("D. PC2 Split-Half Stability (4 seeds)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Noise Floor Characterization Across Seeds (GPT-2 124M, W=10)",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "noise_floor_cross_seed.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}", flush=True)

    # Save JSON
    result = {
        "seeds": SEEDS,
        "summary": summary,
        "cross_seed": {
            "peak_r23_mean": float(np.mean(peaks)),
            "peak_r23_std": float(np.std(peaks)),
            "collapse_step_mean": float(np.mean(collapse_steps)) if collapse_steps else None,
            "collapse_step_std": float(np.std(collapse_steps)) if collapse_steps else None,
        }
    }
    out_json = OUT_DIR / "noise_floor_cross_seed.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=to_native)
    print(f"Saved: {out_json}", flush=True)

    return result


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: GRADIENT NORM COMPARISON
# ══════════════════════════════════════════════════════════════════════

def gradient_norm_comparison():
    """Compare drift_speed vs σ₂/σ₃ as val_loss predictors across seeds."""
    print("\n" + "=" * 80, flush=True)
    print("GRADIENT NORM COMPARISON: drift_speed vs σ₂/σ₃", flush=True)
    print("=" * 80, flush=True)

    all_results = {}

    for seed in SEEDS:
        noise = load_noise_data(seed)
        geo = load_geo_data(seed)
        metrics = load_metrics(seed)

        # Build val_loss map
        val_map = {m["step"]: m["val_loss"] for m in metrics}

        # σ₂/σ₃ data (from noise test)
        noise_steps = [w["step"] for w in noise]
        r23 = [w["sigma_ratio_23"] for w in noise]

        # drift_speed data (from causal_geometry)
        geo_windows = geo["windows"]
        geo_steps = [w["step"] for w in geo_windows]
        drift_speed = [w["drift_speed"] for w in geo_windows]

        # Match both with val_loss on common steps
        common_steps = sorted(set(noise_steps) & set(geo_steps) & set(val_map.keys()))

        r23_map = {w["step"]: w["sigma_ratio_23"] for w in noise}
        ds_map = {w["step"]: w["drift_speed"] for w in geo_windows}

        matched_r23 = [r23_map[s] for s in common_steps]
        matched_ds = [ds_map[s] for s in common_steps]
        matched_vl = [val_map[s] for s in common_steps]

        if len(common_steps) < 10:
            print(f"  Seed {seed}: too few common steps ({len(common_steps)}), skipping", flush=True)
            continue

        # Cross-correlations
        xcorr_r23 = cross_correlate_detrended(matched_r23, matched_vl, common_steps)
        xcorr_ds = cross_correlate_detrended(matched_ds, matched_vl, common_steps)

        # Peak lags
        peak_r23_lag = min(xcorr_r23.keys(), key=lambda k: xcorr_r23[k]["r"])
        peak_r23_r = xcorr_r23[peak_r23_lag]["r"]
        peak_ds_lag = min(xcorr_ds.keys(), key=lambda k: xcorr_ds[k]["r"])
        peak_ds_r = xcorr_ds[peak_ds_lag]["r"]

        all_results[seed] = {
            "n_windows": len(common_steps),
            "xcorr_r23": xcorr_r23,
            "xcorr_drift": xcorr_ds,
            "r23_peak_lag": peak_r23_lag,
            "r23_peak_r": peak_r23_r,
            "drift_peak_lag": peak_ds_lag,
            "drift_peak_r": peak_ds_r,
            "steps": common_steps,
            "r23": matched_r23,
            "drift_speed": matched_ds,
            "val_loss": matched_vl,
        }

        print(f"\n  Seed {seed} ({len(common_steps)} windows):", flush=True)
        print(f"    σ₂/σ₃  → peak lag={peak_r23_lag:+d}, r={peak_r23_r:.3f}", flush=True)
        print(f"    drift  → peak lag={peak_ds_lag:+d}, r={peak_ds_r:.3f}", flush=True)
        print(f"    Winner: {'σ₂/σ₃' if abs(peak_r23_r) > abs(peak_ds_r) else 'drift_speed'}", flush=True)

    # ── Summary table ──
    print(f"\n{'─'*60}", flush=True)
    print(f"{'Seed':>6}  {'σ₂/σ₃ lag':>10}  {'σ₂/σ₃ r':>10}  {'drift lag':>10}  {'drift r':>10}  {'Winner':>10}", flush=True)
    print(f"{'─'*60}", flush=True)
    for seed in SEEDS:
        if seed in all_results:
            r = all_results[seed]
            winner = "σ₂/σ₃" if abs(r["r23_peak_r"]) > abs(r["drift_peak_r"]) else "drift"
            print(f"{seed:>6}  {r['r23_peak_lag']:>+10d}  {r['r23_peak_r']:>10.3f}  "
                  f"{r['drift_peak_lag']:>+10d}  {r['drift_peak_r']:>10.3f}  {winner:>10}", flush=True)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Cross-correlation curves
    ax = axes[0]
    for seed in SEEDS:
        if seed not in all_results:
            continue
        r = all_results[seed]
        # σ₂/σ₃ (solid)
        lags_r23 = sorted(r["xcorr_r23"].keys())
        rs_r23 = [r["xcorr_r23"][l]["r"] for l in lags_r23]
        ax.plot(lags_r23, rs_r23, 'o-', color=SEED_COLORS[seed], markersize=5,
                linewidth=2, label=f"σ₂/σ₃ s{seed}")
        # drift_speed (dashed)
        lags_ds = sorted(r["xcorr_drift"].keys())
        rs_ds = [r["xcorr_drift"][l]["r"] for l in lags_ds]
        ax.plot(lags_ds, rs_ds, 's--', color=SEED_COLORS[seed], markersize=4,
                linewidth=1.5, alpha=0.7, label=f"drift s{seed}")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Lag (positive = predictor leads val_loss)", fontsize=11)
    ax.set_ylabel("Detrended Pearson r", fontsize=11)
    ax.set_title("A. Cross-correlation: σ₂/σ₃ (solid) vs drift (dashed)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Bar chart comparing peak |r|
    ax = axes[1]
    seeds_with_data = [s for s in SEEDS if s in all_results]
    x = np.arange(len(seeds_with_data))
    width = 0.35
    r23_peaks = [abs(all_results[s]["r23_peak_r"]) for s in seeds_with_data]
    ds_peaks = [abs(all_results[s]["drift_peak_r"]) for s in seeds_with_data]
    bars1 = ax.bar(x - width/2, r23_peaks, width, label="σ₂/σ₃",
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, ds_peaks, width, label="drift_speed",
                   color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds_with_data], fontsize=10)
    ax.set_ylabel("Peak |r| (detrended)", fontsize=11)
    ax.set_title("B. Predictive Power: |peak r| vs val_loss", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel C: Time series overlay for seed 42
    ax = axes[2]
    if 42 in all_results:
        r = all_results[42]
        steps = r["steps"]
        ax2 = ax.twinx()
        ax.plot(steps, r["r23"], 'b-o', markersize=3, linewidth=1.5, label="σ₂/σ₃")
        ax.plot(steps, np.array(r["drift_speed"]) / max(r["drift_speed"]) * max(r["r23"]),
                'g--s', markersize=3, linewidth=1.5, alpha=0.7,
                label="drift_speed (scaled)")
        ax2.plot(steps, r["val_loss"], 'r-', linewidth=2, alpha=0.5, label="val_loss")
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_ylabel("σ₂/σ₃ / scaled drift_speed", fontsize=11, color='blue')
        ax2.set_ylabel("val_loss", fontsize=11, color='red')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
    ax.set_title("C. Time Series: seed 42", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle("Gradient Norm Proxy (drift_speed) vs Spectral Gap (σ₂/σ₃) as Val-Loss Predictors",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "grad_norm_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}", flush=True)

    # Save JSON
    json_results = {}
    for seed, r in all_results.items():
        json_results[str(seed)] = {
            "n_windows": r["n_windows"],
            "r23_peak_lag": r["r23_peak_lag"],
            "r23_peak_r": r["r23_peak_r"],
            "drift_peak_lag": r["drift_peak_lag"],
            "drift_peak_r": r["drift_peak_r"],
            "xcorr_r23": r["xcorr_r23"],
            "xcorr_drift": r["xcorr_drift"],
        }
    out_json = OUT_DIR / "grad_norm_comparison.json"
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2, default=to_native)
    print(f"Saved: {out_json}", flush=True)

    return all_results


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: RANDOM PROJECTION (JL) VALIDATION
# ══════════════════════════════════════════════════════════════════════

def discover_checkpoints(run_dir):
    ckpts = []
    for p in Path(run_dir).glob("ckpt_*.pt"):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def jl_projection_test(seed=42, W=10):
    """
    Test JL projection invariance of spectral gap.

    Method: streaming projection. For each projection dimension d, generate
    random vectors φ_j ∈ ℝ^p one at a time (seeded), project each delta,
    then compute the Gram matrix in projected space.
    """
    print("\n" + "=" * 80, flush=True)
    print(f"JL PROJECTION TEST (seed {seed}, W={W})", flush=True)
    print("=" * 80, flush=True)

    run_dir = RUN_BASE / f"pilot_124M_b20.95_s{seed}"
    ckpts = discover_checkpoints(run_dir)

    # Step 1: Load full-dim spectrum (ground truth)
    spectrum_file = run_dir / "causal_geometry_spectrum.json"
    full_spectrum = json.load(open(spectrum_file))
    print(f"  Loaded full spectrum: {len(full_spectrum)} windows", flush=True)

    # Build ground truth maps
    gt_map = {}
    for w in full_spectrum:
        svs = w["singular_values"]
        if len(svs) >= 3 and svs[1] > 1e-15 and svs[2] > 1e-15:
            gt_map[w["step"]] = {
                "sv": svs,
                "r12": svs[0] / svs[1],
                "r23": svs[1] / svs[2],
            }

    # Step 2: Load all checkpoints, compute deltas
    print(f"  Loading {len(ckpts)} checkpoints...", flush=True)
    all_deltas = []
    delta_steps = []
    prev_params = None
    for i, (step, ckpt_path) in enumerate(ckpts):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        parts = []
        for name in sorted(state["model_state_dict"].keys()):
            parts.append(state["model_state_dict"][name].cpu().float().numpy().ravel())
        params = np.concatenate(parts)
        del state
        if prev_params is not None:
            all_deltas.append(params - prev_params)
            delta_steps.append(step)
        prev_params = params
        if (i + 1) % 10 == 0:
            print(f"    loaded {i+1}/{len(ckpts)} checkpoints", flush=True)

    p_dim = len(all_deltas[0])
    N = len(all_deltas)
    print(f"  {N} deltas, p={p_dim:,}", flush=True)

    # Step 3: For each projection dim, streaming project and compute spectrum
    proj_dims = [50, 100, 200, 500, 1000]  # 5W to 100W
    n_proj_seeds = 3
    all_proj_results = {}

    for d in proj_dims:
        print(f"\n  d={d} (={d//W}W):", flush=True)
        seed_results = []

        for ps in range(n_proj_seeds):
            rng = np.random.RandomState(ps * 1000 + d)

            # Project all deltas: δ̃_i ∈ ℝ^d
            # Generate projection matrix row-by-row to save memory
            projected = np.zeros((N, d), dtype=np.float64)

            for j in range(d):
                # Generate one row of Φ (1/√d normalization)
                phi_j = rng.randn(p_dim).astype(np.float32) / np.sqrt(d)
                for i in range(N):
                    projected[i, j] = np.dot(all_deltas[i], phi_j)
                if (j + 1) % 200 == 0:
                    print(f"    proj seed {ps}: {j+1}/{d} dims", flush=True)

            # Compute spectrum for each window
            windows = []
            for idx in range(W - 1, N):
                start = idx - W + 1
                step = delta_steps[idx]
                X_proj = projected[start:start+W, :]  # W × d
                G = X_proj @ X_proj.T  # W × W
                evals, _ = np.linalg.eigh(G)
                evals = np.sort(evals)[::-1]
                evals = np.maximum(evals, 0.0)
                svs = np.sqrt(evals)

                if step in gt_map and svs[1] > 1e-15 and svs[2] > 1e-15:
                    gt = gt_map[step]
                    proj_r23 = float(svs[1] / svs[2])
                    proj_r12 = float(svs[0] / svs[1])
                    windows.append({
                        "step": step,
                        "gt_r23": gt["r23"],
                        "proj_r23": proj_r23,
                        "gt_r12": gt["r12"],
                        "proj_r12": proj_r12,
                        "rel_err_r23": abs(proj_r23 - gt["r23"]) / gt["r23"],
                        "rel_err_r12": abs(proj_r12 - gt["r12"]) / gt["r12"],
                    })

            seed_results.append(windows)
            mean_err_r23 = np.mean([w["rel_err_r23"] for w in windows])
            mean_err_r12 = np.mean([w["rel_err_r12"] for w in windows])
            print(f"    proj seed {ps}: {len(windows)} windows, "
                  f"mean rel err r23={mean_err_r23:.4f}, r12={mean_err_r12:.4f}", flush=True)

        all_proj_results[d] = seed_results

    # Step 4: Aggregate and plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Relative error vs projection dim
    ax = axes[0]
    d_vals = sorted(all_proj_results.keys())
    mean_errs_r23 = []
    std_errs_r23 = []
    mean_errs_r12 = []
    std_errs_r12 = []

    for d in d_vals:
        # Average across windows and projection seeds
        all_errs_r23 = []
        all_errs_r12 = []
        for seed_windows in all_proj_results[d]:
            all_errs_r23.extend([w["rel_err_r23"] for w in seed_windows])
            all_errs_r12.extend([w["rel_err_r12"] for w in seed_windows])
        mean_errs_r23.append(np.mean(all_errs_r23))
        std_errs_r23.append(np.std(all_errs_r23))
        mean_errs_r12.append(np.mean(all_errs_r12))
        std_errs_r12.append(np.std(all_errs_r12))

    ax.errorbar(d_vals, mean_errs_r23, yerr=std_errs_r23, fmt='o-', color='#1f77b4',
                markersize=8, linewidth=2, capsize=5, label="σ₂/σ₃ relative error")
    ax.errorbar(d_vals, mean_errs_r12, yerr=std_errs_r12, fmt='s-', color='#ff7f0e',
                markersize=8, linewidth=2, capsize=5, label="σ₁/σ₂ relative error")
    ax.axhline(0.10, color='red', linestyle='--', alpha=0.7, label="10% threshold")
    ax.set_xlabel("Projection dimension d", fontsize=11)
    ax.set_ylabel("Mean relative error", fontsize=11)
    ax.set_title("A. JL Projection Error vs Dimension", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Add d/W labels
    for d, err in zip(d_vals, mean_errs_r23):
        ax.annotate(f"{d//W}W", (d, err), textcoords="offset points",
                   xytext=(0, 12), fontsize=8, ha='center', color='gray')

    # Panel B: Projected vs true r23 scatter (best d)
    ax = axes[1]
    best_d = 200  # pick a mid-range d
    if best_d in all_proj_results:
        for ps, seed_windows in enumerate(all_proj_results[best_d]):
            gt_r23 = [w["gt_r23"] for w in seed_windows]
            proj_r23 = [w["proj_r23"] for w in seed_windows]
            ax.scatter(gt_r23, proj_r23, s=15, alpha=0.6,
                      label=f"proj seed {ps}" if ps == 0 else None)
        lim = [0.9, max(max(gt_r23), max(proj_r23)) * 1.05]
        ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=1, label="y=x (perfect)")
        ax.set_xlabel("True σ₂/σ₃", fontsize=11)
        ax.set_ylabel(f"Projected σ₂/σ₃ (d={best_d})", fontsize=11)
        ax.set_title(f"B. True vs Projected r₂₃ (d={best_d}={best_d//W}W)",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Panel C: Time series comparison at d=200
    ax = axes[2]
    if best_d in all_proj_results and len(all_proj_results[best_d]) > 0:
        seed_windows = all_proj_results[best_d][0]
        steps = [w["step"] for w in seed_windows]
        gt_r23 = [w["gt_r23"] for w in seed_windows]
        proj_r23 = [w["proj_r23"] for w in seed_windows]
        ax.plot(steps, gt_r23, 'b-o', markersize=4, linewidth=2, label="Full-dim σ₂/σ₃")
        ax.plot(steps, proj_r23, 'r--s', markersize=3, linewidth=1.5, alpha=0.7,
                label=f"Projected (d={best_d})")
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_ylabel("σ₂/σ₃ ratio", fontsize=11)
        ax.set_title(f"C. Full vs Projected Trajectory (d={best_d})",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("JL Random Projection Preserves Spectral Gap (GPT-2 124M, W=10)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "jl_projection_test.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}", flush=True)

    # Save JSON summary
    json_summary = {}
    for d in d_vals:
        all_errs_r23 = []
        all_errs_r12 = []
        for seed_windows in all_proj_results[d]:
            all_errs_r23.extend([w["rel_err_r23"] for w in seed_windows])
            all_errs_r12.extend([w["rel_err_r12"] for w in seed_windows])
        json_summary[str(d)] = {
            "d_over_W": d // W,
            "mean_rel_err_r23": float(np.mean(all_errs_r23)),
            "std_rel_err_r23": float(np.std(all_errs_r23)),
            "max_rel_err_r23": float(np.max(all_errs_r23)),
            "mean_rel_err_r12": float(np.mean(all_errs_r12)),
            "std_rel_err_r12": float(np.std(all_errs_r12)),
            "max_rel_err_r12": float(np.max(all_errs_r12)),
            "n_windows": len(all_errs_r23) // n_proj_seeds,
            "n_proj_seeds": n_proj_seeds,
        }

    out_json = OUT_DIR / "jl_projection_test.json"
    with open(out_json, "w") as f:
        json.dump(json_summary, f, indent=2, default=to_native)
    print(f"Saved: {out_json}", flush=True)

    # Print summary table
    print(f"\n{'─'*70}", flush=True)
    print(f"{'d':>6}  {'d/W':>5}  {'mean err r23':>14}  {'max err r23':>14}  {'< 10%?':>8}", flush=True)
    print(f"{'─'*70}", flush=True)
    for d in d_vals:
        s = json_summary[str(d)]
        ok = "✓" if s["mean_rel_err_r23"] < 0.10 else "✗"
        print(f"{d:>6}  {d//W:>5}  {s['mean_rel_err_r23']:>14.4f}  "
              f"{s['max_rel_err_r23']:>14.4f}  {ok:>8}", flush=True)

    return json_summary


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analysis 3 first (fast — cached JSON)
    noise_results = noise_floor_cross_seed()

    # Analysis 2 (fast — cached JSON)
    grad_results = gradient_norm_comparison()

    # Analysis 1 last (slow — checkpoint loading + projection)
    jl_results = jl_projection_test(seed=42, W=10)

    print("\n" + "=" * 80, flush=True)
    print("ALL THREE ANALYSES COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
