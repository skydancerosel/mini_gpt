#!/usr/bin/env python3
"""
Two analyses for the spectral edge dynamics paper:

1. EVENT STUDY: When does σ₂/σ₃ collapse, and when does val_loss stabilize?
   - For each seed, find the gap-collapse step (σ₂/σ₃ drops below threshold)
   - Find the val_loss stabilization step (|Δval_loss| drops below threshold)
   - Measure lead time Δt
   - Also: event-window plot showing spectral gap + val_loss aligned at collapse

2. W SWEEP: Cross-correlation of σ₂/σ₃ vs val_loss for W=10,15,20,25
   - For each W, compute the detrended cross-correlation at multiple lags
   - Show the peak lag shifts from negative (val_loss leads) to positive (gap leads)
   - W=15,25 require recomputation from checkpoints

Usage:
    python event_study_and_wsweep.py
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch

# ── Paths ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 149]
RUN_BASE = Path(__file__).resolve().parent.parent / "runs" / "scale_124M"
OUT_DIR = RUN_BASE / "pilot_124M_b20.95_s42" / "results"
P_TOTAL = 163_150_848


def load_seed_data(seed):
    """Load pc2_noise_test (has σ ratios) and pilot_metrics for a seed."""
    base = RUN_BASE / f"pilot_124M_b20.95_s{seed}"
    noise = json.load(open(base / "results" / "pc2_noise_test.json"))
    metrics = json.load(open(base / "pilot_metrics.json"))
    geo = json.load(open(base / "causal_geometry.json"))
    return noise, metrics, geo


# ══════════════════════════════════════════════════════════════════════
# PART 1: EVENT STUDY
# ══════════════════════════════════════════════════════════════════════

def event_study():
    """
    For each seed:
    1. Find when σ₂/σ₃ drops below threshold τ (gap collapse)
    2. Find when val_loss rate of change drops below threshold (stabilization)
    3. Compute lead time
    """
    print("=" * 80)
    print("EVENT STUDY: Gap Collapse → Val-Loss Stabilization")
    print("=" * 80)

    # Thresholds to test for robustness
    gap_thresholds = [1.15, 1.20, 1.25, 1.30]

    all_seed_results = {}

    for seed in SEEDS:
        noise, metrics, geo = load_seed_data(seed)

        steps = [w["step"] for w in noise]
        r23 = [w["sigma_ratio_23"] for w in noise]
        r12 = [w["sigma_ratio_12"] for w in noise]
        pc1 = [w["pc1_pct"] for w in noise]
        pc2 = [w["pc2_pct"] for w in noise]

        val_map = {m["step"]: m["val_loss"] for m in metrics}
        val_loss = [val_map.get(s, None) for s in steps]

        # Val_loss rate of change (absolute, smoothed over 3 windows)
        dvl = []
        for i in range(len(val_loss) - 1):
            if val_loss[i] is not None and val_loss[i+1] is not None:
                dvl.append(abs(val_loss[i+1] - val_loss[i]))
            else:
                dvl.append(None)
        dvl.append(None)

        # Smoothed |Δval_loss| (3-window moving average)
        dvl_smooth = []
        for i in range(len(dvl)):
            vals = []
            for j in range(max(0, i-1), min(len(dvl), i+2)):
                if dvl[j] is not None:
                    vals.append(dvl[j])
            dvl_smooth.append(np.mean(vals) if vals else None)

        # For each gap threshold, find collapse step
        seed_result = {
            "steps": steps, "r23": r23, "r12": r12,
            "pc1": pc1, "pc2": pc2,
            "val_loss": val_loss, "dvl": dvl, "dvl_smooth": dvl_smooth,
            "events": {}
        }

        for tau in gap_thresholds:
            # Find LAST step where σ₂/σ₃ > τ (the gap collapses after this)
            collapse_step = None
            collapse_idx = None
            # Look for the peak in σ₂/σ₃, then find where it drops below τ after the peak
            peak_idx = np.argmax(r23)
            for i in range(peak_idx, len(r23)):
                if r23[i] < tau:
                    collapse_step = steps[i]
                    collapse_idx = i
                    break

            # Find val_loss stabilization: first step after collapse where
            # smoothed |Δval_loss| < median of early |Δval_loss| * 0.3
            early_dvl = [d for d in dvl_smooth[:10] if d is not None]
            dvl_threshold = np.median(early_dvl) * 0.3 if early_dvl else 0.01

            stab_step = None
            stab_idx = None
            if collapse_idx is not None:
                for i in range(collapse_idx, len(dvl_smooth)):
                    if dvl_smooth[i] is not None and dvl_smooth[i] < dvl_threshold:
                        stab_step = steps[i]
                        stab_idx = i
                        break

            lead_time = None
            if collapse_step is not None and stab_step is not None:
                lead_time = stab_step - collapse_step

            seed_result["events"][tau] = {
                "collapse_step": collapse_step,
                "collapse_idx": collapse_idx,
                "stab_step": stab_step,
                "stab_idx": stab_idx,
                "lead_time": lead_time,
                "dvl_threshold": float(dvl_threshold) if dvl_threshold else None,
                "r23_at_peak": float(r23[peak_idx]),
                "peak_step": steps[peak_idx],
            }

        all_seed_results[seed] = seed_result

        # Print summary
        print(f"\nSeed {seed}:")
        print(f"  σ₂/σ₃ peak: {r23[np.argmax(r23)]:.3f} at step {steps[np.argmax(r23)]}")
        print(f"  {'τ':>6}  {'collapse':>10}  {'stab':>10}  {'lead_Δt':>10}  {'dvl_thresh':>10}")
        for tau in gap_thresholds:
            e = seed_result["events"][tau]
            cs = str(e["collapse_step"]) if e["collapse_step"] else "---"
            ss = str(e["stab_step"]) if e["stab_step"] else "---"
            lt = str(e["lead_time"]) if e["lead_time"] is not None else "---"
            dt = f"{e['dvl_threshold']:.4f}" if e["dvl_threshold"] else "---"
            print(f"  {tau:>6.2f}  {cs:>10}  {ss:>10}  {lt:>10}  {dt:>10}")

    return all_seed_results


def plot_event_study(results):
    """Create the event study figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, hspace=0.4, wspace=0.35)

    tau_main = 1.20  # primary threshold for visualization

    for col, seed in enumerate(SEEDS):
        r = results[seed]
        steps = r["steps"]
        r23 = r["r23"]
        r12 = r["r12"]
        vl = r["val_loss"]
        dvl_s = r["dvl_smooth"]
        evt = r["events"][tau_main]

        # Panel A: σ₂/σ₃ and σ₁/σ₂ trajectories with collapse marker
        ax = fig.add_subplot(gs[0, col])
        ax.plot(steps, r23, 'b-o', markersize=3, linewidth=1.5, label="σ₂/σ₃")
        ax.plot(steps, r12, 'r-s', markersize=2, linewidth=1, alpha=0.5, label="σ₁/σ₂")
        ax.axhline(tau_main, color='orange', linestyle='--', linewidth=1.5,
                   label=f"τ = {tau_main}")
        if evt["collapse_step"]:
            ax.axvline(evt["collapse_step"], color='red', linestyle='-',
                      linewidth=2, alpha=0.7, label=f"collapse @ {evt['collapse_step']}")
        if evt["stab_step"]:
            ax.axvline(evt["stab_step"], color='green', linestyle='-',
                      linewidth=2, alpha=0.7, label=f"stab @ {evt['stab_step']}")
        ax.set_ylabel("SV ratio")
        ax.set_title(f"Seed {seed}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.15, 0.5, 'A. Spectral Gap', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', rotation=90, va='center')

        # Panel B: val_loss with markers
        ax = fig.add_subplot(gs[1, col])
        valid_vl = [(s, v) for s, v in zip(steps, vl) if v is not None]
        ax.plot([s for s, v in valid_vl], [v for s, v in valid_vl],
                'k-o', markersize=3, linewidth=1.5)
        if evt["collapse_step"]:
            ax.axvline(evt["collapse_step"], color='red', linestyle='-',
                      linewidth=2, alpha=0.7)
        if evt["stab_step"]:
            ax.axvline(evt["stab_step"], color='green', linestyle='-',
                      linewidth=2, alpha=0.7)
        # Shade the lead-time window
        if evt["collapse_step"] and evt["stab_step"]:
            ax.axvspan(evt["collapse_step"], evt["stab_step"],
                      alpha=0.15, color='orange',
                      label=f"Δt = {evt['lead_time']} steps")
            ax.legend(fontsize=9)
        ax.set_ylabel("Val loss")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.15, 0.5, 'B. Val Loss', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', rotation=90, va='center')

        # Panel C: |Δval_loss| (smoothed) with stabilization threshold
        ax = fig.add_subplot(gs[2, col])
        valid_dvl = [(s, d) for s, d in zip(steps, dvl_s) if d is not None]
        ax.plot([s for s, d in valid_dvl], [d for s, d in valid_dvl],
                'purple', linewidth=1.5, marker='o', markersize=3,
                label="|Δval_loss| (3-pt avg)")
        if evt["dvl_threshold"]:
            ax.axhline(evt["dvl_threshold"], color='green', linestyle='--',
                      linewidth=1.5, label=f"stab threshold = {evt['dvl_threshold']:.4f}")
        if evt["collapse_step"]:
            ax.axvline(evt["collapse_step"], color='red', linestyle='-',
                      linewidth=2, alpha=0.7)
        if evt["stab_step"]:
            ax.axvline(evt["stab_step"], color='green', linestyle='-',
                      linewidth=2, alpha=0.7)
        ax.set_ylabel("|Δval_loss|")
        ax.set_xlabel("Training step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.15, 0.5, 'C. Loss Rate', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', rotation=90, va='center')

    fig.suptitle(
        f"Event Study: Spectral Gap Collapse (σ₂/σ₃ < {tau_main}) → Val-Loss Stabilization\n"
        f"Red = collapse, Green = stabilization, Orange = lead time Δt",
        fontsize=14, fontweight='bold')

    out = OUT_DIR / "event_study.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # ── Robustness: lead time vs threshold ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    gap_thresholds = [1.15, 1.20, 1.25, 1.30]
    colors = {42: '#d62728', 123: '#1f77b4', 149: '#2ca02c'}

    for seed in SEEDS:
        lead_times = []
        for tau in gap_thresholds:
            lt = results[seed]["events"][tau]["lead_time"]
            lead_times.append(lt if lt is not None else np.nan)
        ax.plot(gap_thresholds, lead_times, 'o-', color=colors[seed],
                markersize=8, linewidth=2, label=f"Seed {seed}")

    ax.set_xlabel("Gap threshold τ (σ₂/σ₃ < τ triggers event)", fontsize=12)
    ax.set_ylabel("Lead time Δt (steps: collapse → stabilization)", fontsize=12)
    ax.set_title("Lead Time Robustness Across Thresholds and Seeds", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out2 = OUT_DIR / "event_study_robustness.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")


# ══════════════════════════════════════════════════════════════════════
# PART 2: W SWEEP
# ══════════════════════════════════════════════════════════════════════

def discover_checkpoints(run_dir):
    ckpts = []
    for p in Path(run_dir).glob("ckpt_*.pt"):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def compute_spectrum_for_W(ckpts, W):
    """Compute σ₁/σ₂ and σ₂/σ₃ for given window size W."""
    delta_buffer = deque(maxlen=W)
    prev_params = None
    windows = []

    for i, (step, ckpt_path) in enumerate(ckpts):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        parts = []
        for name in sorted(state["model_state_dict"].keys()):
            parts.append(state["model_state_dict"][name].cpu().float().numpy().ravel())
        params = np.concatenate(parts)
        del state

        if prev_params is not None:
            delta_buffer.append(params - prev_params)
        prev_params = params

        if len(delta_buffer) == W:
            deltas = list(delta_buffer)
            G = np.zeros((W, W), dtype=np.float64)
            for ii in range(W):
                for jj in range(ii, W):
                    G[ii, jj] = np.dot(deltas[ii].astype(np.float64),
                                        deltas[jj].astype(np.float64))
                    G[jj, ii] = G[ii, jj]
            evals, _ = np.linalg.eigh(G)
            evals = np.sort(evals)[::-1]
            evals = np.maximum(evals, 0.0)
            svs = np.sqrt(evals)

            if svs[1] > 1e-15 and svs[2] > 1e-15:
                windows.append({
                    "step": step,
                    "r12": float(svs[0] / svs[1]),
                    "r23": float(svs[1] / svs[2]),
                    "drift_speed": float(np.linalg.norm(
                        np.mean([d.astype(np.float64) for d in deltas], axis=0))),
                })

    return windows


def compute_spectrum_multi_W(ckpts, W_values):
    """Compute σ₁/σ₂ and σ₂/σ₃ for multiple W values in a SINGLE pass.

    Key optimization: precompute full N×N pairwise dot product matrix once,
    then extract W×W submatrices for each window. This avoids redundant
    163M-dim dot products.
    """
    # Step 1: Load all checkpoints, compute deltas
    all_deltas = []
    delta_steps = []
    prev_params = None

    print(f"    Loading {len(ckpts)} checkpoints (single pass for W={W_values})...", flush=True)
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
            print(f"      loaded {i+1}/{len(ckpts)} checkpoints", flush=True)

    N = len(all_deltas)
    print(f"    {N} deltas. Precomputing {N}×{N} dot product matrix...", flush=True)

    # Step 2: Precompute full pairwise dot product matrix
    # Also precompute drift norms for each window
    DOT = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        di = all_deltas[i].astype(np.float64)
        DOT[i, i] = np.dot(di, di)
        for j in range(i + 1, N):
            dj = all_deltas[j].astype(np.float64)
            DOT[i, j] = np.dot(di, dj)
            DOT[j, i] = DOT[i, j]
        if (i + 1) % 10 == 0:
            print(f"      dot products: row {i+1}/{N}", flush=True)

    print(f"    Dot product matrix done. Extracting spectra...", flush=True)

    # Step 3: For each W, extract submatrices and eigendecompose
    results = {}
    for W in W_values:
        windows = []
        for idx in range(W - 1, N):
            start = idx - W + 1
            step = delta_steps[idx]
            G = DOT[start:start+W, start:start+W].copy()
            evals, _ = np.linalg.eigh(G)
            evals = np.sort(evals)[::-1]
            evals = np.maximum(evals, 0.0)
            svs = np.sqrt(evals)

            if svs[1] > 1e-15 and svs[2] > 1e-15:
                # Drift speed: ||mean(deltas)||
                # = sqrt(sum_ij DOT[i,j]) / W
                mean_dot = DOT[start:start+W, start:start+W].sum() / (W * W)
                windows.append({
                    "step": step,
                    "r12": float(svs[0] / svs[1]),
                    "r23": float(svs[1] / svs[2]),
                    "drift_speed": float(np.sqrt(max(mean_dot, 0.0))),
                })
        results[W] = windows
        print(f"    W={W}: {len(windows)} windows", flush=True)

    # Free memory
    del all_deltas, DOT

    return results


def detrend(x, steps):
    """Cubic detrend."""
    coeffs = np.polyfit(steps, x, 3)
    return x - np.polyval(coeffs, steps)


def cross_correlate_detrended(x, y, steps, max_lag=6):
    """Detrend both series and compute cross-correlation at multiple lags."""
    x_dt = detrend(np.array(x), np.array(steps, dtype=float))
    y_dt = detrend(np.array(y), np.array(steps, dtype=float))
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


def w_sweep(seed=42):
    """Compute cross-correlation for W=10,15,20,25."""
    print("\n" + "=" * 80, flush=True)
    print(f"W SWEEP: Cross-correlation σ₂/σ₃ vs val_loss (seed {seed})", flush=True)
    print("=" * 80, flush=True)

    run_dir = RUN_BASE / f"pilot_124M_b20.95_s{seed}"
    ckpts = discover_checkpoints(run_dir)
    metrics = json.load(open(run_dir / "pilot_metrics.json"))
    val_map = {m["step"]: m["val_loss"] for m in metrics}

    W_values = [10, 15, 20, 25]
    recompute_Ws = [W for W in W_values if W != 10]

    # Load cached W=10
    noise_data = json.load(open(run_dir / "results" / "pc2_noise_test.json"))
    cached_w10 = [{"step": w["step"],
                   "r12": w["sigma_ratio_12"],
                   "r23": w["sigma_ratio_23"]}
                  for w in noise_data]
    print(f"\n  W=10: Using cached pc2_noise_test.json ({len(cached_w10)} windows)", flush=True)

    # Single pass over checkpoints for W=15,20,25
    multi_results = compute_spectrum_multi_W(ckpts, recompute_Ws)
    multi_results[10] = cached_w10

    all_results = {}

    for W in W_values:
        windows = multi_results[W]

        # Match with val_loss
        matched = []
        for w in windows:
            vl = val_map.get(w["step"])
            if vl is not None:
                matched.append({**w, "val_loss": vl})

        if len(matched) < 10:
            print(f"  W={W}: Too few matched points ({len(matched)}), skipping", flush=True)
            continue

        steps = [m["step"] for m in matched]
        r23 = [m["r23"] for m in matched]
        vl = [m["val_loss"] for m in matched]

        # Cross-correlation
        xcorr = cross_correlate_detrended(r23, vl, steps)

        # Also do drift_speed vs val_loss (for gradient norm comparison)
        if "drift_speed" in matched[0]:
            ds = [m["drift_speed"] for m in matched]
            xcorr_drift = cross_correlate_detrended(ds, vl, steps)
        else:
            xcorr_drift = None

        # Find peak lag
        peak_lag = min(xcorr.keys(), key=lambda k: xcorr[k]["r"])  # most negative r
        peak_r = xcorr[peak_lag]["r"]

        all_results[W] = {
            "n_windows": len(matched),
            "step_range": (steps[0], steps[-1]),
            "xcorr": xcorr,
            "xcorr_drift": xcorr_drift,
            "peak_lag": peak_lag,
            "peak_r": peak_r,
        }

        print(f"\n  W={W}: {len(matched)} windows, steps {steps[0]}..{steps[-1]}", flush=True)
        print(f"    Peak: lag={peak_lag}, r={peak_r:.3f}", flush=True)
        print(f"    All lags: ", end="", flush=True)
        for lag in sorted(xcorr.keys()):
            print(f"lag={lag:+d}:{xcorr[lag]['r']:+.3f} ", end="")
        print(flush=True)

    return all_results


def plot_w_sweep(results):
    """Plot W sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    W_values = sorted(results.keys())
    colors = {10: '#d62728', 15: '#ff7f0e', 20: '#1f77b4', 25: '#2ca02c'}

    # Panel A: Cross-correlation curves
    ax = axes[0]
    for W in W_values:
        xcorr = results[W]["xcorr"]
        lags = sorted(xcorr.keys())
        rs = [xcorr[l]["r"] for l in lags]
        ax.plot(lags, rs, 'o-', color=colors[W], markersize=6,
                linewidth=2, label=f"W={W} (n={results[W]['n_windows']})")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Lag (positive = σ₂/σ₃ leads val_loss)", fontsize=11)
    ax.set_ylabel("Detrended Pearson r", fontsize=11)
    ax.set_title("A. Cross-correlation: σ₂/σ₃ vs val_loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Peak lag vs W
    ax = axes[1]
    peak_lags = [results[W]["peak_lag"] for W in W_values]
    peak_rs = [results[W]["peak_r"] for W in W_values]
    ax.plot(W_values, peak_lags, 'ko-', markersize=10, linewidth=2)
    for W, pl, pr in zip(W_values, peak_lags, peak_rs):
        ax.annotate(f"r={pr:.2f}", (W, pl), textcoords="offset points",
                   xytext=(10, 5), fontsize=9, color=colors[W])
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label="lag=0 (concurrent)")
    ax.set_xlabel("Window size W", fontsize=11)
    ax.set_ylabel("Peak lag (positive = spectral gap leads)", fontsize=11)
    ax.set_title("B. Peak Lag Shifts with Window Size", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(W_values)

    # Panel C: Comparison σ₂/σ₃ vs drift_speed (gradient norm proxy)
    ax = axes[2]
    for W in W_values:
        xcorr_drift = results[W].get("xcorr_drift")
        if xcorr_drift:
            lags = sorted(xcorr_drift.keys())
            rs = [xcorr_drift[l]["r"] for l in lags]
            ax.plot(lags, rs, 's--', color=colors[W], markersize=5,
                    linewidth=1.5, alpha=0.7, label=f"drift_speed W={W}")
        xcorr_gap = results[W]["xcorr"]
        lags = sorted(xcorr_gap.keys())
        rs = [xcorr_gap[l]["r"] for l in lags]
        ax.plot(lags, rs, 'o-', color=colors[W], markersize=5,
                linewidth=2, label=f"σ₂/σ₃ W={W}")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Detrended Pearson r", fontsize=11)
    ax.set_title("C. σ₂/σ₃ vs drift_speed (grad norm proxy)", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "w_sweep_crosscorr.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Event study
    event_results = event_study()
    plot_event_study(event_results)

    # Part 2: W sweep (seed 42 only — has checkpoints for recomputation)
    wsweep_results = w_sweep(seed=42)
    plot_w_sweep(wsweep_results)

    # Save all results
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = OUT_DIR / "event_study_results.json"
    # Simplify event results for JSON
    event_json = {}
    for seed in SEEDS:
        event_json[str(seed)] = {
            "steps": event_results[seed]["steps"],
            "r23": event_results[seed]["r23"],
            "r12": event_results[seed]["r12"],
            "events": event_results[seed]["events"],
        }
    with open(out, "w") as f:
        json.dump(event_json, f, indent=2, default=to_native)
    print(f"\nSaved: {out}")

    out2 = OUT_DIR / "w_sweep_results.json"
    with open(out2, "w") as f:
        json.dump({str(k): {
            "n_windows": v["n_windows"],
            "step_range": v["step_range"],
            "xcorr": v["xcorr"],
            "peak_lag": v["peak_lag"],
            "peak_r": v["peak_r"],
        } for k, v in wsweep_results.items()}, f, indent=2, default=to_native)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
