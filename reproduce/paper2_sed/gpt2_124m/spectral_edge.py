#!/usr/bin/env python3
"""
GPT-2 124M Spectral Edge Dynamics: Global rolling-window SVD across distribution shift.

Applies the SED framework to the FineWeb → OWT fine-tuning experiment:
- 60 combined checkpoints (steps 13800-25600, every 200 steps)
- Distribution shift at step 17800 (FineWeb → OpenWebText)
- Val_loss: FineWeb metrics for steps 13800-17600, OWT eigenvalue data for 17800-25600

Analyses:
1. Global rolling SVD (W=10, W=20): σ₁/σ₂, σ₂/σ₃ trajectories
2. Cross-correlation of σ₂/σ₃ vs val_loss (detrended, multiple lags)
3. Shift event study: spectral behavior before/at/after distribution shift
4. Noise floor: MP prediction vs observed eigenvalue ratios
5. Phase-specific correlations: pre-shift / rapid-improvement / overfitting

Usage:
    python gpt2_spectral_edge.py
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

import torch

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
COMBINED_DIR = SCRIPT_DIR / "combined_pretrain_finetune"
FW_METRICS = SCRIPT_DIR.parent / "karpathy_llmc" / "runs" / "gpt2_fineweb10B" / "pilot_metrics.json"
EIG_JSON = COMBINED_DIR / "drift_top8_eigenvalues.json"
OUT_DIR = COMBINED_DIR / "sed_results"

SHIFT_STEP = 17800  # Distribution shift: FineWeb → OWT

# Keys to skip: attn.bias (causal mask), lm_head.weight (tied to wte.weight)
SKIP_PATTERNS = ["attn.bias", "lm_head.weight"]


def normalize_key(key):
    """Strip _orig_mod. prefix for consistent naming across checkpoints."""
    return key.replace("_orig_mod.", "")


def discover_checkpoints():
    """Find all combined checkpoints, return sorted [(step, path)]."""
    ckpts = []
    for f in sorted(COMBINED_DIR.glob("ckpt_*.pt")):
        step = int(f.stem.split("_")[1])
        ckpts.append((step, f))
    return ckpts


def build_val_loss_map():
    """Combine val_loss from FineWeb metrics and OWT eigenvalue data."""
    vl_map = {}

    # FineWeb metrics (steps 0-17600)
    with open(FW_METRICS) as f:
        for entry in json.load(f):
            vl_map[entry["step"]] = entry["val_loss"]

    # OWT eigenvalues (steps 17800-25600) — use layer_0 for val_loss
    with open(EIG_JSON) as f:
        eig_data = json.load(f)
    for entry in eig_data["layer_0"]:
        vl_map[entry["step"]] = entry["val_loss"]

    return vl_map


def load_flat_params(ckpt_path):
    """Load checkpoint, normalize keys, skip masks/tied weights, return flat vector."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]

    normalized = {}
    for k, v in sd.items():
        nk = normalize_key(k)
        if any(pat in nk for pat in SKIP_PATTERNS):
            continue
        normalized[nk] = v.float().numpy().ravel()

    # Concatenate in sorted key order for consistency
    parts = [normalized[k] for k in sorted(normalized.keys())]
    return np.concatenate(parts)


def load_deltas_and_dot_matrix(ckpts):
    """
    Load all checkpoints, compute consecutive deltas, precompute N×N DOT matrix.

    Returns:
        delta_steps: list of step numbers for each delta
        DOT: N×N dot product matrix (float64)
        P: dimensionality of each delta vector
    """
    all_deltas = []
    delta_steps = []
    prev_params = None

    print(f"  Loading {len(ckpts)} checkpoints...", flush=True)
    t0 = time.time()
    for i, (step, ckpt_path) in enumerate(ckpts):
        params = load_flat_params(ckpt_path)

        if prev_params is not None:
            delta = params - prev_params
            all_deltas.append(delta)
            delta_steps.append(step)
        prev_params = params

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    loaded {i+1}/{len(ckpts)} ({elapsed:.0f}s)", flush=True)

    N = len(all_deltas)
    P = len(all_deltas[0])
    print(f"  {N} deltas, P={P:,} dims", flush=True)

    # Precompute DOT matrix
    print(f"  Precomputing {N}x{N} DOT matrix...", flush=True)
    t0 = time.time()
    DOT = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        di = all_deltas[i].astype(np.float64)
        DOT[i, i] = np.dot(di, di)
        for j in range(i + 1, N):
            dj = all_deltas[j].astype(np.float64)
            DOT[i, j] = np.dot(di, dj)
            DOT[j, i] = DOT[i, j]
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    dot products: row {i+1}/{N} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  DOT matrix done in {elapsed:.0f}s", flush=True)

    # Free delta memory
    del all_deltas

    return delta_steps, DOT, P


def compute_spectra_multi_W(delta_steps, DOT, W_values):
    """
    Extract rolling-window spectra for multiple W values from precomputed DOT matrix.

    Returns: {W: [{"step", "r12", "r23", "sv1", "sv2", "sv3", "drift_speed", "pc1_frac"}, ...]}
    """
    N = DOT.shape[0]
    results = {}

    for W in W_values:
        windows = []
        for idx in range(W - 1, N):
            start = idx - W + 1
            step = delta_steps[idx]

            G = DOT[start:start+W, start:start+W].copy()
            evals, evecs = np.linalg.eigh(G)
            evals = np.sort(evals)[::-1]
            evals = np.maximum(evals, 0.0)
            svs = np.sqrt(evals)

            total_var = evals.sum()

            if svs[1] > 1e-15 and svs[2] > 1e-15:
                # Drift speed from DOT submatrix
                mean_dot = DOT[start:start+W, start:start+W].sum() / (W * W)
                windows.append({
                    "step": step,
                    "r12": float(svs[0] / svs[1]),
                    "r23": float(svs[1] / svs[2]),
                    "sv1": float(svs[0]),
                    "sv2": float(svs[1]),
                    "sv3": float(svs[2]),
                    "pc1_frac": float(evals[0] / total_var) if total_var > 0 else 0.0,
                    "pc2_frac": float(evals[1] / total_var) if total_var > 0 else 0.0,
                    "drift_speed": float(np.sqrt(max(mean_dot, 0.0))),
                })
        results[W] = windows
        print(f"  W={W}: {len(windows)} windows", flush=True)

    return results


def detrend(x, steps):
    """Cubic detrend."""
    coeffs = np.polyfit(steps, x, 3)
    return x - np.polyval(coeffs, steps)


def cross_correlate_detrended(x, y, steps, max_lag=8):
    """Detrend both series and compute cross-correlation at multiple lags."""
    x_dt = detrend(np.array(x), np.array(steps, dtype=float))
    y_dt = detrend(np.array(y), np.array(steps, dtype=float))
    results = {}
    for lag in range(-5, max_lag):
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


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Global Rolling SVD + Cross-Correlation
# ══════════════════════════════════════════════════════════════════════

def analysis_1_global_svd(spectra, vl_map):
    """
    Main analysis: σ₂/σ₃ and σ₁/σ₂ trajectories, cross-correlation with val_loss.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Global Rolling SVD + Cross-Correlation")
    print("=" * 80)

    results = {}
    W_values = sorted(spectra.keys())

    for W in W_values:
        windows = spectra[W]
        steps = [w["step"] for w in windows]
        r23 = [w["r23"] for w in windows]
        r12 = [w["r12"] for w in windows]
        drift = [w["drift_speed"] for w in windows]
        pc1 = [w["pc1_frac"] for w in windows]

        # Match val_loss to window steps
        vl = [vl_map.get(s) for s in steps]
        valid = [(i, s, v) for i, (s, v) in enumerate(zip(steps, vl)) if v is not None]

        if len(valid) < 10:
            print(f"  W={W}: only {len(valid)} matched val_loss points, skipping")
            continue

        idx_v = [x[0] for x in valid]
        steps_v = [x[1] for x in valid]
        vl_v = [x[2] for x in valid]
        r23_v = [r23[i] for i in idx_v]
        r12_v = [r12[i] for i in idx_v]
        drift_v = [drift[i] for i in idx_v]

        # Cross-correlations
        cc_r23 = cross_correlate_detrended(r23_v, vl_v, steps_v)
        cc_r12 = cross_correlate_detrended(r12_v, vl_v, steps_v)
        cc_drift = cross_correlate_detrended(drift_v, vl_v, steps_v)

        # Find peak correlations
        peak_r23 = min(cc_r23.items(), key=lambda x: x[1]["r"])
        peak_r12 = min(cc_r12.items(), key=lambda x: x[1]["r"])
        peak_drift = min(cc_drift.items(), key=lambda x: x[1]["r"])

        results[W] = {
            "steps": steps,
            "r23": r23,
            "r12": r12,
            "drift_speed": drift,
            "pc1_frac": pc1,
            "val_loss": vl,
            "cc_r23": {str(k): v for k, v in cc_r23.items()},
            "cc_r12": {str(k): v for k, v in cc_r12.items()},
            "cc_drift": {str(k): v for k, v in cc_drift.items()},
            "peak_r23_lag": int(peak_r23[0]),
            "peak_r23_r": peak_r23[1]["r"],
            "peak_r12_lag": int(peak_r12[0]),
            "peak_r12_r": peak_r12[1]["r"],
            "peak_drift_lag": int(peak_drift[0]),
            "peak_drift_r": peak_drift[1]["r"],
        }

        print(f"\n  W={W}:")
        print(f"    σ₂/σ₃ vs val_loss: peak lag={peak_r23[0]}, r={peak_r23[1]['r']:.3f}")
        print(f"    σ₁/σ₂ vs val_loss: peak lag={peak_r12[0]}, r={peak_r12[1]['r']:.3f}")
        print(f"    drift  vs val_loss: peak lag={peak_drift[0]}, r={peak_drift[1]['r']:.3f}")

        # Shift-aligned statistics
        pre_shift = [w for w in windows if w["step"] < SHIFT_STEP]
        post_shift = [w for w in windows if w["step"] >= SHIFT_STEP]
        if pre_shift and post_shift:
            pre_r23_mean = np.mean([w["r23"] for w in pre_shift])
            post_r23_mean = np.mean([w["r23"] for w in post_shift])
            pre_r23_max = max(w["r23"] for w in pre_shift)
            post_r23_max = max(w["r23"] for w in post_shift)
            print(f"    Pre-shift  σ₂/σ₃: mean={pre_r23_mean:.3f}, max={pre_r23_max:.3f}")
            print(f"    Post-shift σ₂/σ₃: mean={post_r23_mean:.3f}, max={post_r23_max:.3f}")
            results[W]["pre_shift_r23_mean"] = pre_r23_mean
            results[W]["post_shift_r23_mean"] = post_r23_mean

    return results


def plot_analysis_1(results):
    """Plot global SVD trajectories and cross-correlations."""
    W_values = sorted(results.keys())
    n_W = len(W_values)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, n_W, hspace=0.4, wspace=0.3)

    for col, W in enumerate(W_values):
        r = results[W]
        steps = r["steps"]
        r23 = r["r23"]
        r12 = r["r12"]
        vl = r["val_loss"]
        drift = r["drift_speed"]

        # Panel A: σ₂/σ₃ and σ₁/σ₂ trajectories
        ax = fig.add_subplot(gs[0, col])
        ax.plot(steps, r23, "b-o", markersize=2, linewidth=1.5, label="σ₂/σ₃")
        ax.plot(steps, r12, "r-s", markersize=1.5, linewidth=1, alpha=0.5, label="σ₁/σ₂")
        ax.axvline(SHIFT_STEP, color="orange", linestyle="--", linewidth=2, alpha=0.7,
                   label=f"shift @ {SHIFT_STEP}")
        ax.set_ylabel("SV ratio")
        ax.set_title(f"W = {W}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.18, 0.5, "A. Spectral Ratios", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", rotation=90, va="center")

        # Panel B: Val loss
        ax = fig.add_subplot(gs[1, col])
        valid_vl = [(s, v) for s, v in zip(steps, vl) if v is not None]
        ax.plot([s for s, v in valid_vl], [v for s, v in valid_vl],
                "k-o", markersize=2, linewidth=1.5)
        ax.axvline(SHIFT_STEP, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_ylabel("Val loss")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.18, 0.5, "B. Val Loss", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", rotation=90, va="center")

        # Panel C: Drift speed
        ax = fig.add_subplot(gs[2, col])
        ax.plot(steps, drift, "g-o", markersize=2, linewidth=1.5)
        ax.axvline(SHIFT_STEP, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_ylabel("Drift speed")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.18, 0.5, "C. Drift Speed", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", rotation=90, va="center")

        # Panel D: Cross-correlation
        ax = fig.add_subplot(gs[3, col])
        cc_r23 = r["cc_r23"]
        cc_drift = r["cc_drift"]
        lags_r23 = sorted([int(k) for k in cc_r23.keys()])
        lags_drift = sorted([int(k) for k in cc_drift.keys()])
        ax.plot(lags_r23, [cc_r23[str(l)]["r"] for l in lags_r23],
                "b-o", markersize=4, linewidth=2, label="σ₂/σ₃")
        ax.plot(lags_drift, [cc_drift[str(l)]["r"] for l in lags_drift],
                "g--s", markersize=3, linewidth=1.5, label="drift")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        peak_lag = r["peak_r23_lag"]
        peak_r = r["peak_r23_r"]
        ax.annotate(f"lag={peak_lag}, r={peak_r:.3f}",
                    xy=(peak_lag, peak_r), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="blue"),
                    textcoords="offset points", xytext=(10, 10))
        ax.set_xlabel("Lag (window steps)")
        ax.set_ylabel("Cross-corr r")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.text(-0.18, 0.5, "D. Cross-Corr", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", rotation=90, va="center")

    fig.suptitle(
        "GPT-2 124M: Global Spectral Edge Dynamics Across Distribution Shift\n"
        f"FineWeb → OWT shift at step {SHIFT_STEP}",
        fontsize=14, fontweight="bold")

    out = OUT_DIR / "gpt2_global_svd.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Shift Event Study
# ══════════════════════════════════════════════════════════════════════

def analysis_2_shift_event(spectra, vl_map):
    """
    Detailed event study: how spectral gap ratio behaves before, at, and after
    the distribution shift.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Distribution Shift Event Study")
    print("=" * 80)

    # Use W=10 for highest temporal resolution
    W = 10
    windows = spectra[W]
    steps = np.array([w["step"] for w in windows])
    r23 = np.array([w["r23"] for w in windows])
    r12 = np.array([w["r12"] for w in windows])
    sv1 = np.array([w["sv1"] for w in windows])
    sv2 = np.array([w["sv2"] for w in windows])
    sv3 = np.array([w["sv3"] for w in windows])
    pc1 = np.array([w["pc1_frac"] for w in windows])
    pc2 = np.array([w["pc2_frac"] for w in windows])
    drift = np.array([w["drift_speed"] for w in windows])

    # Find the window closest to the shift
    shift_idx = np.argmin(np.abs(steps - SHIFT_STEP))
    print(f"  Shift step {SHIFT_STEP}, closest window step {steps[shift_idx]}")

    # Pre/post statistics
    pre_mask = steps < SHIFT_STEP
    post_mask = steps >= SHIFT_STEP
    # Further split post into improvement (17800-22000) and overfitting (22200+)
    overfit_step = 22200
    improve_mask = (steps >= SHIFT_STEP) & (steps < overfit_step)
    overfit_mask = steps >= overfit_step

    phases = {
        "pre_shift": pre_mask,
        "rapid_improve": improve_mask,
        "overfitting": overfit_mask,
    }

    results = {
        "shift_step": SHIFT_STEP,
        "W": W,
        "phases": {},
    }

    print(f"\n  Phase statistics (W={W}):")
    for phase_name, mask in phases.items():
        if mask.sum() == 0:
            continue
        r23_phase = r23[mask]
        r12_phase = r12[mask]
        drift_phase = drift[mask]
        pc1_phase = pc1[mask]

        stats_dict = {
            "n_windows": int(mask.sum()),
            "step_range": [int(steps[mask].min()), int(steps[mask].max())],
            "r23_mean": float(r23_phase.mean()),
            "r23_std": float(r23_phase.std()),
            "r23_max": float(r23_phase.max()),
            "r23_min": float(r23_phase.min()),
            "r12_mean": float(r12_phase.mean()),
            "drift_mean": float(drift_phase.mean()),
            "pc1_mean": float(pc1_phase.mean()),
        }
        results["phases"][phase_name] = stats_dict

        print(f"\n    {phase_name} ({stats_dict['n_windows']} windows, "
              f"steps {stats_dict['step_range'][0]}-{stats_dict['step_range'][1]}):")
        print(f"      σ₂/σ₃: {stats_dict['r23_mean']:.3f} ± {stats_dict['r23_std']:.3f} "
              f"(max={stats_dict['r23_max']:.3f})")
        print(f"      σ₁/σ₂: {stats_dict['r12_mean']:.3f}")
        print(f"      drift:  {stats_dict['drift_mean']:.4f}")
        print(f"      PC1%:   {stats_dict['pc1_mean']:.1%}")

    # Check Prediction #5: signal rank increases at distribution shift
    # σ₂/σ₃ spike = new signal dimension emerging
    pre_r23_mean = results["phases"]["pre_shift"]["r23_mean"]
    post_r23_max = results["phases"]["rapid_improve"]["r23_max"]
    ratio_change = post_r23_max / pre_r23_mean
    results["prediction5_ratio_change"] = float(ratio_change)
    print(f"\n  Prediction #5 test: post-shift σ₂/σ₃ max / pre-shift mean = {ratio_change:.2f}x")

    return results, steps, r23, r12, sv1, sv2, sv3, drift, pc1, pc2


def plot_analysis_2(results, steps, r23, r12, sv1, sv2, sv3, drift, pc1, pc2, vl_map):
    """Plot shift event study."""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, hspace=0.4, wspace=0.35)

    vl = np.array([vl_map.get(s) for s in steps])

    # Panel A: σ₂/σ₃ with shift marker and phase shading
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, r23, "b-o", markersize=3, linewidth=2, label="σ₂/σ₃")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8, label=f"shift @ {SHIFT_STEP}")
    ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6, label="overfit start")
    ax.axvspan(steps.min(), SHIFT_STEP, alpha=0.08, color="green", label="pre-shift")
    ax.axvspan(SHIFT_STEP, 22200, alpha=0.08, color="orange", label="rapid improve")
    ax.axvspan(22200, steps.max(), alpha=0.08, color="red", label="overfitting")
    ax.set_ylabel("σ₂/σ₃")
    ax.set_title("Spectral Gap Ratio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel B: σ₁/σ₂
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, r12, "r-o", markersize=3, linewidth=2, label="σ₁/σ₂")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("σ₁/σ₂")
    ax.set_title("Leading Ratio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: Val loss
    ax = fig.add_subplot(gs[0, 2])
    valid = [(s, v) for s, v in zip(steps, vl) if v is not None]
    ax.plot([s for s, v in valid], [v for s, v in valid],
            "k-o", markersize=3, linewidth=2, label="val_loss")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("Val loss")
    ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Raw singular values
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, sv1, "r-", linewidth=2, label="σ₁")
    ax.plot(steps, sv2, "b-", linewidth=2, label="σ₂")
    ax.plot(steps, sv3, "g-", linewidth=2, label="σ₃")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.set_ylabel("Singular value")
    ax.set_title("Raw Singular Values", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel E: PC1% and PC2%
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, np.array(pc1) * 100, "r-o", markersize=3, linewidth=2, label="PC1%")
    ax.plot(steps, np.array(pc2) * 100, "b-s", markersize=2, linewidth=1.5, label="PC2%")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("Variance fraction (%)")
    ax.set_title("PC Variance Fractions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel F: Drift speed
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps, drift, "g-o", markersize=3, linewidth=2, label="drift speed")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("||mean(Δ)||")
    ax.set_title("Drift Speed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel G: σ₂/σ₃ normalized by pre-shift baseline
    ax = fig.add_subplot(gs[2, 0])
    pre_mask = steps < SHIFT_STEP
    if pre_mask.sum() > 0:
        baseline = r23[pre_mask].mean()
        r23_norm = r23 / baseline
        ax.plot(steps, r23_norm, "b-o", markersize=3, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
        ax.set_ylabel("σ₂/σ₃ (normalized)")
        ax.set_title("Normalized σ₂/σ₃ (pre-shift = 1.0)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Panel H: Eigenvalue gaps (σ₁-σ₂ vs σ₂-σ₃)
    ax = fig.add_subplot(gs[2, 1])
    gap12 = sv1 - sv2
    gap23 = sv2 - sv3
    ax.plot(steps, gap12, "r-", linewidth=2, label="σ₁ − σ₂")
    ax.plot(steps, gap23, "b-", linewidth=2, label="σ₂ − σ₃")
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.set_ylabel("Gap")
    ax.set_xlabel("Training step")
    ax.set_title("Spectral Gaps", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel I: σ₂/σ₃ rate of change
    ax = fig.add_subplot(gs[2, 2])
    dr23 = np.diff(r23)
    step_mid = (steps[:-1] + steps[1:]) / 2
    ax.plot(step_mid, dr23, "b-o", markersize=3, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax.set_ylabel("Δ(σ₂/σ₃)")
    ax.set_xlabel("Training step")
    ax.set_title("Rate of Change", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"GPT-2 124M: Distribution Shift Event Study (W=10)\n"
        f"FineWeb (pre) → OWT (post) at step {SHIFT_STEP}",
        fontsize=14, fontweight="bold")

    out = OUT_DIR / "gpt2_shift_event.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Phase-Specific Correlations
# ══════════════════════════════════════════════════════════════════════

def analysis_3_phase_correlations(spectra, vl_map):
    """
    Compute Pearson r (with p-value) between spectral gap and val_loss
    separately for pre-shift, rapid-improvement, and overfitting phases.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Phase-Specific Correlations")
    print("=" * 80)

    W = 10
    windows = spectra[W]
    steps = np.array([w["step"] for w in windows])
    r23 = np.array([w["r23"] for w in windows])
    drift = np.array([w["drift_speed"] for w in windows])

    # Build aligned val_loss
    vl = np.array([vl_map.get(s, np.nan) for s in steps])
    valid = ~np.isnan(vl)

    # Define phases
    phases = {
        "pre_shift": (steps < SHIFT_STEP) & valid,
        "rapid_improve": (steps >= SHIFT_STEP) & (steps < 22200) & valid,
        "overfitting": (steps >= 22200) & valid,
        "all_post": (steps >= SHIFT_STEP) & valid,
        "full": valid,
    }

    results = {}
    print(f"\n  Phase-specific Pearson r (W={W}):")
    print(f"  {'Phase':<20} {'N':>4} {'r(gap,vl)':>10} {'p(gap)':>10} "
          f"{'r(drift,vl)':>12} {'p(drift)':>10}")
    print(f"  {'-'*70}")

    for phase_name, mask in phases.items():
        n = mask.sum()
        if n < 4:
            print(f"  {phase_name:<20} {n:>4}  (too few points)")
            continue

        r_gap, p_gap = stats.pearsonr(r23[mask], vl[mask])
        r_drift, p_drift = stats.pearsonr(drift[mask], vl[mask])

        results[phase_name] = {
            "n": int(n),
            "r_gap_vl": float(r_gap),
            "p_gap_vl": float(p_gap),
            "r_drift_vl": float(r_drift),
            "p_drift_vl": float(p_drift),
        }

        print(f"  {phase_name:<20} {n:>4} {r_gap:>10.3f} {p_gap:>10.2e} "
              f"{r_drift:>12.3f} {p_drift:>10.2e}")

    # Sliding window correlation (window=8 points)
    slide_w = 8
    slide_results = []
    for i in range(len(steps) - slide_w + 1):
        mask_i = slice(i, i + slide_w)
        s_i = steps[mask_i]
        r23_i = r23[mask_i]
        vl_i = vl[mask_i]
        if np.any(np.isnan(vl_i)):
            continue
        r_i, p_i = stats.pearsonr(r23_i, vl_i)
        slide_results.append({
            "step": int(s_i[-1]),
            "step_mid": int(s_i[len(s_i)//2]),
            "r": float(r_i),
            "p": float(p_i),
        })

    results["sliding_window"] = slide_results
    results["sliding_w"] = slide_w

    return results


def plot_analysis_3(results, spectra, vl_map):
    """Plot phase-specific correlations."""
    W = 10
    windows = spectra[W]
    steps = np.array([w["step"] for w in windows])
    r23 = np.array([w["r23"] for w in windows])
    vl = np.array([vl_map.get(s, np.nan) for s in steps])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Scatter plots by phase
    ax = axes[0, 0]
    colors = {"pre_shift": "green", "rapid_improve": "orange", "overfitting": "red"}
    labels = {"pre_shift": "Pre-shift", "rapid_improve": "Rapid improve", "overfitting": "Overfitting"}
    phase_masks = {
        "pre_shift": (steps < SHIFT_STEP) & ~np.isnan(vl),
        "rapid_improve": (steps >= SHIFT_STEP) & (steps < 22200) & ~np.isnan(vl),
        "overfitting": (steps >= 22200) & ~np.isnan(vl),
    }
    for phase, mask in phase_masks.items():
        if mask.sum() < 2:
            continue
        r_val = results.get(phase, {}).get("r_gap_vl", None)
        label = f"{labels[phase]} (r={r_val:.3f})" if r_val else labels[phase]
        ax.scatter(r23[mask], vl[mask], c=colors[phase], s=30, alpha=0.7, label=label)
    ax.set_xlabel("σ₂/σ₃")
    ax.set_ylabel("Val loss")
    ax.set_title("Phase-Specific: σ₂/σ₃ vs Val Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Bar chart of correlations
    ax = axes[0, 1]
    phase_order = ["pre_shift", "rapid_improve", "overfitting", "full"]
    phase_labels = ["Pre-shift", "Rapid improve", "Overfitting", "Full"]
    bar_colors = ["green", "orange", "red", "blue"]
    gap_r = [results.get(p, {}).get("r_gap_vl", 0) for p in phase_order]
    drift_r = [results.get(p, {}).get("r_drift_vl", 0) for p in phase_order]
    x = np.arange(len(phase_order))
    w_bar = 0.35
    ax.bar(x - w_bar/2, gap_r, w_bar, color=bar_colors, alpha=0.7, label="σ₂/σ₃ vs vl")
    ax.bar(x + w_bar/2, drift_r, w_bar, color=bar_colors, alpha=0.4,
           hatch="//", label="drift vs vl")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, rotation=20, fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation by Phase", fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel C: Sliding window correlation
    ax = axes[1, 0]
    slide = results.get("sliding_window", [])
    if slide:
        s_steps = [s["step_mid"] for s in slide]
        s_r = [s["r"] for s in slide]
        ax.plot(s_steps, s_r, "b-o", markersize=3, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8, label="shift")
        ax.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6,
                   label="overfit start")
        ax.axvspan(steps.min(), SHIFT_STEP, alpha=0.06, color="green")
        ax.axvspan(SHIFT_STEP, 22200, alpha=0.06, color="orange")
        ax.axvspan(22200, steps.max(), alpha=0.06, color="red")
        ax.set_xlabel("Training step (window midpoint)")
        ax.set_ylabel(f"Local Pearson r (w={results['sliding_w']})")
        ax.set_title("Sliding-Window Correlation", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel D: Overlay — σ₂/σ₃ (left axis) and val_loss (right axis)
    ax1 = axes[1, 1]
    ax1.plot(steps, r23, "b-o", markersize=3, linewidth=2, label="σ₂/σ₃")
    ax1.set_ylabel("σ₂/σ₃", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.axvline(SHIFT_STEP, color="red", linewidth=2.5, alpha=0.8)
    ax1.axvline(22200, color="purple", linewidth=1.5, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    valid = ~np.isnan(vl)
    ax2.plot(steps[valid], vl[valid], "k--", markersize=2, linewidth=1.5,
             alpha=0.6, label="val_loss")
    ax2.set_ylabel("Val loss", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_xlabel("Training step")
    ax1.set_title("Overlay: σ₂/σ₃ vs Val Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        "GPT-2 124M: Phase-Specific Spectral Correlations",
        fontsize=14, fontweight="bold")

    out = OUT_DIR / "gpt2_phase_correlations.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Noise Floor / Marchenko-Pastur
# ══════════════════════════════════════════════════════════════════════

def analysis_4_noise_floor(spectra, P):
    """
    Compare observed eigenvalue ratios to Marchenko-Pastur predictions.
    Under pure noise: X ~ N(0, σ²/p), Gram matrix G = XXT has MP distribution.
    For W×P with γ = W/P → 0, the MP upper edge = σ²(1+√γ)², lower = σ²(1-√γ)².
    Adjacent eigenvalue ratio under MP ≈ 1 + O(W^{-2/3}).
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Noise Floor Characterization")
    print("=" * 80)

    results = {}

    for W in sorted(spectra.keys()):
        windows = spectra[W]
        steps = [w["step"] for w in windows]
        r23 = np.array([w["r23"] for w in windows])
        r12 = np.array([w["r12"] for w in windows])

        gamma = W / P
        # MP prediction for adjacent eigenvalue ratio at the edge
        # Tracy-Widom: spacing ~ W^{-2/3}, ratio ≈ 1 + c * W^{-2/3}
        mp_ratio_approx = 1.0 + 2.0 * (W ** (-2.0/3.0))
        # Also compute BBP threshold: signal detectable if σ_signal > σ_noise * γ^{1/4}
        # For our case γ ~ 10/124M ~ 8e-8, γ^{1/4} ~ 0.017
        bbp_threshold = gamma ** 0.25

        results[W] = {
            "gamma": float(gamma),
            "mp_ratio_approx": float(mp_ratio_approx),
            "bbp_threshold_gamma14": float(bbp_threshold),
            "r23_mean": float(r23.mean()),
            "r23_std": float(r23.std()),
            "r23_min": float(r23.min()),
            "r23_max": float(r23.max()),
            "r12_mean": float(r12.mean()),
            "r12_max": float(r12.max()),
            "excess_over_mp": float(r23.mean() - mp_ratio_approx),
            "n_above_mp": int(np.sum(r23 > mp_ratio_approx)),
            "n_total": len(r23),
        }

        print(f"\n  W={W} (γ = {gamma:.2e}):")
        print(f"    MP predicted ratio: {mp_ratio_approx:.4f}")
        print(f"    Observed σ₂/σ₃: {r23.mean():.3f} ± {r23.std():.3f} "
              f"(range [{r23.min():.3f}, {r23.max():.3f}])")
        print(f"    Excess over MP: {r23.mean() - mp_ratio_approx:.3f}")
        print(f"    Windows above MP: {np.sum(r23 > mp_ratio_approx)}/{len(r23)} "
              f"({100*np.sum(r23 > mp_ratio_approx)/len(r23):.0f}%)")
        print(f"    Observed σ₁/σ₂: {r12.mean():.3f} ± {r12.std():.3f}")

    return results


def plot_analysis_4(noise_results, spectra, P):
    """Plot noise floor characterization."""
    W_values = sorted(noise_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: σ₂/σ₃ histograms per W
    ax = axes[0]
    for W in W_values:
        windows = spectra[W]
        r23 = [w["r23"] for w in windows]
        ax.hist(r23, bins=20, alpha=0.5, label=f"W={W}")
        mp = noise_results[W]["mp_ratio_approx"]
        ax.axvline(mp, linestyle="--", linewidth=1.5,
                   label=f"MP W={W}: {mp:.3f}")
    ax.set_xlabel("σ₂/σ₃")
    ax.set_ylabel("Count")
    ax.set_title("σ₂/σ₃ Distributions vs MP Prediction", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: σ₂/σ₃ trajectory (W=10) with MP floor
    ax = axes[1]
    W = 10
    windows = spectra[W]
    steps = [w["step"] for w in windows]
    r23 = [w["r23"] for w in windows]
    mp = noise_results[W]["mp_ratio_approx"]
    ax.plot(steps, r23, "b-o", markersize=3, linewidth=2, label="σ₂/σ₃ (W=10)")
    ax.axhline(mp, color="red", linestyle="--", linewidth=2,
               label=f"MP noise floor: {mp:.3f}")
    ax.axvline(SHIFT_STEP, color="orange", linestyle="--", linewidth=2, alpha=0.7,
               label=f"shift @ {SHIFT_STEP}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("σ₂/σ₃")
    ax.set_title("σ₂/σ₃ Trajectory vs Noise Floor", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Excess ratio over MP as function of W
    ax = axes[2]
    Ws = W_values
    excess = [noise_results[W]["r23_mean"] - noise_results[W]["mp_ratio_approx"] for W in Ws]
    mp_vals = [noise_results[W]["mp_ratio_approx"] for W in Ws]
    obs_vals = [noise_results[W]["r23_mean"] for W in Ws]
    ax.bar(range(len(Ws)), obs_vals, alpha=0.7, color="blue", label="Observed mean")
    ax.bar(range(len(Ws)), mp_vals, alpha=0.3, color="red", label="MP prediction")
    ax.set_xticks(range(len(Ws)))
    ax.set_xticklabels([f"W={W}" for W in Ws])
    ax.set_ylabel("σ₂/σ₃")
    ax.set_title("Observed vs MP Prediction by W", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "GPT-2 124M: Noise Floor Characterization (Marchenko-Pastur)",
        fontsize=14, fontweight="bold")

    out = OUT_DIR / "gpt2_noise_floor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Comparison with TinyStories (cross-scale)
# ══════════════════════════════════════════════════════════════════════

def analysis_5_cross_scale(spectra, vl_map, P):
    """
    Compare GPT-2 spectral gap dynamics with TinyStories results.
    Load TinyStories cached data if available.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Cross-Scale Comparison (GPT-2 vs TinyStories)")
    print("=" * 80)

    # Load TinyStories data (seed 42)
    ts_path = SCRIPT_DIR.parent / "runs" / "scale_124M" / "pilot_124M_b20.95_s42" / "results" / "pc2_noise_test.json"
    if not ts_path.exists():
        print(f"  TinyStories data not found at {ts_path}, skipping comparison.")
        return None

    with open(ts_path) as f:
        ts_data = json.load(f)

    ts_steps = [w["step"] for w in ts_data]
    ts_r23 = [w["sigma_ratio_23"] for w in ts_data]
    ts_r12 = [w["sigma_ratio_12"] for w in ts_data]

    # GPT-2 W=10 data
    W = 10
    gpt2_windows = spectra[W]
    gpt2_steps = [w["step"] for w in gpt2_windows]
    gpt2_r23 = [w["r23"] for w in gpt2_windows]
    gpt2_r12 = [w["r12"] for w in gpt2_windows]

    results = {
        "tinystories": {
            "r23_mean": float(np.mean(ts_r23)),
            "r23_max": float(np.max(ts_r23)),
            "r23_min": float(np.min(ts_r23)),
            "r12_mean": float(np.mean(ts_r12)),
            "n_windows": len(ts_r23),
        },
        "gpt2": {
            "r23_mean": float(np.mean(gpt2_r23)),
            "r23_max": float(np.max(gpt2_r23)),
            "r23_min": float(np.min(gpt2_r23)),
            "r12_mean": float(np.mean(gpt2_r12)),
            "n_windows": len(gpt2_r23),
        },
    }

    print(f"\n  Cross-scale comparison (W=10):")
    print(f"  {'Metric':<20} {'TinyStories':<15} {'GPT-2':<15}")
    print(f"  {'-'*50}")
    print(f"  {'σ₂/σ₃ mean':<20} {results['tinystories']['r23_mean']:<15.3f} {results['gpt2']['r23_mean']:<15.3f}")
    print(f"  {'σ₂/σ₃ max':<20} {results['tinystories']['r23_max']:<15.3f} {results['gpt2']['r23_max']:<15.3f}")
    print(f"  {'σ₁/σ₂ mean':<20} {results['tinystories']['r12_mean']:<15.3f} {results['gpt2']['r12_mean']:<15.3f}")
    print(f"  {'N windows':<20} {results['tinystories']['n_windows']:<15} {results['gpt2']['n_windows']:<15}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: σ₂/σ₃ trajectories (normalized x-axis)
    ax = axes[0]
    ts_x = np.linspace(0, 1, len(ts_r23))
    gpt2_x = np.linspace(0, 1, len(gpt2_r23))
    ax.plot(ts_x, ts_r23, "b-", linewidth=2, label="TinyStories (51M)")
    ax.plot(gpt2_x, gpt2_r23, "r-", linewidth=2, label="GPT-2 (124M)")
    ax.set_xlabel("Normalized training progress")
    ax.set_ylabel("σ₂/σ₃")
    ax.set_title("Spectral Gap Ratio: Cross-Scale", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: σ₂/σ₃ distributions
    ax = axes[1]
    ax.hist(ts_r23, bins=20, alpha=0.5, color="blue", label="TinyStories", density=True)
    ax.hist(gpt2_r23, bins=20, alpha=0.5, color="red", label="GPT-2", density=True)
    ax.set_xlabel("σ₂/σ₃")
    ax.set_ylabel("Density")
    ax.set_title("σ₂/σ₃ Distributions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: σ₁/σ₂ comparison
    ax = axes[2]
    ax.plot(ts_x, ts_r12, "b-", linewidth=2, label="TinyStories")
    ax.plot(gpt2_x, gpt2_r12, "r-", linewidth=2, label="GPT-2")
    ax.set_xlabel("Normalized training progress")
    ax.set_ylabel("σ₁/σ₂")
    ax.set_title("Leading Ratio: Cross-Scale", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cross-Scale Comparison: TinyStories (51M) vs GPT-2 (124M)",
        fontsize=14, fontweight="bold")

    out = OUT_DIR / "gpt2_cross_scale.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # Step 1: Discover checkpoints and build val_loss map
    ckpts = discover_checkpoints()
    print(f"Found {len(ckpts)} checkpoints (steps {ckpts[0][0]}-{ckpts[-1][0]})")

    vl_map = build_val_loss_map()
    print(f"Val_loss map: {len(vl_map)} entries")

    # Step 2: Load deltas and precompute DOT matrix
    delta_steps, DOT, P = load_deltas_and_dot_matrix(ckpts)

    # Step 3: Compute spectra for W=10 and W=20
    W_values = [10, 20]
    spectra = compute_spectra_multi_W(delta_steps, DOT, W_values)

    # Free DOT matrix
    del DOT

    # Step 4: Run analyses
    results_1 = analysis_1_global_svd(spectra, vl_map)
    plot_analysis_1(results_1)

    results_2, steps, r23, r12, sv1, sv2, sv3, drift, pc1, pc2 = \
        analysis_2_shift_event(spectra, vl_map)
    plot_analysis_2(results_2, steps, r23, r12, sv1, sv2, sv3, drift, pc1, pc2, vl_map)

    results_3 = analysis_3_phase_correlations(spectra, vl_map)
    plot_analysis_3(results_3, spectra, vl_map)

    results_4 = analysis_4_noise_floor(spectra, P)
    plot_analysis_4(results_4, spectra, P)

    results_5 = analysis_5_cross_scale(spectra, vl_map, P)

    # Save all results
    all_results = {
        "config": {
            "shift_step": SHIFT_STEP,
            "overfit_step": 22200,
            "W_values": W_values,
            "P_total": P,
            "n_checkpoints": len(ckpts),
            "n_deltas": len(delta_steps),
        },
        "analysis_1_global_svd": {W: {k: v for k, v in r.items() if k != "val_loss"}
                                  for W, r in results_1.items()},
        "analysis_2_shift_event": results_2,
        "analysis_3_phase_correlations": results_3,
        "analysis_4_noise_floor": results_4,
    }
    if results_5:
        all_results["analysis_5_cross_scale"] = results_5

    out_json = OUT_DIR / "gpt2_spectral_edge.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nSaved results: {out_json}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
