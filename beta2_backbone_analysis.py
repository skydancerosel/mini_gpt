#!/usr/bin/env python3
"""
Full backbone decomposition for the beta2 ablation runs (4000-step, λ=2.0).

For each β₂ ∈ {0.99, 0.95, 0.90, 0.80, 0.0}:
  1. Load all 52 checkpoints, trunk-only params (25.2M)
  2. Row-normalized uncentered SVD → v_b, PC1%
  3. a(t) = ⟨Δθ_t, v_b⟩,  r(t) = ‖Δθ_t − a(t)·v_b‖
  4. Finite-difference speeds ȧ(t), ṙ(t) at 200-step resolution
  5. Piecewise power-law fits in regimes: [0-2000], [2000-4000], [0-4000]
  6. Pearson correlations corr(p_ood, ‖r(t)‖) per regime
  7. 200-step update alignment: |cos(u(t), v_b)|
  8. Merge with pilot_metrics.json → per-beta2 CSV
  9. Cross-beta2 summary CSV + JSON + comparison plots

Output:  analysis/beta2_backbone/
"""

import csv
import gc
import json
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


# ═══════════════════════════════════════════════════════════════════════════
# Trunk-only parameter extraction (matches attractor_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)

BASE_DIR = Path("runs/beta2_ablation")
OUT_DIR = Path("analysis/beta2_backbone")

BETA2_VALUES = [0.99, 0.95, 0.90, 0.80, 0.0]


def flatten_trunk(state_dict):
    parts = []
    for key in sorted(state_dict.keys()):
        if TRUNK_PATTERN.match(key):
            parts.append(state_dict[key].cpu().numpy().astype(np.float64).ravel())
    if not parts:
        raise ValueError("No params matched TRUNK_PATTERN")
    return np.concatenate(parts)


def load_trunk(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vec = flatten_trunk(ckpt["model_state_dict"])
    del ckpt
    return vec


def power_law_fit(steps, values):
    """Fit log(y) = γ·log(t) + log(C) via OLS.  Returns (γ, C, R²)."""
    mask = [(s > 0 and v is not None and v > 0) for s, v in zip(steps, values)]
    xs = np.array([np.log(s) for s, m in zip(steps, mask) if m])
    ys = np.array([np.log(v) for v, m in zip(values, mask) if m])
    if len(xs) < 2:
        return None, None, None
    A = np.vstack([xs, np.ones_like(xs)]).T
    result = np.linalg.lstsq(A, ys, rcond=None)
    gamma, log_C = result[0]
    C = np.exp(log_C)
    ss_res = np.sum((ys - (gamma * xs + log_C)) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    return float(gamma), float(C), float(r2)


def pearson_corr(x, y):
    """Pearson correlation, returning None if insufficient data."""
    valid = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(valid) < 3:
        return None
    xv, yv = zip(*valid)
    xv, yv = np.array(xv), np.array(yv)
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Per-beta2 analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_beta2(beta2, run_dir, out_dir):
    """Full backbone decomposition + regime analysis for one beta2 run."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b2_label = f"b2={beta2}"

    # Discover checkpoints
    ckpt_files = sorted(run_dir.glob("ckpt_*.pt"))
    steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)
    print(f"\n  [{b2_label}] {len(steps)} checkpoints [{steps[0]}, {steps[-1]}]")

    # Load θ₀ (step 1)
    ref_step = steps[0]
    theta0 = load_trunk(run_dir / f"ckpt_{ref_step:06d}.pt")
    n_trunk = len(theta0)
    print(f"  [{b2_label}] Trunk dim: {n_trunk:,}")

    # Load all checkpoints and compute deltas
    other_steps = [s for s in steps if s != ref_step]
    print(f"  [{b2_label}] Loading {len(other_steps)} checkpoints...")
    thetas = {ref_step: theta0}
    deltas = {}
    for i, s in enumerate(other_steps):
        theta_t = load_trunk(run_dir / f"ckpt_{s:06d}.pt")
        thetas[s] = theta_t
        deltas[s] = theta_t - theta0
        if (i + 1) % 10 == 0:
            print(f"    [{b2_label}] {i+1}/{len(other_steps)}")

    # ── SVD ──
    X = np.stack([deltas[s] for s in other_steps], axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_rn = X / np.maximum(norms, 1e-12)

    U, S, Vt = np.linalg.svd(X_rn, full_matrices=False)
    v_b = Vt[0]

    total_var = np.sum(S ** 2)
    pc1_pct = S[0] ** 2 / total_var * 100
    pc2_pct = S[1] ** 2 / total_var * 100 if len(S) > 1 else 0

    # Raw variance
    _, S_raw, _ = np.linalg.svd(X, full_matrices=False)
    total_var_raw = np.sum(S_raw ** 2)
    pc1_pct_raw = S_raw[0] ** 2 / total_var_raw * 100

    # Sign fix: a(last) > 0
    if np.dot(deltas[other_steps[-1]], v_b) < 0:
        v_b = -v_b

    print(f"  [{b2_label}] PC1(rownorm)={pc1_pct:.1f}%, PC1(raw)={pc1_pct_raw:.1f}%")

    # ── a(t), r(t) time series ──
    rows = []
    r_vecs = {}

    rows.append({
        "step": ref_step, "a_t": 0.0, "norm_r_t": 0.0,
        "norm_delta_t": 0.0, "backbone_frac": 0.0,
    })
    r_vecs[ref_step] = np.zeros(n_trunk)

    for s in other_steps:
        d = deltas[s]
        a_t = float(np.dot(d, v_b))
        r_vec = d - a_t * v_b
        norm_r = float(np.linalg.norm(r_vec))
        norm_d = float(np.linalg.norm(d))
        bfrac = (a_t ** 2) / (norm_d ** 2) if norm_d > 0 else 0.0
        rows.append({
            "step": s, "a_t": a_t, "norm_r_t": norm_r,
            "norm_delta_t": norm_d, "backbone_frac": float(bfrac),
        })
        r_vecs[s] = r_vec

    rows.sort(key=lambda r: r["step"])

    # ── Speeds ──
    for i, row in enumerate(rows):
        s = row["step"]
        if i == 0:
            row["a_dot"] = 0.0
            row["r_dot"] = 0.0
            continue
        prev = rows[i - 1]
        dt = s - prev["step"]
        if dt <= 0:
            row["a_dot"] = 0.0
            row["r_dot"] = 0.0
            continue
        row["a_dot"] = (row["a_t"] - prev["a_t"]) / dt
        row["r_dot"] = float(np.linalg.norm(r_vecs[s] - r_vecs[prev["step"]])) / dt

    # ── Update alignment: cos(u(t), v_b) ──
    print(f"  [{b2_label}] Computing update--backbone alignment...")
    all_steps_sorted = sorted(thetas.keys())
    for i, row in enumerate(rows):
        s = row["step"]
        # Find previous checkpoint 200 steps back
        prev_s = s - 200
        if prev_s in thetas and s in thetas:
            u = thetas[s] - thetas[prev_s]
            norm_u = np.linalg.norm(u)
            if norm_u > 1e-12:
                cos_val = float(np.dot(u, v_b) / norm_u)
                row["update_cos_vb"] = cos_val
                row["update_abs_cos_vb"] = abs(cos_val)
            else:
                row["update_cos_vb"] = 0.0
                row["update_abs_cos_vb"] = 0.0
        else:
            row["update_cos_vb"] = None
            row["update_abs_cos_vb"] = None

    # Free heavy memory
    del deltas, r_vecs, X, X_rn, thetas
    gc.collect()

    # ── Merge metrics ──
    metrics_path = run_dir / "pilot_metrics.json"
    metrics_by_step = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            for m in json.load(f):
                metrics_by_step[m["step"]] = m

    for row in rows:
        s = row["step"]
        if s in metrics_by_step:
            m = metrics_by_step[s]
            row["probe_ood_acc"] = m.get("probe_ood_acc")
            row["probe_in_acc"] = m.get("probe_in_acc")
            row["val_loss"] = m.get("val_loss")
            row["train_loss"] = m.get("train_loss")

    # ── Power-law fits ──
    steps_list = [r["step"] for r in rows]
    a_abs = [abs(r["a_t"]) if r["a_t"] is not None else None for r in rows]
    r_list = [r["norm_r_t"] for r in rows]
    p_ood = [r.get("probe_ood_acc") for r in rows]

    regimes = [
        ("0-2000", 0, 2000),
        ("2000-4000", 2000, 4000),
        ("0-4000", 0, 4000),
    ]

    print(f"\n  [{b2_label}] Power-law fits:")
    print(f"  {'regime':<12s}  {'γ_a':>7s}  {'R²_a':>6s}  {'γ_r':>7s}  {'R²_r':>6s}")
    print(f"  {'-'*45}")

    fit_results = {}
    for name, lo, hi in regimes:
        idx = [i for i, s in enumerate(steps_list) if lo < s <= hi]
        if not idx:
            idx = [i for i, s in enumerate(steps_list) if lo <= s <= hi and s > 0]
        s_r = [steps_list[i] for i in idx]
        a_r = [a_abs[i] for i in idx]
        r_r = [r_list[i] for i in idx]

        ga, Ca, r2a = power_law_fit(s_r, a_r)
        gr, Cr, r2r = power_law_fit(s_r, r_r)
        fit_results[name] = {
            "gamma_a": ga, "C_a": Ca, "R2_a": r2a,
            "gamma_r": gr, "C_r": Cr, "R2_r": r2r,
        }
        ga_s = f"{ga:+.3f}" if ga is not None else "N/A"
        r2a_s = f"{r2a:.3f}" if r2a is not None else "N/A"
        gr_s = f"{gr:+.3f}" if gr is not None else "N/A"
        r2r_s = f"{r2r:.3f}" if r2r is not None else "N/A"
        print(f"  {name:<12s}  {ga_s:>7s}  {r2a_s:>6s}  {gr_s:>7s}  {r2r_s:>6s}")

    # ── Correlations ──
    print(f"\n  [{b2_label}] Correlations corr(p_ood, ‖r(t)‖):")
    corr_results = {}
    for name, lo, hi in [("full", 0, 4000)] + regimes:
        idx = [i for i, s in enumerate(steps_list) if lo < s <= hi]
        if not idx:
            idx = [i for i, s in enumerate(steps_list) if lo <= s <= hi and s > 0]
        pv = [p_ood[i] for i in idx]
        rv = [r_list[i] for i in idx]
        c = pearson_corr(pv, rv)
        corr_results[name] = c
        c_s = f"{c:+.4f}" if c is not None else "N/A"
        print(f"    {name:<12s}: {c_s}")

    # ── Update alignment summary ──
    ucos = [r.get("update_abs_cos_vb") for r in rows if r.get("update_abs_cos_vb") is not None]
    mean_ucos = float(np.mean(ucos)) if ucos else None
    # Early alignment (steps < 2000)
    ucos_early = [r.get("update_abs_cos_vb") for r in rows
                  if r.get("update_abs_cos_vb") is not None and r["step"] <= 2000]
    mean_ucos_early = float(np.mean(ucos_early)) if ucos_early else None

    print(f"\n  [{b2_label}] Update alignment: mean |cos(u,v_b)|={mean_ucos:.4f}" if mean_ucos else "")
    print(f"  [{b2_label}] Early (<2k) alignment: {mean_ucos_early:.4f}" if mean_ucos_early else "")

    # ── Drift magnitude ──
    final_row = rows[-1]
    drift_mag = final_row["norm_delta_t"]
    best_pood = max((r.get("probe_ood_acc", 0) or 0) for r in rows)

    # ── Save CSV ──
    b2_str = str(beta2).replace(".", "")
    csv_path = out_dir / f"backbone_timeseries_b2{b2_str}.csv"
    keys = ["step", "a_t", "norm_r_t", "norm_delta_t", "backbone_frac",
            "a_dot", "r_dot", "update_cos_vb", "update_abs_cos_vb",
            "probe_ood_acc", "probe_in_acc", "val_loss", "train_loss"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = [str(row.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")
    print(f"  [{b2_label}] Saved: {csv_path}")

    # ── Save singular values ──
    sv_path = out_dir / f"singular_values_b2{b2_str}.json"
    n_sv = min(20, len(S))
    with open(sv_path, "w") as f:
        json.dump({
            "beta2": beta2,
            "trunk_dim": n_trunk,
            "n_checkpoints": len(other_steps),
            "pc1_pct_rownorm": float(pc1_pct),
            "pc2_pct_rownorm": float(pc2_pct),
            "pc1_pct_raw": float(pc1_pct_raw),
            "singular_values_rownorm": S[:n_sv].tolist(),
            "explained_var_rownorm": [(s**2 / total_var) for s in S[:n_sv]],
        }, f, indent=2)

    # ── Plots ──
    if HAS_MPL:
        make_per_beta2_plots(beta2, rows, fit_results, pc1_pct, out_dir)

    # Return summary
    return {
        "beta2": beta2,
        "pc1_pct_rownorm": pc1_pct,
        "pc2_pct_rownorm": pc2_pct,
        "pc1_pct_raw": pc1_pct_raw,
        "drift_magnitude": drift_mag,
        "best_pood": best_pood,
        "mean_update_cos": mean_ucos,
        "mean_update_cos_early": mean_ucos_early,
        "gamma_a_0_2k": fit_results["0-2000"].get("gamma_a"),
        "gamma_r_0_2k": fit_results["0-2000"].get("gamma_r"),
        "gamma_a_2k_4k": fit_results["2000-4000"].get("gamma_a"),
        "gamma_r_2k_4k": fit_results["2000-4000"].get("gamma_r"),
        "gamma_a_full": fit_results["0-4000"].get("gamma_a"),
        "gamma_r_full": fit_results["0-4000"].get("gamma_r"),
        "R2_a_full": fit_results["0-4000"].get("R2_a"),
        "R2_r_full": fit_results["0-4000"].get("R2_r"),
        "corr_pood_r_full": corr_results.get("full"),
        "corr_pood_r_0_2k": corr_results.get("0-2000"),
        "corr_pood_r_2k_4k": corr_results.get("2000-4000"),
        "fit_results": fit_results,
        "corr_results": corr_results,
        "timeseries": rows,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def make_per_beta2_plots(beta2, rows, fit_results, pc1_pct, out_dir):
    """Per-beta2 4-panel plot: a(t), r(t), p_ood vs r, log-log fits."""
    b2_str = str(beta2).replace(".", "")

    steps = [r["step"] for r in rows]
    a_t = [r["a_t"] for r in rows]
    norm_r = [r["norm_r_t"] for r in rows]
    p_ood = [r.get("probe_ood_acc") for r in rows]
    ucos = [r.get("update_cos_vb") for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Backbone Decomposition — β₂={beta2}  (PC1={pc1_pct:.1f}%)",
                 fontsize=14, fontweight="bold")

    # 1. a(t)
    ax = axes[0, 0]
    ax.plot(steps, a_t, "b-o", ms=3)
    ax.set_xlabel("Step"); ax.set_ylabel("a(t)")
    ax.set_title("Backbone projection a(t)")
    ax.grid(True, alpha=0.3)

    # 2. ‖r(t)‖
    ax = axes[0, 1]
    ax.plot(steps, norm_r, "r-o", ms=3)
    ax.set_xlabel("Step"); ax.set_ylabel("‖r(t)‖")
    ax.set_title("Residual norm ‖r(t)‖")
    ax.grid(True, alpha=0.3)

    # 3. p_ood + ‖r(t)‖ overlay
    ax = axes[0, 2]
    ax_r = ax.twinx()
    valid = [(s, p) for s, p in zip(steps, p_ood) if p is not None]
    if valid:
        sp, pv = zip(*valid)
        ax.plot(sp, pv, "g-o", ms=3, label="p_ood")
    ax_r.plot(steps, norm_r, "r-s", ms=2, alpha=0.6, label="‖r(t)‖")
    ax.set_xlabel("Step"); ax.set_ylabel("p_ood", color="green")
    ax_r.set_ylabel("‖r(t)‖", color="red")
    ax.set_title("p_ood vs ‖r(t)‖")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Log-log with piecewise fits
    ax = axes[1, 0]
    mask = [s > 0 for s in steps]
    log_s = [np.log10(s) for s, m in zip(steps, mask) if m]
    log_a = [np.log10(abs(a)) if abs(a) > 0 else np.nan for a, m in zip(a_t, mask) if m]
    ax.plot(log_s, log_a, "ko", ms=4, alpha=0.5, label="data")
    for name, lo, hi, color in [("0-2000", 0, 2000, "C0"), ("2000-4000", 2000, 4000, "C1")]:
        fr = fit_results.get(name, {})
        g, C, r2 = fr.get("gamma_a"), fr.get("C_a"), fr.get("R2_a")
        if g is None: continue
        t_fit = np.linspace(max(lo, 1), hi, 100)
        y_fit = C * t_fit ** g
        ax.plot(np.log10(t_fit), np.log10(y_fit), color=color, lw=2,
                label=f"{name}: γ={g:+.2f} (R²={r2:.2f})")
    ax.set_xlabel("log₁₀(step)"); ax.set_ylabel("log₁₀(|a(t)|)")
    ax.set_title("|a(t)| power-law"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Log-log ‖r(t)‖
    ax = axes[1, 1]
    log_r = [np.log10(r) if r > 0 else np.nan for r, m in zip(norm_r, mask) if m]
    ax.plot(log_s, log_r, "ko", ms=4, alpha=0.5, label="data")
    for name, lo, hi, color in [("0-2000", 0, 2000, "C0"), ("2000-4000", 2000, 4000, "C1")]:
        fr = fit_results.get(name, {})
        g, C, r2 = fr.get("gamma_r"), fr.get("C_r"), fr.get("R2_r")
        if g is None: continue
        t_fit = np.linspace(max(lo, 1), hi, 100)
        y_fit = C * t_fit ** g
        valid_y = y_fit[y_fit > 0]
        valid_t = t_fit[y_fit > 0]
        if len(valid_y) > 0:
            ax.plot(np.log10(valid_t), np.log10(valid_y), color=color, lw=2,
                    label=f"{name}: γ={g:+.2f} (R²={r2:.2f})")
    ax.set_xlabel("log₁₀(step)"); ax.set_ylabel("log₁₀(‖r(t)‖)")
    ax.set_title("‖r(t)‖ power-law"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6. Update alignment signed cos
    ax = axes[1, 2]
    valid_u = [(s, c) for s, c in zip(steps, ucos) if c is not None]
    if valid_u:
        su, cu = zip(*valid_u)
        ax.plot(su, cu, "k-o", ms=3)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("cos(u(t), v_b)")
    ax.set_title("Update alignment (signed)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"backbone_b2{b2_str}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def make_cross_beta2_plots(summaries, out_dir):
    """Cross-β₂ comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cross-β₂ Backbone Comparison (4000 steps, λ=2.0)",
                 fontsize=14, fontweight="bold")

    cmap = plt.cm.viridis
    colors = {s["beta2"]: cmap(i / max(len(summaries)-1, 1))
              for i, s in enumerate(summaries)}

    # 1. a(t) across beta2
    ax = axes[0, 0]
    for s in summaries:
        ts = s["timeseries"]
        steps = [r["step"] for r in ts]
        at = [r["a_t"] for r in ts]
        ax.plot(steps, at, "-o", ms=2, color=colors[s["beta2"]],
                label=f"β₂={s['beta2']}")
    ax.set_xlabel("Step"); ax.set_ylabel("a(t)")
    ax.set_title("Backbone projection a(t)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2. ‖r(t)‖ across beta2
    ax = axes[0, 1]
    for s in summaries:
        ts = s["timeseries"]
        steps = [r["step"] for r in ts]
        nr = [r["norm_r_t"] for r in ts]
        ax.plot(steps, nr, "-o", ms=2, color=colors[s["beta2"]],
                label=f"β₂={s['beta2']}")
    ax.set_xlabel("Step"); ax.set_ylabel("‖r(t)‖")
    ax.set_title("Residual norm ‖r(t)‖")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. p_ood across beta2
    ax = axes[0, 2]
    for s in summaries:
        ts = s["timeseries"]
        valid = [(r["step"], r.get("probe_ood_acc")) for r in ts if r.get("probe_ood_acc") is not None]
        if valid:
            sp, pv = zip(*valid)
            ax.plot(sp, pv, "-o", ms=2, color=colors[s["beta2"]],
                    label=f"β₂={s['beta2']}")
    ax.set_xlabel("Step"); ax.set_ylabel("probe_ood_acc")
    ax.set_title("Probe OOD accuracy")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4. Backbone fraction across beta2
    ax = axes[1, 0]
    for s in summaries:
        ts = s["timeseries"]
        steps = [r["step"] for r in ts]
        bf = [r["backbone_frac"] for r in ts]
        ax.plot(steps, bf, "-o", ms=2, color=colors[s["beta2"]],
                label=f"β₂={s['beta2']}")
    ax.set_xlabel("Step"); ax.set_ylabel("Backbone fraction")
    ax.set_title("a(t)² / ‖Δθ‖²")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Update alignment across beta2
    ax = axes[1, 1]
    for s in summaries:
        ts = s["timeseries"]
        valid = [(r["step"], r.get("update_abs_cos_vb")) for r in ts
                 if r.get("update_abs_cos_vb") is not None]
        if valid:
            su, cu = zip(*valid)
            ax.plot(su, cu, "-o", ms=2, color=colors[s["beta2"]],
                    label=f"β₂={s['beta2']}")
    ax.set_xlabel("Step"); ax.set_ylabel("|cos(u, v_b)|")
    ax.set_title("Update--backbone alignment")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6. Summary bar chart: PC1% and best_pood
    ax = axes[1, 2]
    b2_labels = [str(s["beta2"]) for s in summaries]
    pc1_vals = [s["pc1_pct_rownorm"] for s in summaries]
    pood_vals = [s["best_pood"] * 100 for s in summaries]
    x = np.arange(len(b2_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, pc1_vals, w, label="PC1%", color="C0", alpha=0.7)
    bars2 = ax.bar(x + w/2, pood_vals, w, label="best p_ood×100", color="C2", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(b2_labels)
    ax.set_xlabel("β₂"); ax.set_ylabel("%")
    ax.set_title("PC1% and best p_ood")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = out_dir / "backbone_cross_beta2.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Beta2 Ablation — Full Backbone Decomposition")
    print("  (trunk-only, row-normalized SVD, 4000-step runs)")
    print("=" * 70)

    summaries = []
    for beta2 in BETA2_VALUES:
        b2_str = f"{beta2:.2f}" if beta2 != 0.0 else "0.0"
        run_dir = BASE_DIR / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
        if not run_dir.exists():
            print(f"\n  [SKIP] β₂={beta2}: {run_dir}")
            continue
        result = analyze_beta2(beta2, run_dir, OUT_DIR)
        summaries.append(result)

    # ── Cross-beta2 summary CSV ──
    csv_path = OUT_DIR / "beta2_backbone_summary.csv"
    summary_keys = [
        "beta2", "pc1_pct_rownorm", "pc2_pct_rownorm", "pc1_pct_raw",
        "drift_magnitude", "best_pood",
        "mean_update_cos", "mean_update_cos_early",
        "gamma_a_0_2k", "gamma_r_0_2k",
        "gamma_a_2k_4k", "gamma_r_2k_4k",
        "gamma_a_full", "gamma_r_full",
        "R2_a_full", "R2_r_full",
        "corr_pood_r_full", "corr_pood_r_0_2k", "corr_pood_r_2k_4k",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(summary_keys) + "\n")
        for s in summaries:
            vals = []
            for k in summary_keys:
                v = s.get(k)
                vals.append(f"{v}" if v is not None else "")
            f.write(",".join(vals) + "\n")
    print(f"\n  Saved summary CSV: {csv_path}")

    # ── Summary JSON ──
    json_path = OUT_DIR / "beta2_backbone_summary.json"
    json_summaries = []
    for s in summaries:
        d = {k: s.get(k) for k in summary_keys}
        d["fit_results"] = s.get("fit_results", {})
        d["corr_results"] = s.get("corr_results", {})
        json_summaries.append(d)
    with open(json_path, "w") as f:
        json.dump(json_summaries, f, indent=2)
    print(f"  Saved summary JSON: {json_path}")

    # ── Print summary table ──
    print(f"\n{'='*90}")
    print(f"  Cross-β₂ Summary")
    print(f"{'='*90}")
    print(f"  {'β₂':>5s}  {'PC1%':>6s}  {'drift':>8s}  {'best_pood':>9s}  "
          f"{'|cos|':>6s}  {'γ_a(0-2k)':>10s}  {'γ_r(0-2k)':>10s}  "
          f"{'γ_a(2-4k)':>10s}  {'γ_r(2-4k)':>10s}  {'corr':>6s}")
    print(f"  {'-'*85}")
    for s in summaries:
        b2 = s["beta2"]
        pc1 = s["pc1_pct_rownorm"]
        drift = s["drift_magnitude"]
        bp = s["best_pood"]
        uc = s["mean_update_cos"]
        ga1 = s.get("gamma_a_0_2k")
        gr1 = s.get("gamma_r_0_2k")
        ga2 = s.get("gamma_a_2k_4k")
        gr2 = s.get("gamma_r_2k_4k")
        cr = s.get("corr_pood_r_full")
        ga1_s = f"{ga1:+.3f}" if ga1 is not None else "N/A"
        gr1_s = f"{gr1:+.3f}" if gr1 is not None else "N/A"
        ga2_s = f"{ga2:+.3f}" if ga2 is not None else "N/A"
        gr2_s = f"{gr2:+.3f}" if gr2 is not None else "N/A"
        cr_s = f"{cr:+.3f}" if cr is not None else "N/A"
        uc_s = f"{uc:.4f}" if uc is not None else "N/A"
        print(f"  {b2:>5.2f}  {pc1:>6.1f}  {drift:>8.1f}  {bp:>9.3f}  "
              f"{uc_s:>6s}  {ga1_s:>10s}  {gr1_s:>10s}  "
              f"{ga2_s:>10s}  {gr2_s:>10s}  {cr_s:>6s}")

    # ── Cross-beta2 plots ──
    if HAS_MPL and len(summaries) >= 2:
        make_cross_beta2_plots(summaries, OUT_DIR)

    print(f"\n{'='*70}")
    print(f"  Done. All output in {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
