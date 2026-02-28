#!/usr/bin/env python3
"""
Step 1: Piecewise power-law fits  a(t) = C_a t^γ_a,  r(t) = C_r t^γ_r
        in three regimes: [0,2000], [2000,6000], [6000,10000].
Step 2: Pearson correlation corr(p_ood, ‖r(t)‖).

Reads backbone_timeseries_seed{42,271}.csv from analysis/backbone_decomposition/.
"""

import csv
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


DATA_DIR = Path("analysis/backbone_decomposition")
OUT_DIR = DATA_DIR


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = None
            rows.append(d)
    return rows


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
    # R²
    ss_res = np.sum((ys - (gamma * xs + log_C)) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    return float(gamma), float(C), float(r2)


def analyze_seed(seed, rows):
    steps = [r["step"] for r in rows]
    a_t = [abs(r["a_t"]) if r["a_t"] is not None else None for r in rows]
    r_t = [r["norm_r_t"] for r in rows]
    p_ood = [r["probe_ood_acc"] for r in rows]

    regimes = [
        ("0-2000", 0, 2000),
        ("2000-4000", 2000, 4000),
        ("0-4000", 0, 4000),
        ("4000-6000", 4000, 6000),
        ("4000-10000", 4000, 10000),
        ("6000-10000", 6000, 10000),
    ]

    print(f"\n{'='*70}")
    print(f"  Seed {seed}")
    print(f"{'='*70}")

    # ── Step 1: Piecewise power-law fits ──
    print(f"\n  Step 1: Power-law fits  y = C · t^γ")
    print(f"  {'regime':<12s}  {'γ_a':>6s}  {'C_a':>10s}  {'R²_a':>6s}  "
          f"{'γ_r':>6s}  {'C_r':>10s}  {'R²_r':>6s}")
    print(f"  {'-'*62}")

    fit_results = {}
    for name, lo, hi in regimes:
        idx = [i for i, s in enumerate(steps) if lo < s <= hi]
        if not idx:
            # include lo==0 edge case for first regime
            idx = [i for i, s in enumerate(steps) if lo <= s <= hi and s > 0]
        s_regime = [steps[i] for i in idx]
        a_regime = [a_t[i] for i in idx]
        r_regime = [r_t[i] for i in idx]

        ga, Ca, r2a = power_law_fit(s_regime, a_regime)
        gr, Cr, r2r = power_law_fit(s_regime, r_regime)

        fit_results[name] = {
            "gamma_a": ga, "C_a": Ca, "R2_a": r2a,
            "gamma_r": gr, "C_r": Cr, "R2_r": r2r,
        }

        ga_s = f"{ga:.3f}" if ga is not None else "N/A"
        Ca_s = f"{Ca:.4e}" if Ca is not None else "N/A"
        r2a_s = f"{r2a:.3f}" if r2a is not None else "N/A"
        gr_s = f"{gr:.3f}" if gr is not None else "N/A"
        Cr_s = f"{Cr:.4e}" if Cr is not None else "N/A"
        r2r_s = f"{r2r:.3f}" if r2r is not None else "N/A"

        print(f"  {name:<12s}  {ga_s:>6s}  {Ca_s:>10s}  {r2a_s:>6s}  "
              f"{gr_s:>6s}  {Cr_s:>10s}  {r2r_s:>6s}")

    # Interpret: before vs after 4k
    g_pre = fit_results.get("0-4000", {}).get("gamma_a")
    g_post = fit_results.get("4000-10000", {}).get("gamma_a")
    if g_pre is not None and g_post is not None:
        print(f"\n  → Before 4k: γ_a={g_pre:.3f}  |  After 4k: γ_a={g_post:.3f}")
        if g_post < 0:
            print(f"    Backbone RETREATING after step 4000")
        elif g_post < 0.1:
            print(f"    Backbone PLATEAUED after step 4000")
        else:
            print(f"    Backbone still growing after step 4000 (slower)")

    # ── Step 2: Correlations ──
    print(f"\n  Step 2: Correlations")

    # Full range
    valid = [(p, r) for p, r in zip(p_ood, r_t) if p is not None and r is not None and r > 0]
    if valid:
        pv, rv = zip(*valid)
        corr_full = float(np.corrcoef(pv, rv)[0, 1])
        print(f"  corr(p_ood, ‖r(t)‖) full range:     {corr_full:+.4f}")

    # Per regime
    for name, lo, hi in regimes:
        idx = [i for i, s in enumerate(steps) if lo < s <= hi]
        if not idx:
            idx = [i for i, s in enumerate(steps) if lo <= s <= hi and s > 0]
        pv_r = [p_ood[i] for i in idx if p_ood[i] is not None and r_t[i] is not None and r_t[i] > 0]
        rv_r = [r_t[i] for i in idx if p_ood[i] is not None and r_t[i] is not None and r_t[i] > 0]
        if len(pv_r) >= 3:
            c = float(np.corrcoef(pv_r, rv_r)[0, 1])
            print(f"  corr(p_ood, ‖r(t)‖) {name:<12s}:  {c:+.4f}")

    # Also: corr with a_dot and r_dot
    a_dot = [r.get("a_dot") for r in rows]
    r_dot = [r.get("r_dot") for r in rows]
    valid_ad = [(p, ad) for p, ad in zip(p_ood, a_dot)
                if p is not None and ad is not None and ad != 0]
    valid_rd = [(p, rd) for p, rd in zip(p_ood, r_dot)
                if p is not None and rd is not None and rd != 0]
    if valid_ad:
        pv, av = zip(*valid_ad)
        print(f"  corr(p_ood, ȧ(t)) full range:       {float(np.corrcoef(pv, av)[0, 1]):+.4f}")
    if valid_rd:
        pv, rv = zip(*valid_rd)
        print(f"  corr(p_ood, ṙ(t)) full range:       {float(np.corrcoef(pv, rv)[0, 1]):+.4f}")

    return fit_results


def make_regime_plots(seed, rows, fit_results):
    """Piecewise fits overlaid on data + correlation scatter."""
    steps = np.array([r["step"] for r in rows])
    a_t = np.array([abs(r["a_t"]) if r["a_t"] is not None else np.nan for r in rows])
    r_t = np.array([r["norm_r_t"] if r["norm_r_t"] is not None else np.nan for r in rows])
    p_ood = np.array([r["probe_ood_acc"] if r["probe_ood_acc"] is not None else np.nan for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Regime Analysis — Seed {seed}", fontsize=14, fontweight="bold")

    regimes_spec = [
        ("0-4000", 0, 4000, "C0"),
        ("4000-10000", 4000, 10000, "C1"),
    ]

    # ── Panel 1: log-log a(t) with piecewise fits ──
    ax = axes[0]
    mask = steps > 0
    ax.plot(np.log10(steps[mask]), np.log10(a_t[mask]), "ko", ms=4, alpha=0.5, label="data")
    for name, lo, hi, color in regimes_spec:
        fr = fit_results.get(name, {})
        g, C = fr.get("gamma_a"), fr.get("C_a")
        r2 = fr.get("R2_a")
        if g is None:
            continue
        t_fit = np.linspace(max(lo, 1), hi, 100)
        y_fit = C * t_fit ** g
        ax.plot(np.log10(t_fit), np.log10(y_fit), color=color, lw=2,
                label=f"{name}: γ={g:.2f} (R²={r2:.2f})")
    ax.set_xlabel("log₁₀(step)"); ax.set_ylabel("log₁₀(|a(t)|)")
    ax.set_title("|a(t)| piecewise power-law")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2: log-log ‖r(t)‖ with piecewise fits ──
    ax = axes[1]
    ax.plot(np.log10(steps[mask]), np.log10(r_t[mask]), "ko", ms=4, alpha=0.5, label="data")
    for name, lo, hi, color in regimes_spec:
        fr = fit_results.get(name, {})
        g, C = fr.get("gamma_r"), fr.get("C_r")
        r2 = fr.get("R2_r")
        if g is None:
            continue
        t_fit = np.linspace(max(lo, 1), hi, 100)
        y_fit = C * t_fit ** g
        ax.plot(np.log10(t_fit), np.log10(y_fit), color=color, lw=2,
                label=f"{name}: γ={g:.2f} (R²={r2:.2f})")
    ax.set_xlabel("log₁₀(step)"); ax.set_ylabel("log₁₀(‖r(t)‖)")
    ax.set_title("‖r(t)‖ piecewise power-law")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 3: p_ood vs ‖r(t)‖ scatter colored by regime ──
    ax = axes[2]
    for name, lo, hi, color in regimes_spec:
        idx = [i for i, s in enumerate(steps) if lo < s <= hi]
        if not idx:
            idx = [i for i, s in enumerate(steps) if lo <= s <= hi and s > 0]
        pv = [p_ood[i] for i in idx if not np.isnan(p_ood[i]) and not np.isnan(r_t[i])]
        rv = [r_t[i] for i in idx if not np.isnan(p_ood[i]) and not np.isnan(r_t[i])]
        if pv:
            ax.scatter(rv, pv, c=color, s=30, label=name, alpha=0.7, edgecolors="k", lw=0.3)
    ax.set_xlabel("‖r(t)‖"); ax.set_ylabel("probe_ood_acc")
    ax.set_title("p_ood vs ‖r(t)‖ by regime")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / f"regime_analysis_seed{seed}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    all_fits = {}
    for seed in [42, 271]:
        csv_path = DATA_DIR / f"backbone_timeseries_seed{seed}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} not found")
            continue
        rows = load_csv(csv_path)
        fits = analyze_seed(seed, rows)
        all_fits[seed] = fits

        if HAS_MPL:
            make_regime_plots(seed, rows, fits)

    # Save combined results
    out = {}
    for seed, fits in all_fits.items():
        out[f"seed_{seed}"] = fits
    with open(OUT_DIR / "regime_fits.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'regime_fits.json'}")


if __name__ == "__main__":
    main()
