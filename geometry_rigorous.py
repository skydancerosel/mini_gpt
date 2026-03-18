#!/usr/bin/env python3
"""
Rigorous statistical tests for geometry → performance prediction.

Test 1: Granger causality — does geometry add predictive power beyond AR(p) + LR?
Test 2: Block permutation — honest p-values respecting autocorrelation
Test 3: First-difference correlations — removes shared trends entirely
Test 4: Reheating cross-sectional prediction — LR confound eliminated (needs Phase 3 data)

Usage:
    python geometry_rigorous.py --tests 1,2,3
    python geometry_rigorous.py --tests 4
    python geometry_rigorous.py --tests all
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path("runs/beta2_ablation")
RESULTS_DIR = Path("results")

RUN_DIRS = [
    BASE_DIR / "pilot_wd0.5_lr0.001_lp2.0_b20.99_s42",
    BASE_DIR / "pilot_wd0.5_lr0.001_lp2.0_b20.95_s42",
    BASE_DIR / "pilot_wd0.5_lr0.001_lp2.0_b20.80_s42",
    BASE_DIR / "pilot_wd0.5_lr0.001_lp0.0_b20.95_s42",
]

RUN_LABELS = ["β2=0.99 λ=2", "β2=0.95 λ=2", "β2=0.80 λ=2", "β2=0.95 λ=0"]

GEOMETRY_FEATURES = ["drift_speed", "align_u", "k95_roll", "PC1_roll",
                     "a_speed", "r_speed", "kappa", "gamma"]
TARGETS = ["val_loss", "probe_ood_acc"]
TOP_FEATURES = ["drift_speed", "align_u", "k95_roll"]  # from prior analysis


# ── Data Loading ──────────────────────────────────────────────────────────

def load_aligned_data(run_dir):
    """Load geometry + metrics aligned at common steps.

    Returns dict with arrays: steps, val_loss, probe_ood_acc, lr,
    and each geometry feature, all aligned at ~46 common steps.
    """
    geo_path = run_dir / "causal_geometry.json"
    met_path = run_dir / "pilot_metrics.json"
    if not geo_path.exists() or not met_path.exists():
        return None

    geo = json.load(open(geo_path))
    met = json.load(open(met_path))

    # Build lookup tables
    geo_by_step = {w["step"]: w for w in geo["windows"]}
    met_by_step = {m["step"]: m for m in met}

    # Find common steps
    common_steps = sorted(set(geo_by_step.keys()) & set(met_by_step.keys()))
    if len(common_steps) < 10:
        return None

    result = {"steps": np.array(common_steps)}
    result["val_loss"] = np.array([met_by_step[s]["val_loss"] for s in common_steps])
    result["probe_ood_acc"] = np.array([met_by_step[s]["probe_ood_acc"] for s in common_steps])
    result["lr"] = np.array([met_by_step[s]["lr"] for s in common_steps])

    for feat in GEOMETRY_FEATURES:
        vals = []
        for s in common_steps:
            v = geo_by_step[s].get(feat)
            vals.append(v if v is not None else np.nan)
        result[feat] = np.array(vals)

    return result


# ── OLS Helpers ───────────────────────────────────────────────────────────

def ols_fit(X, y):
    """Ordinary least squares: y = Xβ + ε. Returns β, residuals, RSS."""
    # Add intercept
    n = X.shape[0]
    X1 = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None, np.inf
    resid = y - X1 @ beta
    rss = np.sum(resid ** 2)
    return beta, resid, rss


def aic(rss, n, k):
    """Akaike Information Criterion. k = number of parameters including intercept."""
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def f_test_nested(rss_r, rss_u, n, k_r, k_u):
    """F-test for nested models. H0: restricted model is adequate.
    k_r, k_u = number of parameters (including intercept) in restricted/unrestricted.
    """
    q = k_u - k_r  # number of restrictions
    df_u = n - k_u
    if df_u <= 0 or q <= 0 or rss_u <= 0:
        return 0.0, 1.0
    F = ((rss_r - rss_u) / q) / (rss_u / df_u)
    p = 1.0 - stats.f.cdf(F, q, df_u)
    return float(F), float(p)


# ── Test 1: Granger Causality ────────────────────────────────────────────

def granger_test(data, feature, target, max_lag=2):
    """
    Granger causality test: does `feature` help predict `target` beyond
    AR(p) of target + LR?

    Tests lag orders p=1..max_lag, picks best by AIC.
    Returns dict with F, p, AIC values for best lag.
    """
    y_full = data[target]
    x_full = data[feature]
    lr_full = data["lr"]

    # Remove NaN rows
    valid = ~(np.isnan(y_full) | np.isnan(x_full) | np.isnan(lr_full))
    y_full = y_full[valid]
    x_full = x_full[valid]
    lr_full = lr_full[valid]

    best_result = None
    best_aic_u = np.inf

    for p in range(1, max_lag + 1):
        n = len(y_full) - p
        if n < 10:
            continue

        y = y_full[p:]

        # Restricted: AR(p) of target + LR
        X_r_cols = [lr_full[p:]]
        for i in range(1, p + 1):
            X_r_cols.append(y_full[p - i: -i if i < len(y_full) else None])
        # Handle edge: when i == len(y_full)-p, slice is y_full[0:...]
        X_r_cols_fixed = [lr_full[p:]]
        for i in range(1, p + 1):
            X_r_cols_fixed.append(y_full[p - i: len(y_full) - i])
        X_r = np.column_stack(X_r_cols_fixed)

        # Unrestricted: AR(p) + LR + lagged geometry
        X_u_cols = list(X_r_cols_fixed)  # copy
        for i in range(1, p + 1):
            X_u_cols.append(x_full[p - i: len(x_full) - i])
        X_u = np.column_stack(X_u_cols)

        _, _, rss_r = ols_fit(X_r, y)
        _, _, rss_u = ols_fit(X_u, y)

        k_r = X_r.shape[1] + 1  # +1 for intercept
        k_u = X_u.shape[1] + 1

        aic_r = aic(rss_r, n, k_r)
        aic_u = aic(rss_u, n, k_u)
        F, p_val = f_test_nested(rss_r, rss_u, n, k_r, k_u)

        if aic_u < best_aic_u:
            best_aic_u = aic_u
            best_result = {
                "lag_order": p,
                "n_obs": int(n),
                "F_stat": round(F, 4),
                "p_value": round(p_val, 6),
                "AIC_restricted": round(aic_r, 2),
                "AIC_unrestricted": round(aic_u, 2),
                "delta_AIC": round(aic_r - aic_u, 2),
                "RSS_restricted": round(float(rss_r), 6),
                "RSS_unrestricted": round(float(rss_u), 6),
            }

    return best_result


def run_test1(all_data):
    """Run Granger causality for all (run, feature, target) combinations."""
    print("\n" + "=" * 70)
    print("TEST 1: Granger Causality")
    print("=" * 70)
    print("H0: geometry does NOT help predict target beyond AR(p) + LR")
    print()

    results = {}

    for target in TARGETS:
        results[target] = {}
        for feat in TOP_FEATURES:
            results[target][feat] = {}
            for run_dir, label, data in zip(RUN_DIRS, RUN_LABELS, all_data):
                if data is None:
                    continue
                res = granger_test(data, feat, target, max_lag=2)
                if res is None:
                    continue
                results[target][feat][label] = res

                sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
                print(f"  {feat:15s} → {target:15s} [{label:15s}]  "
                      f"F={res['F_stat']:7.2f}  p={res['p_value']:.4f} {sig:3s}  "
                      f"ΔAIC={res['delta_AIC']:+.1f}  lag={res['lag_order']}")

    # Summary
    print("\n--- Summary ---")
    for target in TARGETS:
        for feat in TOP_FEATURES:
            n_sig = sum(1 for r in results[target][feat].values() if r["p_value"] < 0.05)
            n_total = len(results[target][feat])
            print(f"  {feat:15s} → {target:15s}: {n_sig}/{n_total} runs significant (p<0.05)")

    return results


# ── Test 2: Block Permutation ─────────────────────────────────────────────

def partial_corr(x, y, z):
    """Partial correlation of x, y controlling for z (1D arrays)."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 5:
        return np.nan

    # Residualize both on z
    z1 = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.lstsq(z1, x, rcond=None)[0]
    by = np.linalg.lstsq(z1, y, rcond=None)[0]
    rx = x - z1 @ bx
    ry = y - z1 @ by

    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def circular_block_permute(series, block_size, rng):
    """Circular block permutation of a time series.

    Randomly shifts the series by a random offset, preserving autocorrelation.
    """
    n = len(series)
    # Random circular shift in units of block_size
    n_blocks = max(1, n // block_size)
    shift = rng.integers(1, n)  # shift by 1..n-1 elements
    return np.roll(series, shift)


def block_permutation_test(data, feature, target, lag=2, block_size=5,
                           n_perms=10000, seed=42):
    """
    Block permutation test for partial correlation.

    Computes partial_corr(geometry(t), target(t+lag) | LR(t+lag)),
    then permutes geometry with circular block bootstrap to get null distribution.
    """
    x = data[feature]
    y = data[target]
    lr = data["lr"]

    # Apply lag: geometry(t) vs target(t+lag)
    if lag > 0:
        x_lagged = x[:-lag]
        y_lagged = y[lag:]
        lr_lagged = lr[lag:]
    else:
        x_lagged = x
        y_lagged = y
        lr_lagged = lr

    # Remove NaN
    valid = ~(np.isnan(x_lagged) | np.isnan(y_lagged) | np.isnan(lr_lagged))
    x_clean = x_lagged[valid]
    y_clean = y_lagged[valid]
    lr_clean = lr_lagged[valid]

    if len(x_clean) < 10:
        return None

    # Observed statistic
    observed_r = partial_corr(x_clean, y_clean, lr_clean)

    # Null distribution via circular block permutation
    rng = np.random.default_rng(seed)
    null_rs = np.empty(n_perms)
    for i in range(n_perms):
        x_perm = circular_block_permute(x_clean, block_size, rng)
        null_rs[i] = partial_corr(x_perm, y_clean, lr_clean)

    # Two-sided p-value
    p_value = np.mean(np.abs(null_rs) >= np.abs(observed_r))

    return {
        "observed_r": round(float(observed_r), 4),
        "p_perm": round(float(p_value), 4),
        "null_mean": round(float(np.mean(null_rs)), 4),
        "null_std": round(float(np.std(null_rs)), 4),
        "null_95th": round(float(np.percentile(np.abs(null_rs), 95)), 4),
        "null_99th": round(float(np.percentile(np.abs(null_rs), 99)), 4),
        "n_obs": int(len(x_clean)),
        "lag": lag,
        "block_size": block_size,
        "n_perms": n_perms,
    }


def run_test2(all_data):
    """Run block permutation for top features × targets."""
    print("\n" + "=" * 70)
    print("TEST 2: Block Permutation Test")
    print("=" * 70)
    print("H0: partial correlation (geometry, target | LR) = 0")
    print("Null distribution via circular block permutation (B=5, 10K perms)")
    print()

    results = {}

    # Test at lag=2 (400 steps) for val_loss, lag=0 for probe_ood
    lag_map = {"val_loss": 2, "probe_ood_acc": 0}

    for target in TARGETS:
        lag = lag_map[target]
        results[target] = {}
        for feat in TOP_FEATURES:
            results[target][feat] = {}
            for run_dir, label, data in zip(RUN_DIRS, RUN_LABELS, all_data):
                if data is None:
                    continue
                res = block_permutation_test(data, feat, target, lag=lag)
                if res is None:
                    continue
                results[target][feat][label] = res

                sig = "***" if res["p_perm"] < 0.001 else "**" if res["p_perm"] < 0.01 else "*" if res["p_perm"] < 0.05 else ""
                print(f"  {feat:15s} → {target:15s} [{label:15s}]  "
                      f"r={res['observed_r']:+.3f}  p_perm={res['p_perm']:.4f} {sig:3s}  "
                      f"null_95={res['null_95th']:.3f}  lag={res['lag']}")

    # Summary
    print("\n--- Summary ---")
    for target in TARGETS:
        for feat in TOP_FEATURES:
            n_sig = sum(1 for r in results[target][feat].values() if r["p_perm"] < 0.05)
            n_total = len(results[target][feat])
            print(f"  {feat:15s} → {target:15s}: {n_sig}/{n_total} runs significant (p_perm<0.05)")

    return results


# ── Test 3: First-Difference Correlations ─────────────────────────────────

def effective_dof(series):
    """Compute effective degrees of freedom using Bartlett's formula.

    n_eff = n * (1 - ρ₁) / (1 + ρ₁)
    where ρ₁ is lag-1 autocorrelation.
    """
    n = len(series)
    if n < 5:
        return n
    mean = np.mean(series)
    centered = series - mean
    var = np.sum(centered ** 2)
    if var < 1e-15:
        return n
    rho1 = np.sum(centered[:-1] * centered[1:]) / var
    rho1 = np.clip(rho1, -0.99, 0.99)
    n_eff = n * (1 - rho1) / (1 + rho1)
    return max(4, n_eff)  # Floor at 4


def first_diff_correlation(data, feature, target, max_lag=5):
    """
    Correlate Δgeometry(t) with Δtarget(t+lag) using first differences.
    Also computes partial correlation controlling for ΔLR.
    Reports effective d.f. via Bartlett's formula.
    """
    x = data[feature]
    y = data[target]
    lr = data["lr"]

    # First differences
    dx = np.diff(x)
    dy = np.diff(y)
    dlr = np.diff(lr)

    # Remove NaN from differences
    valid = ~(np.isnan(dx) | np.isnan(dy) | np.isnan(dlr))
    dx = dx[valid]
    dy = dy[valid]
    dlr = dlr[valid]

    results_by_lag = []

    for lag in range(0, max_lag + 1):
        if lag > 0:
            dx_lag = dx[:-lag]
            dy_lag = dy[lag:]
            dlr_lag = dlr[lag:]
        else:
            dx_lag = dx
            dy_lag = dy
            dlr_lag = dlr

        n = len(dx_lag)
        if n < 8:
            continue

        # Raw correlation
        r_raw, _ = stats.pearsonr(dx_lag, dy_lag)

        # Partial correlation controlling for ΔLR
        r_partial = partial_corr(dx_lag, dy_lag, dlr_lag)

        # Effective d.f. for the raw correlation
        n_eff_x = effective_dof(dx_lag)
        n_eff_y = effective_dof(dy_lag)
        n_eff = min(n_eff_x, n_eff_y)

        # t-test with effective d.f.
        df = n_eff - 2
        if df < 2 or abs(r_raw) >= 1.0:
            p_adj = 1.0
            t_stat = 0.0
        else:
            t_stat = r_raw * math.sqrt(df / (1 - r_raw ** 2))
            p_adj = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # Same for partial
        if df < 2 or abs(r_partial) >= 1.0:
            p_partial = 1.0
        else:
            t_p = r_partial * math.sqrt(df / (1 - r_partial ** 2))
            p_partial = 2 * (1 - stats.t.cdf(abs(t_p), df))

        results_by_lag.append({
            "lag": lag,
            "lag_steps": lag * 200,
            "n_obs": int(n),
            "n_eff": round(float(n_eff), 1),
            "r_raw": round(float(r_raw), 4),
            "r_partial": round(float(r_partial), 4),
            "t_stat": round(float(t_stat), 3),
            "p_adjusted": round(float(p_adj), 4),
            "p_partial_adjusted": round(float(p_partial), 4),
        })

    return results_by_lag


def run_test3(all_data):
    """Run first-difference correlations for all combinations."""
    print("\n" + "=" * 70)
    print("TEST 3: First-Difference Correlations")
    print("=" * 70)
    print("Δgeometry(t) vs Δtarget(t+lag), with Bartlett-corrected d.f.")
    print()

    results = {}

    for target in TARGETS:
        results[target] = {}
        for feat in TOP_FEATURES:
            results[target][feat] = {}
            for run_dir, label, data in zip(RUN_DIRS, RUN_LABELS, all_data):
                if data is None:
                    continue
                res = first_diff_correlation(data, feat, target, max_lag=5)
                if not res:
                    continue
                results[target][feat][label] = res

                # Find best lag by |r_partial|
                best = max(res, key=lambda r: abs(r["r_partial"]))
                sig = "***" if best["p_partial_adjusted"] < 0.001 else "**" if best["p_partial_adjusted"] < 0.01 else "*" if best["p_partial_adjusted"] < 0.05 else ""
                print(f"  Δ{feat:14s} → Δ{target:14s} [{label:15s}]  "
                      f"best_lag={best['lag']}({best['lag_steps']}st)  "
                      f"r_partial={best['r_partial']:+.3f}  "
                      f"n_eff={best['n_eff']:.0f}  "
                      f"p_adj={best['p_partial_adjusted']:.4f} {sig}")

    # Summary
    print("\n--- Summary ---")
    for target in TARGETS:
        for feat in TOP_FEATURES:
            n_sig = 0
            n_total = 0
            for label, res_list in results[target][feat].items():
                best = max(res_list, key=lambda r: abs(r["r_partial"]))
                n_total += 1
                if best["p_partial_adjusted"] < 0.05:
                    n_sig += 1
            print(f"  Δ{feat:14s} → Δ{target:14s}: {n_sig}/{n_total} runs significant (p_adj<0.05)")

    return results


# ── Test 4: Reheating Cross-Sectional ─────────────────────────────────────

def load_parent_metrics(run_dir):
    """Load pilot_metrics.json from parent run directory."""
    rd = Path(run_dir) if Path(run_dir).exists() else BASE_DIR / run_dir
    met_path = rd / "pilot_metrics.json"
    if met_path.exists():
        return json.load(open(met_path))
    return None


def get_prev_best_pood(parent_metrics, ckpt_step):
    """Get the historical best p_ood up to ckpt_step from parent training."""
    if not parent_metrics:
        return None
    best = 0.0
    for m in parent_metrics:
        if m["step"] <= ckpt_step:
            best = max(best, m["probe_ood_acc"])
    return best


def run_test4():
    """
    Cross-sectional test: geometry at checkpoint t predicts reheat recovery.
    Requires Phase 3 reheat data (reheat_summary.csv).

    Targets:
      G = peak_reheat_pood - p0  (raw gain from reheat start)
      R = (peak_reheat_pood - p0) / (prev_best - p0)  (recovery ratio vs historical peak)
      D = p_after_warmup - p0  (early warmup gain)
    """
    print("\n" + "=" * 70)
    print("TEST 4: Reheating Cross-Sectional Prediction")
    print("=" * 70)

    reheat_csv = Path("runs/reheat_sweep/reheat_summary.csv")
    if not reheat_csv.exists():
        print("  ⚠ reheat_summary.csv not found — run Phase 3 first:")
        print("    python geometry_sweep_reheat.py --beta2s 0.95,0.80 \\")
        print("      --ckpts 1000,4000,10000 --lrs 0.001,0.0006,0.0003 --lams 2.0,4.0")
        return None

    import csv
    with open(reheat_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 5:
        print(f"  ⚠ Only {len(rows)} rows in reheat_summary.csv — need more data")
        return None

    print(f"  Loaded {len(rows)} reheat conditions\n")

    # Load geometry and parent metrics for each parent run
    geo_cache = {}
    parent_met_cache = {}
    for row in rows:
        # Determine parent run dir from source path
        source = row.get("source", "")
        # source is like "runs/beta2_ablation/.../reheat_ckptXXXX_lrY_lamZ"
        # parent is everything up to the reheat_ subdirectory
        parent_dir = str(Path(source).parent) if "reheat_" in source else source
        run_key = parent_dir

        if run_key not in geo_cache:
            rd = Path(run_key) if Path(run_key).exists() else BASE_DIR / run_key
            geo_path = rd / "causal_geometry.json"
            if geo_path.exists():
                geo = json.load(open(geo_path))
                geo_cache[run_key] = {w["step"]: w for w in geo["windows"]}
            # Also load parent training metrics for historical best
            parent_met_cache[run_key] = load_parent_metrics(run_key)

        # Store parent dir in row for later lookup
        row["_parent_dir"] = run_key

    # Build feature matrix
    REHEAT_FEATURES = ["PC1_roll", "k95_roll", "drift_speed", "a_speed", "r_speed", "kappa"]

    X_rows = []
    G_vals = []
    R_vals = []
    D_vals = []
    labels = []
    detail_rows = []

    for row in rows:
        ckpt_step = int(row.get("ckpt_step", row.get("ckpt", 0)))
        run_key = row.get("_parent_dir", "")

        if run_key not in geo_cache:
            continue

        geo_at_step = geo_cache[run_key]

        # Find closest geometry window to ckpt_step
        available = sorted(geo_at_step.keys())
        closest = min(available, key=lambda s: abs(s - ckpt_step))
        if abs(closest - ckpt_step) > 100:
            continue

        w = geo_at_step[closest]
        feat_vals = []
        skip = False
        for f in REHEAT_FEATURES:
            v = w.get(f)
            if v is None:
                skip = True
                break
            feat_vals.append(float(v))
        if skip:
            continue

        G = float(row.get("G", 0))
        D = float(row.get("D", 0))
        p0 = float(row.get("p0", 0))
        peak_pood = float(row.get("peak_p_ood", 0))

        # Compute recovery ratio R vs historical best
        prev_best = get_prev_best_pood(parent_met_cache.get(run_key), ckpt_step)
        if prev_best is not None and prev_best > p0 + 0.01:
            R = (peak_pood - p0) / (prev_best - p0)
        else:
            # No loss to recover (p0 ≈ prev_best or prev_best unknown)
            R = 0.0 if G <= 0.001 else float('inf')

        X_rows.append(feat_vals)
        G_vals.append(G)
        R_vals.append(R)
        D_vals.append(D)
        labels.append(f"β2={row.get('beta2','')} ckpt={ckpt_step} lr={row.get('lr','')}")
        detail_rows.append({
            "label": labels[-1],
            "lam": row.get("lam", ""),
            "p0": round(p0, 4),
            "peak_reheat": round(peak_pood, 4),
            "prev_best": round(prev_best, 4) if prev_best is not None else None,
            "G": round(G, 4),
            "R": round(R, 4) if R != float('inf') else "inf",
        })

    # Print recovery details
    print("  Recovery details:")
    print(f"  {'label':45s} {'p0':>6s} {'prev_best':>9s} {'peak_rht':>8s} {'G':>7s} {'R':>6s}")
    for d in detail_rows:
        r_str = f"{d['R']:.2f}" if d['R'] != 'inf' else "inf"
        pb_str = f"{d['prev_best']:.3f}" if d['prev_best'] is not None else "N/A"
        print(f"  {d['label']:45s} {d['p0']:6.3f} {pb_str:>9s} {d['peak_reheat']:8.3f} {d['G']:+7.4f} {r_str:>6s}")
    print()

    if len(X_rows) < 5:
        print(f"  ⚠ Only {len(X_rows)} matched geometry-reheat pairs — need more data")
        return None

    X = np.array(X_rows)
    G = np.array(G_vals)
    R = np.array(R_vals)
    D = np.array(D_vals)
    n = len(X)

    # Filter out inf R values for the R target
    R_valid = np.isfinite(R)

    print(f"  Feature matrix: {n} × {len(REHEAT_FEATURES)}")
    print(f"  Features: {REHEAT_FEATURES}")
    print(f"  R valid (finite): {R_valid.sum()}/{n}\n")

    results = {}

    targets_list = [("G", G, np.ones(n, dtype=bool)),
                    ("R", R, R_valid),
                    ("D", D, np.ones(n, dtype=bool))]

    for target_name, target_vals, valid_mask in targets_list:
        X_t = X[valid_mask]
        y_t = target_vals[valid_mask]
        n_t = len(y_t)

        if n_t < 5:
            print(f"  --- Target: {target_name} --- SKIP (only {n_t} valid points)")
            continue
        print(f"  --- Target: {target_name} (n={n_t}) ---")

        # Univariate correlations
        uni_corrs = {}
        for i, feat in enumerate(REHEAT_FEATURES):
            r, p = stats.pearsonr(X_t[:, i], y_t)
            uni_corrs[feat] = {"r": round(float(r), 3), "p": round(float(p), 4)}
            sig = "*" if p < 0.05 else ""
            print(f"    {feat:15s}: r={r:+.3f}  p={p:.4f} {sig}")

        # LOOCV R²
        y_pred_loo = np.empty(n_t)
        for i in range(n_t):
            X_train = np.delete(X_t, i, axis=0)
            y_train = np.delete(y_t, i)
            X_test = X_t[i:i+1]

            # Standardize
            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0)
            sd[sd < 1e-10] = 1.0
            X_train_s = (X_train - mu) / sd
            X_test_s = (X_test - mu) / sd

            beta, _, _ = ols_fit(X_train_s, y_train)
            if beta is None:
                y_pred_loo[i] = np.mean(y_train)
            else:
                X_test_1 = np.column_stack([np.ones(1), X_test_s])
                y_pred_loo[i] = X_test_1 @ beta

        ss_res = np.sum((y_t - y_pred_loo) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2_loo = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Full-data R² for comparison
        mu = X_t.mean(axis=0)
        sd = X_t.std(axis=0)
        sd[sd < 1e-10] = 1.0
        X_s = (X_t - mu) / sd
        beta_full, resid_full, rss_full = ols_fit(X_s, y_t)
        r2_full = 1 - rss_full / ss_tot if ss_tot > 0 else 0

        print(f"    Full R² = {r2_full:.3f},  LOOCV R² = {r2_loo:.3f}")
        print()

        results[target_name] = {
            "univariate": uni_corrs,
            "R2_full": round(float(r2_full), 4),
            "R2_LOOCV": round(float(r2_loo), 4),
            "n": int(n_t),
        }

    # Also store recovery details
    results["recovery_details"] = detail_rows

    return results


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_granger(results):
    """Heatmap of Granger F-stats colored by p-value."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, TARGETS):
        # Build matrix: features × runs
        f_vals = np.full((len(TOP_FEATURES), len(RUN_LABELS)), np.nan)
        p_vals = np.full((len(TOP_FEATURES), len(RUN_LABELS)), np.nan)

        for i, feat in enumerate(TOP_FEATURES):
            for j, label in enumerate(RUN_LABELS):
                if label in results.get(target, {}).get(feat, {}):
                    r = results[target][feat][label]
                    f_vals[i, j] = r["F_stat"]
                    p_vals[i, j] = r["p_value"]

        # Color by -log10(p)
        neg_log_p = -np.log10(np.clip(p_vals, 1e-10, 1))

        im = ax.imshow(neg_log_p, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=4)

        # Annotate with F-stat and stars
        for i in range(len(TOP_FEATURES)):
            for j in range(len(RUN_LABELS)):
                if not np.isnan(f_vals[i, j]):
                    p = p_vals[i, j]
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ax.text(j, i, f"F={f_vals[i,j]:.1f}\n{stars}",
                           ha="center", va="center", fontsize=9,
                           color="white" if neg_log_p[i,j] > 2 else "black")

        ax.set_xticks(range(len(RUN_LABELS)))
        ax.set_xticklabels([l.replace(" ", "\n") for l in RUN_LABELS], fontsize=8)
        ax.set_yticks(range(len(TOP_FEATURES)))
        ax.set_yticklabels(TOP_FEATURES)
        ax.set_title(f"Granger: geometry → {target}\n(color = -log₁₀ p)")
        plt.colorbar(im, ax=ax, label="-log₁₀(p)")

    # Add significance thresholds as text
    fig.text(0.5, 0.01, "Significance: * p<0.05  ** p<0.01  *** p<0.001",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = RESULTS_DIR / "rigorous_granger.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def plot_permutation(results):
    """Null distributions with observed values marked."""
    if not HAS_MPL:
        return

    # Just show summary stats as a bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target in zip(axes, TARGETS):
        feat_labels = []
        observed_rs = []
        null_95s = []
        p_perms = []

        for feat in TOP_FEATURES:
            for label in RUN_LABELS:
                if label in results.get(target, {}).get(feat, {}):
                    r = results[target][feat][label]
                    feat_labels.append(f"{feat}\n{label}")
                    observed_rs.append(abs(r["observed_r"]))
                    null_95s.append(r["null_95th"])
                    p_perms.append(r["p_perm"])

        x = np.arange(len(feat_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, observed_rs, width, label="|observed r|",
                       color=["green" if p < 0.05 else "salmon" for p in p_perms])
        bars2 = ax.bar(x + width/2, null_95s, width, label="null 95th pct",
                       color="lightgray", edgecolor="gray")

        # Add p-value annotations
        for i, (obs, p) in enumerate(zip(observed_rs, p_perms)):
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(i - width/2, obs + 0.02, stars, ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(feat_labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("|partial correlation|")
        ax.set_title(f"Block Permutation: → {target}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "rigorous_permutation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_firstdiff(results):
    """First-difference correlation profiles across lags."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(len(TOP_FEATURES), 2, figsize=(14, 4 * len(TOP_FEATURES)))

    for row, feat in enumerate(TOP_FEATURES):
        for col, target in enumerate(TARGETS):
            ax = axes[row, col]

            for label in RUN_LABELS:
                res_list = results.get(target, {}).get(feat, {}).get(label, [])
                if not res_list:
                    continue

                lags = [r["lag"] for r in res_list]
                r_partials = [r["r_partial"] for r in res_list]
                p_vals = [r["p_partial_adjusted"] for r in res_list]

                ax.plot(lags, r_partials, "o-", label=label, markersize=5)

                # Mark significant points
                for l, rp, p in zip(lags, r_partials, p_vals):
                    if p < 0.05:
                        ax.plot(l, rp, "o", markersize=10, markerfacecolor="none",
                               markeredgecolor="red", markeredgewidth=2)

            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Lag (units of 200 steps)")
            ax.set_ylabel("Δ partial r (|LR)")
            ax.set_title(f"Δ{feat} → Δ{target}")
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = RESULTS_DIR / "rigorous_firstdiff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rigorous statistical tests")
    parser.add_argument("--tests", default="1,2,3",
                       help="Comma-separated test numbers (1,2,3,4,all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    tests = args.tests
    if tests == "all":
        test_nums = {1, 2, 3, 4}
    else:
        test_nums = {int(t.strip()) for t in tests.split(",")}

    # Load data for within-run tests
    all_data = []
    if test_nums & {1, 2, 3}:
        print("Loading aligned data...")
        for rd, label in zip(RUN_DIRS, RUN_LABELS):
            data = load_aligned_data(rd)
            if data is None:
                print(f"  ⚠ Missing data for {label} ({rd})")
            else:
                print(f"  ✓ {label}: {len(data['steps'])} aligned points")
            all_data.append(data)
        print()

    all_results = {}

    if 1 in test_nums:
        r1 = run_test1(all_data)
        all_results["test1_granger"] = r1
        if HAS_MPL:
            plot_granger(r1)

    if 2 in test_nums:
        r2 = run_test2(all_data)
        all_results["test2_permutation"] = r2
        if HAS_MPL:
            plot_permutation(r2)

    if 3 in test_nums:
        r3 = run_test3(all_data)
        all_results["test3_firstdiff"] = r3
        if HAS_MPL:
            plot_firstdiff(r3)

    if 4 in test_nums:
        r4 = run_test4()
        if r4:
            all_results["test4_reheat"] = r4

    # Save JSON
    out_json = RESULTS_DIR / "rigorous_tests.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)

    if "test1_granger" in all_results:
        print("\nTest 1 (Granger):")
        for target in TARGETS:
            for feat in TOP_FEATURES:
                entries = all_results["test1_granger"].get(target, {}).get(feat, {})
                n_sig = sum(1 for r in entries.values() if r["p_value"] < 0.05)
                n_tot = len(entries)
                verdict = "✓ PASS" if n_sig >= 3 else "~ MIXED" if n_sig >= 2 else "✗ FAIL"
                print(f"  {feat:15s} → {target:15s}: {n_sig}/{n_tot} sig  {verdict}")

    if "test2_permutation" in all_results:
        print("\nTest 2 (Permutation):")
        for target in TARGETS:
            for feat in TOP_FEATURES:
                entries = all_results["test2_permutation"].get(target, {}).get(feat, {})
                n_sig = sum(1 for r in entries.values() if r["p_perm"] < 0.05)
                n_tot = len(entries)
                verdict = "✓ PASS" if n_sig >= 3 else "~ MIXED" if n_sig >= 2 else "✗ FAIL"
                print(f"  {feat:15s} → {target:15s}: {n_sig}/{n_tot} sig  {verdict}")

    if "test3_firstdiff" in all_results:
        print("\nTest 3 (First-Diff):")
        for target in TARGETS:
            for feat in TOP_FEATURES:
                entries = all_results["test3_firstdiff"].get(target, {}).get(feat, {})
                n_sig = 0
                n_tot = 0
                for label, res_list in entries.items():
                    best = max(res_list, key=lambda r: abs(r["r_partial"]))
                    n_tot += 1
                    if best["p_partial_adjusted"] < 0.05:
                        n_sig += 1
                verdict = "✓ PASS" if n_sig >= 3 else "~ MIXED" if n_sig >= 2 else "✗ FAIL"
                print(f"  Δ{feat:14s} → Δ{target:14s}: {n_sig}/{n_tot} sig  {verdict}")


if __name__ == "__main__":
    main()
