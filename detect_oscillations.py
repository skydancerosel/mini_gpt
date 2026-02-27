#!/usr/bin/env python3
"""
Auto-detect oscillation peaks and troughs from pilot_metrics.json.

Outputs oscillation_manifest.json with all step markers needed by
basin_geometry.py, attractor_analysis.py, and directional_probing.py,
so no seed-specific checkpoint steps are ever hardcoded downstream.

Detection parameters are FROZEN CONSTANTS — not CLI args — to prevent
post-hoc tuning / p-hacking across seeds.

If eval_noise.json exists in run-dir (from estimate_eval_noise.py),
switch pairs are tiered by significance:
  Tier A:  delta >= K_A * std(p_ood)   (strongly significant)
  Tier B:  delta >= K_B * std(p_ood)   (significant)
  Tier C:  below K_B                    (noise-level, borderline)

Also selects representative steps for expensive analyses:
  - early peak  (headline replication)
  - mid peak    (strongest mid-range oscillation)
  - late peak   (only if Tier A or B)
  - transition  (one mid trough for basin depth reference)

Usage:
    # Without noise calibration (backwards-compatible):
    python detect_oscillations.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s137/

    # With noise calibration (run estimate_eval_noise.py first):
    python estimate_eval_noise.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s137/
    python detect_oscillations.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s137/
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# FROZEN DETECTION CONSTANTS — do NOT make these CLI args
# ═══════════════════════════════════════════════════════════════════════════
PROMINENCE_THRESHOLD = 0.03
METRIC_NAME = "probe_ood_acc"

# Tiering thresholds (multiples of std(p_ood))
K_A = 3   # Tier A: strongly significant
K_B = 2   # Tier B: significant


def identify_peaks_troughs(metrics, key=METRIC_NAME, min_prominence=PROMINENCE_THRESHOLD):
    """Find local maxima (peaks) and minima (troughs) of a metric.

    Duplicated from attractor_analysis.py to keep this script dependency-free.
    Same algorithm: ±3 step window for prominence, ±1 for local extremum.

    Returns list of (step_idx, step, value, label) tuples.
    """
    vals = [m[key] for m in metrics]
    steps = [m["step"] for m in metrics]
    markers = []

    for i in range(1, len(vals) - 1):
        if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
            left_min = min(vals[max(0, i - 3):i])
            right_min = min(vals[i + 1:min(len(vals), i + 4)])
            prom = vals[i] - max(left_min, right_min)
            if prom >= min_prominence:
                markers.append((i, steps[i], vals[i], "peak"))
        elif vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            left_max = max(vals[max(0, i - 3):i])
            right_max = max(vals[i + 1:min(len(vals), i + 4)])
            prom = min(left_max, right_max) - vals[i]
            if prom >= min_prominence:
                markers.append((i, steps[i], vals[i], "trough"))

    return markers


def get_git_commit():
    """Get current git HEAD hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def build_switch_pairs(peaks, troughs):
    """Match each peak to nearest subsequent trough for switch-pairs.

    Returns list of (peak_step, trough_step) tuples.
    """
    pairs = []
    for ps in peaks:
        for ts in troughs:
            if ts > ps:
                pairs.append((ps, ts))
                break
    return pairs


def build_extended_pairs(markers):
    """Build extended peak-trough pairs for manifold dimensionality (B7).

    Strategy: pair every detected peak with its nearest trough (before or after),
    plus pair consecutive peaks/troughs to capture intra-regime drift.
    All pairs derived from detected markers — NO fallback to seed-42 defaults.
    """
    peaks = sorted([s for _, s, _, l in markers if l == "peak"])
    troughs = sorted([s for _, s, _, l in markers if l == "trough"])
    all_steps = sorted(peaks + troughs)

    pairs = []
    seen = set()

    # 1. Each peak paired with nearest subsequent trough
    for ps in peaks:
        for ts in troughs:
            if ts > ps:
                key = (ps, ts)
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
                break

    # 2. Each peak paired with nearest preceding trough
    for ps in peaks:
        for ts in reversed(troughs):
            if ts < ps:
                key = (ps, ts)
                if key not in seen:
                    pairs.append(key)
                    seen.add(key)
                break

    # 3. Consecutive event pairs (to capture drift within oscillation cycle)
    for i in range(len(all_steps) - 1):
        a, b = all_steps[i], all_steps[i + 1]
        key = (max(a, b), min(a, b))  # (higher_step, lower_step)
        if key not in seen:
            pairs.append(key)
            seen.add(key)

    # Sort chronologically by first element
    pairs.sort(key=lambda p: (p[0], p[1]))
    return pairs


def tier_switch_pairs(switch_pairs, step_pood, noise_std):
    """Classify switch pairs by significance relative to noise floor.

    Returns list of dicts: {peak, trough, delta, tier, k_sigma}
    """
    tiered = []
    for peak, trough in switch_pairs:
        p_peak = step_pood.get(peak, 0)
        p_trough = step_pood.get(trough, 0)
        delta = p_peak - p_trough

        if noise_std > 0:
            k_sigma = delta / noise_std
        else:
            k_sigma = float("inf") if delta > 0 else 0

        if k_sigma >= K_A:
            tier = "A"
        elif k_sigma >= K_B:
            tier = "B"
        else:
            tier = "C"

        tiered.append({
            "peak": peak,
            "trough": trough,
            "delta": round(delta, 4),
            "k_sigma": round(k_sigma, 2),
            "tier": tier,
        })

    return tiered


def select_representative_steps(peaks, troughs, tiered_pairs, step_pood, last_step):
    """Select representative steps for expensive analyses.

    Strategy:
      - early_peak:  first peak (headline replication)
      - mid_peak:    peak with largest delta among Tier A/B pairs in mid range
      - late_peak:   latest peak with Tier A or B pair (if any)
      - transition:  one mid-range trough for basin depth reference

    Also selects basin_depth_steps: early peak, mid peak, late step, + transition.
    """
    rep = {}

    # Early peak = first peak (always included)
    if peaks:
        rep["early_peak"] = peaks[0]

    # Mid peak = peak with largest delta among pairs in the middle third
    mid_start = last_step * 0.25
    mid_end = last_step * 0.75
    mid_pairs = [p for p in tiered_pairs
                 if mid_start <= p["peak"] <= mid_end and p["tier"] in ("A", "B")]
    if mid_pairs:
        best_mid = max(mid_pairs, key=lambda p: p["delta"])
        rep["mid_peak"] = best_mid["peak"]
        rep["mid_trough"] = best_mid["trough"]
    elif peaks:
        # Fallback: pick peak closest to midpoint with highest p_ood
        mid_target = last_step * 0.5
        mid_peaks = sorted(peaks, key=lambda s: abs(s - mid_target))
        rep["mid_peak"] = mid_peaks[0]

    # Late peak = latest Tier A/B peak
    late_ab = [p for p in tiered_pairs if p["tier"] in ("A", "B")]
    if late_ab:
        latest = max(late_ab, key=lambda p: p["peak"])
        if latest["peak"] != rep.get("early_peak") and latest["peak"] != rep.get("mid_peak"):
            rep["late_peak"] = latest["peak"]

    # Transition = one mid trough
    if troughs:
        mid_troughs = [t for t in troughs if mid_start <= t <= mid_end]
        if mid_troughs:
            rep["transition_trough"] = mid_troughs[len(mid_troughs) // 2]
        else:
            rep["transition_trough"] = troughs[len(troughs) // 2]

    # Basin depth grid: compact set of steps
    basin_steps = []
    for key in ["early_peak", "mid_peak", "late_peak", "transition_trough"]:
        if key in rep and rep[key] not in basin_steps:
            basin_steps.append(rep[key])
    # Always include last step (late LM baseline)
    if last_step not in basin_steps:
        basin_steps.append(last_step)
    basin_steps.sort()
    rep["basin_depth_steps"] = basin_steps

    # Priority pairs for directional probing (max 3)
    priority_pairs = []
    # 1. Headline: early peak → first trough
    early_pair = [p for p in tiered_pairs if p["peak"] == rep.get("early_peak")]
    if early_pair:
        priority_pairs.append(early_pair[0])
    # 2. Mid pair
    if mid_pairs:
        best = max(mid_pairs, key=lambda p: p["delta"])
        if best not in priority_pairs:
            priority_pairs.append(best)
    # 3. Late pair only if Tier A/B
    late_pairs = [p for p in tiered_pairs
                  if p["peak"] > mid_end and p["tier"] in ("A", "B")]
    if late_pairs:
        best_late = max(late_pairs, key=lambda p: p["delta"])
        if best_late not in priority_pairs:
            priority_pairs.append(best_late)
    rep["priority_pairs"] = [{"peak": p["peak"], "trough": p["trough"]}
                              for p in priority_pairs[:3]]

    return rep


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect oscillation peaks/troughs → oscillation_manifest.json"
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to pilot run directory with pilot_metrics.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "pilot_metrics.json"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    print(f"Loaded {len(metrics)} eval points from {metrics_path}")
    print(f"Detection constants: metric={METRIC_NAME}, prominence_threshold={PROMINENCE_THRESHOLD}")

    # Step → p_ood lookup
    step_pood = {m["step"]: m[METRIC_NAME] for m in metrics}

    # Load noise floor if available
    noise_path = run_dir / "eval_noise.json"
    noise_std = None
    if noise_path.exists():
        with open(noise_path) as f:
            noise = json.load(f)
        noise_std = noise["std_p_ood"]
        print(f"\nNoise calibration loaded from {noise_path}:")
        print(f"  std(p_ood) = {noise_std:.4f}")
        print(f"  Tier A threshold (k={K_A}): delta >= {K_A * noise_std:.4f}")
        print(f"  Tier B threshold (k={K_B}): delta >= {K_B * noise_std:.4f}")
    else:
        print(f"\nNo eval_noise.json found — tiering disabled.")
        print(f"  Run: python estimate_eval_noise.py --run-dir {run_dir}")

    # Detect oscillations
    markers = identify_peaks_troughs(metrics)

    peaks = sorted([s for _, s, _, l in markers if l == "peak"])
    troughs = sorted([s for _, s, _, l in markers if l == "trough"])

    print(f"\nDetected {len(peaks)} peaks: {peaks}")
    print(f"Detected {len(troughs)} troughs: {troughs}")
    for _, s, v, l in markers:
        print(f"  {l:>6s} @ step {s}: {METRIC_NAME}={v:.4f}")

    # Fail loudly if no oscillations detected
    if len(peaks) == 0:
        print(f"\nERROR: No peaks detected with prominence >= {PROMINENCE_THRESHOLD}.",
              file=sys.stderr)
        print("This seed may not exhibit oscillations. Try a different seed.", file=sys.stderr)
        sys.exit(1)

    if len(troughs) == 0:
        print(f"\nERROR: Peaks detected but no troughs with prominence >= {PROMINENCE_THRESHOLD}.",
              file=sys.stderr)
        sys.exit(1)

    # Build derived quantities
    last_step = metrics[-1]["step"]
    late_lm = [last_step]

    switch_pairs = build_switch_pairs(peaks, troughs)
    switch_pairs_str = ",".join(f"{p}:{t}" for p, t in switch_pairs)

    extended_pairs = build_extended_pairs(markers)

    # Tier switch pairs (if noise calibration available)
    tiered = None
    representative = None
    if noise_std is not None:
        tiered = tier_switch_pairs(switch_pairs, step_pood, noise_std)

        print(f"\nSwitch pair tiering (std={noise_std:.4f}):")
        for t in tiered:
            print(f"  {t['tier']}  peak {t['peak']:5d} → trough {t['trough']:5d}  "
                  f"delta={t['delta']:+.4f}  ({t['k_sigma']:.1f}σ)")

        n_a = sum(1 for t in tiered if t["tier"] == "A")
        n_b = sum(1 for t in tiered if t["tier"] == "B")
        n_c = sum(1 for t in tiered if t["tier"] == "C")
        print(f"  Summary: {n_a} Tier A, {n_b} Tier B, {n_c} Tier C")

        # Select representative steps
        representative = select_representative_steps(
            peaks, troughs, tiered, step_pood, last_step
        )
        print(f"\nRepresentative steps:")
        for k, v in representative.items():
            print(f"  {k}: {v}")
    else:
        # No noise info — still select basic representatives by position
        representative = select_representative_steps(
            peaks, troughs,
            # Without noise info, treat all pairs as Tier B
            [{"peak": p, "trough": t, "delta": step_pood.get(p, 0) - step_pood.get(t, 0),
              "k_sigma": float("nan"), "tier": "B"}
             for p, t in switch_pairs],
            step_pood, last_step
        )

    # Build manifest
    manifest = {
        "peaks": peaks,
        "troughs": troughs,
        "late_lm": late_lm,
        "switch_pairs": [{"peak": p, "trough": t} for p, t in switch_pairs],
        "switch_pairs_str": switch_pairs_str,
        "extended_pairs": [{"peak": p, "trough": t} for p, t in extended_pairs],
        "markers": [
            {"step_idx": i, "step": s, "value": round(v, 4), "label": l}
            for i, s, v, l in markers
        ],
        "provenance": {
            "prominence_threshold": PROMINENCE_THRESHOLD,
            "metric_name": METRIC_NAME,
            "detection_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "metrics_file": str(metrics_path),
            "n_eval_points": len(metrics),
        },
    }

    # Add noise-calibrated fields (additive — backwards compatible)
    if tiered is not None:
        manifest["tiered_pairs"] = tiered
        manifest["noise_floor"] = {
            "std_p_ood": noise_std,
            "k_a": K_A,
            "k_b": K_B,
            "source": str(noise_path),
        }
    if representative is not None:
        manifest["representative"] = representative

    # Save manifest
    out_path = run_dir / "oscillation_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {out_path}")
    print(f"  peaks:          {peaks}")
    print(f"  troughs:        {troughs}")
    print(f"  late_lm:        {late_lm}")
    print(f"  switch_pairs:   {switch_pairs_str}")
    print(f"  extended_pairs: {len(extended_pairs)} pairs")
    if representative:
        print(f"  representative: {representative}")


if __name__ == "__main__":
    main()
