#!/usr/bin/env python3
"""
Compute causal rolling geometry metrics from training checkpoints.

Right-aligned sliding SVD (W=10 windows) on consecutive parameter deltas.
Produces causal_geometry.json compatible with geometry_rigorous.py.

Memory-efficient: streams checkpoints and maintains a circular buffer of deltas.
Uses Gram matrix trick (W×W eigendecomp) instead of full W×D SVD.

Usage:
    python causal_geometry.py --run-dir ../runs/scale_124M/pilot_124M_b20.95_s42
    python causal_geometry.py --run-dir <path> --W 10 --trunk-only
"""

import argparse
import json
import re
from pathlib import Path
from collections import deque

import numpy as np
import torch


def discover_checkpoints(run_dir):
    """Find all ckpt_XXXXXX.pt files, return sorted list of (step, path)."""
    ckpts = []
    for p in Path(run_dir).glob("ckpt_*.pt"):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def flatten_params(state_dict, trunk_only=False):
    """Flatten model parameters into a single 1D float32 numpy array.

    If trunk_only, exclude embedding layers (tok_emb, pos_emb).
    """
    parts = []
    for name in sorted(state_dict.keys()):
        if trunk_only and ("tok_emb" in name or "pos_emb" in name):
            continue
        parts.append(state_dict[name].cpu().float().numpy().ravel())
    return np.concatenate(parts)


def compute_window_metrics(deltas, prev_v1=None):
    """Compute geometry metrics for a window of W delta vectors.

    Memory-efficient: computes Gram matrix without creating full W×D matrix.

    Args:
        deltas: list of W numpy vectors (each shape [D,], float32)
        prev_v1: previous window's first right singular vector (for gamma/kappa)

    Returns:
        (metrics_dict, current_v1) or (None, None) if degenerate.
    """
    W = len(deltas)

    # Gram matrix: G[i,j] = dot(delta_i, delta_j), computed in float64 for accuracy
    G = np.zeros((W, W), dtype=np.float64)
    for i in range(W):
        for j in range(i, W):
            G[i, j] = np.dot(deltas[i].astype(np.float64), deltas[j].astype(np.float64))
            G[j, i] = G[i, j]

    # Eigendecompose G (real symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    sigma = np.sqrt(eigenvalues)
    sigma_sq = eigenvalues  # sigma^2 = eigenvalues of Gram matrix
    total_var = sigma_sq.sum()

    if total_var < 1e-30:
        return None, None

    # PC1_roll: % variance explained by first singular value
    PC1_roll = float(sigma_sq[0] / total_var * 100)

    # k95_roll: number of components for 95% variance
    cumvar = np.cumsum(sigma_sq) / total_var
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)
    k95 = min(k95, W)

    # First right singular vector: v1 = sum(u1[i] * delta_i) / sigma_1
    u1 = eigenvectors[:, 0]
    v1 = None
    if sigma[0] > 1e-15:
        v1 = np.zeros_like(deltas[0], dtype=np.float64)
        for i in range(W):
            v1 += u1[i] * deltas[i].astype(np.float64)
        v1 /= sigma[0]
        v1_norm = np.linalg.norm(v1)
        if v1_norm > 1e-15:
            v1 /= v1_norm
        else:
            v1 = None

    # gamma/kappa: direction stability between consecutive windows
    gamma = None
    kappa = None
    if prev_v1 is not None and v1 is not None:
        cos_sim = abs(float(np.dot(v1, prev_v1)))
        gamma = min(cos_sim, 1.0)
        kappa = 1.0 - gamma

    # align_u: alignment of last delta with PC1
    last_delta = deltas[-1]
    if v1 is not None:
        norm_last = np.linalg.norm(last_delta)
        if norm_last > 1e-15:
            align_u = abs(float(np.dot(last_delta.astype(np.float64), v1) / norm_last))
        else:
            align_u = 0.0
    else:
        align_u = 0.0

    # Mean delta and speed decomposition
    mean_delta = np.zeros_like(deltas[0], dtype=np.float64)
    for d in deltas:
        mean_delta += d.astype(np.float64)
    mean_delta /= W

    drift_speed = float(np.linalg.norm(mean_delta))

    if v1 is not None:
        a_proj = float(np.dot(mean_delta, v1))
        a_speed = abs(a_proj)
        residual = mean_delta - a_proj * v1
        r_speed = float(np.linalg.norm(residual))
    else:
        a_speed = drift_speed
        r_speed = 0.0

    # Convert v1 back to float32 for storage efficiency
    v1_out = v1.astype(np.float32) if v1 is not None else None

    return {
        "PC1_roll": PC1_roll,
        "k95_roll": k95,
        "gamma": gamma,
        "kappa": kappa,
        "align_u": align_u,
        "drift_speed": drift_speed,
        "a_speed": a_speed,
        "r_speed": r_speed,
    }, v1_out


def main():
    parser = argparse.ArgumentParser(description="Compute causal rolling geometry")
    parser.add_argument("--run-dir", required=True, help="Path to run directory with checkpoints")
    parser.add_argument("--W", type=int, default=10, help="Window size (number of deltas)")
    parser.add_argument("--trunk-only", action="store_true",
                        help="Exclude embedding layers from geometry")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    W = args.W

    print(f"Run dir: {run_dir}")
    print(f"Window size: {W}")
    print(f"Trunk only: {args.trunk_only}")

    # Discover checkpoints
    ckpts = discover_checkpoints(run_dir)
    print(f"Found {len(ckpts)} checkpoints")
    print(f"  Steps: {ckpts[0][0]} ... {ckpts[-1][0]}")

    if len(ckpts) < W + 1:
        print(f"Need at least {W+1} checkpoints for W={W} windows, got {len(ckpts)}")
        return

    # Stream through checkpoints, computing deltas
    windows = []
    delta_buffer = deque(maxlen=W)
    prev_v1 = None
    prev_params = None

    for i, (step, ckpt_path) in enumerate(ckpts):
        print(f"  [{i+1}/{len(ckpts)}] step {step} ...", end="", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        params = flatten_params(state["model_state_dict"], trunk_only=args.trunk_only)
        del state

        if prev_params is not None:
            delta = params - prev_params
            delta_buffer.append(delta)
            print(f" ||δ||={np.linalg.norm(delta):.2f}", end="")

        prev_params = params

        # Once we have W deltas, compute window metrics
        if len(delta_buffer) == W:
            metrics, v1 = compute_window_metrics(list(delta_buffer), prev_v1)
            if metrics is not None:
                metrics["step"] = step
                windows.append(metrics)
                prev_v1 = v1
                print(f"  → drift={metrics['drift_speed']:.2f} PC1={metrics['PC1_roll']:.1f}%")
            else:
                print(f"  → degenerate (skipped)")
        else:
            print(f"  (buffer: {len(delta_buffer)}/{W})")

    # Compute summaries
    summaries = {}
    if windows:
        steps = [w["step"] for w in windows]
        for cutoff_label, cutoff in [("0_2k", 2000), ("0_10k", max(steps) + 1)]:
            subset = [w for w in windows if w["step"] <= cutoff]
            if subset:
                for key in ["PC1_roll", "align_u", "drift_speed", "kappa"]:
                    vals = [w[key] for w in subset if w[key] is not None]
                    if vals:
                        summaries[f"mean_{key}_{cutoff_label}"] = float(np.mean(vals))

        gammas = [w["gamma"] for w in windows if w["gamma"] is not None]
        if gammas:
            angles = [np.arccos(min(g, 1.0)) for g in gammas]
            summaries["total_turning"] = float(np.sum(angles) * 180 / np.pi)

        # Report drift_speed early mean
        if "mean_drift_speed_0_2k" in summaries:
            summaries["mean_drift_speed_0_2k"] = summaries["mean_drift_speed_0_2k"]

    # Save
    result = {
        "run_dir": str(run_dir),
        "W": W,
        "n_checkpoints": len(ckpts),
        "n_windows": len(windows),
        "first_valid_step": windows[0]["step"] if windows else None,
        "last_valid_step": windows[-1]["step"] if windows else None,
        "summaries": summaries,
        "windows": windows,
    }

    out_path = run_dir / "causal_geometry.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {len(windows)} windows to {out_path}")


if __name__ == "__main__":
    main()
