#!/usr/bin/env python3
"""
Per-layer causal rolling geometry for 124M model.

Computes SVD geometry metrics separately for each transformer block
(and embeddings, final LN) to diagnose whether multi-dimensionality
is within-layer or cross-layer.

If each layer has 1D drift but they point in different directions,
the full-model k95 will be high even though per-layer k95 is low.

Usage:
    python causal_geometry_perlayer.py --run-dir ../runs/scale_124M/pilot_124M_b20.95_s42
"""

import argparse
import json
import re
from pathlib import Path
from collections import deque, OrderedDict

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def discover_checkpoints(run_dir):
    ckpts = []
    for p in Path(run_dir).glob("ckpt_*.pt"):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def partition_state_dict(state_dict):
    """Partition state dict keys into named groups (layers).

    Returns OrderedDict: group_name -> list of param keys.
    """
    groups = OrderedDict()

    # Embeddings
    groups["tok_emb"] = []
    groups["pos_emb"] = []

    # Detect number of blocks
    block_ids = set()
    for k in state_dict.keys():
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            block_ids.add(int(m.group(1)))

    n_blocks = max(block_ids) + 1 if block_ids else 0

    # Per-block: separate attn and mlp
    for i in range(n_blocks):
        groups[f"block{i}_attn"] = []
        groups[f"block{i}_mlp"] = []
        groups[f"block{i}_ln"] = []

    # Final LN
    groups["ln_f"] = []

    # Classify each key
    for k in sorted(state_dict.keys()):
        if "tok_emb" in k:
            groups["tok_emb"].append(k)
        elif "pos_emb" in k:
            groups["pos_emb"].append(k)
        elif "ln_f" in k:
            groups["ln_f"].append(k)
        else:
            m = re.match(r"blocks\.(\d+)\.(.*)", k)
            if m:
                block_i = int(m.group(1))
                subkey = m.group(2)
                if "attn" in subkey:
                    groups[f"block{block_i}_attn"].append(k)
                elif "mlp" in subkey:
                    groups[f"block{block_i}_mlp"].append(k)
                elif "ln" in subkey:
                    groups[f"block{block_i}_ln"].append(k)
                else:
                    # Fallback: put in attn
                    groups[f"block{block_i}_attn"].append(k)

    # Also create combined block groups
    for i in range(n_blocks):
        groups[f"block{i}"] = (groups[f"block{block_i}_attn"] +
                                groups[f"block{block_i}_mlp"] +
                                groups[f"block{block_i}_ln"])

    # Wait, that's wrong - block_i is the last one. Let me fix:
    # Remove the combined groups and redo
    for i in range(n_blocks):
        if f"block{i}" in groups:
            del groups[f"block{i}"]

    for i in range(n_blocks):
        attn_keys = groups[f"block{i}_attn"]
        mlp_keys = groups[f"block{i}_mlp"]
        ln_keys = groups[f"block{i}_ln"]
        groups[f"block{i}"] = attn_keys + mlp_keys + ln_keys

    # Remove empty groups
    groups = OrderedDict((k, v) for k, v in groups.items() if v)

    return groups


def flatten_group(state_dict, keys):
    """Flatten selected parameters into a 1D float32 array."""
    parts = []
    for k in sorted(keys):
        parts.append(state_dict[k].cpu().float().numpy().ravel())
    return np.concatenate(parts)


def compute_window_metrics(deltas):
    """Compute geometry metrics for one window of W deltas."""
    W = len(deltas)

    G = np.zeros((W, W), dtype=np.float64)
    for i in range(W):
        for j in range(i, W):
            G[i, j] = np.dot(deltas[i].astype(np.float64),
                              deltas[j].astype(np.float64))
            G[j, i] = G[i, j]

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 0.0)

    sigma_sq = eigenvalues
    total = sigma_sq.sum()
    if total < 1e-30:
        return None

    PC1 = float(sigma_sq[0] / total * 100)
    cumvar = np.cumsum(sigma_sq) / total
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)
    k95 = min(k95, W)

    # Variance explained per PC
    var_explained = (sigma_sq / total * 100).tolist()

    # Drift speed (norm of mean delta)
    mean_delta = np.zeros_like(deltas[0], dtype=np.float64)
    for d in deltas:
        mean_delta += d.astype(np.float64)
    mean_delta /= W
    drift_speed = float(np.linalg.norm(mean_delta))

    # Per-delta norms (to see if one step dominates)
    delta_norms = [float(np.linalg.norm(d)) for d in deltas]

    return {
        "PC1": PC1,
        "k95": k95,
        "drift_speed": drift_speed,
        "var_explained": var_explained,
        "mean_delta_norm": float(np.mean(delta_norms)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--W", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    W = args.W

    ckpts = discover_checkpoints(run_dir)
    print(f"Found {len(ckpts)} checkpoints")

    if len(ckpts) < W + 1:
        print(f"Need >= {W+1} checkpoints")
        return

    # Discover layer groups from first checkpoint
    state0 = torch.load(ckpts[0][1], map_location="cpu", weights_only=True)
    groups = partition_state_dict(state0["model_state_dict"])

    # We want: per-block combined, plus full model
    # Filter to the groups we care about
    analysis_groups = OrderedDict()
    n_blocks = sum(1 for k in groups if re.match(r"block\d+$", k))

    for i in range(n_blocks):
        key = f"block{i}"
        if key in groups:
            n_params = sum(state0["model_state_dict"][k].numel()
                          for k in groups[key])
            analysis_groups[key] = groups[key]
            print(f"  {key}: {len(groups[key])} tensors, {n_params:,} params")

    # Also add attn-only and mlp-only for a few representative blocks
    for i in [0, n_blocks // 2, n_blocks - 1]:
        for sub in ["attn", "mlp"]:
            key = f"block{i}_{sub}"
            if key in groups and groups[key]:
                n_params = sum(state0["model_state_dict"][k].numel()
                              for k in groups[key])
                analysis_groups[key] = groups[key]
                print(f"  {key}: {len(groups[key])} tensors, {n_params:,} params")

    del state0
    print(f"\nAnalyzing {len(analysis_groups)} groups over {len(ckpts)} checkpoints\n")

    # Stream through checkpoints
    delta_buffers = {name: deque(maxlen=W) for name in analysis_groups}
    prev_flat = {}
    results = {name: [] for name in analysis_groups}

    for ci, (step, ckpt_path) in enumerate(ckpts):
        print(f"  [{ci+1}/{len(ckpts)}] step {step}", end="", flush=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = state["model_state_dict"]
        del state

        for name, keys in analysis_groups.items():
            flat = flatten_group(sd, keys)

            if name in prev_flat:
                delta = flat - prev_flat[name]
                delta_buffers[name].append(delta)

            prev_flat[name] = flat

            if len(delta_buffers[name]) == W:
                metrics = compute_window_metrics(list(delta_buffers[name]))
                if metrics is not None:
                    metrics["step"] = step
                    results[name].append(metrics)

        del sd

        # Print summary for first block
        first_name = list(analysis_groups.keys())[0]
        if results[first_name] and results[first_name][-1]["step"] == step:
            m = results[first_name][-1]
            print(f"  {first_name}: PC1={m['PC1']:.1f}% k95={m['k95']}")
        else:
            print()

    # Save results
    out_path = run_dir / "causal_geometry_perlayer.json"
    # Convert for JSON serialization
    save_data = {}
    for name, windows in results.items():
        save_data[name] = windows
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Group':20s} {'mean_PC1':>8s} {'mean_k95':>8s} {'min_k95':>7s} {'max_k95':>7s} {'n_wins':>6s}")
    print(f"{'='*80}")
    for name, windows in results.items():
        if not windows:
            continue
        pc1s = [w["PC1"] for w in windows]
        k95s = [w["k95"] for w in windows]
        print(f"{name:20s} {np.mean(pc1s):8.1f}% {np.mean(k95s):8.1f} {min(k95s):7d} {max(k95s):7d} {len(windows):6d}")

    # Plot
    if HAS_MPL:
        plot_perlayer(results, n_blocks, run_dir)


def plot_perlayer(results, n_blocks, run_dir):
    """Plot per-layer k95 and PC1 over training."""
    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Color map for blocks
    cmap = plt.cm.viridis(np.linspace(0, 1, n_blocks))

    # Panel A: k95 per block over training
    ax = axes[0]
    for i in range(n_blocks):
        name = f"block{i}"
        if name not in results or not results[name]:
            continue
        steps = [w["step"] for w in results[name]]
        k95s = [w["k95"] for w in results[name]]
        ax.plot(steps, k95s, '-', color=cmap[i], linewidth=1.5,
                label=f"L{i}", alpha=0.8)

    ax.set_ylabel("k95 (PCs for 95% variance)", fontsize=12)
    ax.set_title("Per-Layer Drift Dimensionality (124M)", fontsize=14)
    ax.legend(fontsize=8, ncol=4, loc='upper right')
    ax.set_ylim(0, 11)
    ax.axhline(10, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'A', transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top')

    # Panel B: PC1% per block over training
    ax = axes[1]
    for i in range(n_blocks):
        name = f"block{i}"
        if name not in results or not results[name]:
            continue
        steps = [w["step"] for w in results[name]]
        pc1s = [w["PC1"] for w in results[name]]
        ax.plot(steps, pc1s, '-', color=cmap[i], linewidth=1.5,
                label=f"L{i}", alpha=0.8)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("PC1 Variance Explained (%)", fontsize=12)
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'B', transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top')

    plt.tight_layout()
    out = out_dir / "perlayer_geometry.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # ── Figure 2: Heatmap of k95 by layer × step ────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Get common steps from block0
    if "block0" in results and results["block0"]:
        common_steps = [w["step"] for w in results["block0"]]
    else:
        return

    # Build k95 heatmap
    k95_matrix = np.zeros((n_blocks, len(common_steps)))
    pc1_matrix = np.zeros((n_blocks, len(common_steps)))

    for i in range(n_blocks):
        name = f"block{i}"
        if name not in results:
            continue
        step_to_idx = {w["step"]: j for j, w in enumerate(results[name])}
        for j, s in enumerate(common_steps):
            if s in step_to_idx:
                wi = step_to_idx[s]
                k95_matrix[i, j] = results[name][wi]["k95"]
                pc1_matrix[i, j] = results[name][wi]["PC1"]

    # k95 heatmap
    ax = axes[0]
    im = ax.imshow(k95_matrix, aspect='auto', cmap='YlOrRd',
                   extent=[common_steps[0], common_steps[-1],
                           n_blocks - 0.5, -0.5],
                   vmin=1, vmax=10)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("k95 by Layer over Training (124M)", fontsize=14)
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f"L{i}" for i in range(n_blocks)])
    plt.colorbar(im, ax=ax, label="k95", shrink=0.8)

    # PC1 heatmap
    ax = axes[1]
    im = ax.imshow(pc1_matrix, aspect='auto', cmap='YlOrRd',
                   extent=[common_steps[0], common_steps[-1],
                           n_blocks - 0.5, -0.5],
                   vmin=0, vmax=100)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_title("PC1% by Layer over Training (124M)", fontsize=14)
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f"L{i}" for i in range(n_blocks)])
    plt.colorbar(im, ax=ax, label="PC1 (%)", shrink=0.8)

    plt.tight_layout()
    out2 = out_dir / "perlayer_heatmap.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: Attn vs MLP comparison ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in [
        (axes[0], "k95", "k95"),
        (axes[1], "PC1", "PC1 (%)"),
    ]:
        for name in results:
            if "_attn" in name or "_mlp" in name:
                if not results[name]:
                    continue
                steps = [w["step"] for w in results[name]]
                vals = [w[metric] for w in results[name]]
                style = '-' if '_attn' in name else '--'
                color = 'tab:blue' if '_attn' in name else 'tab:red'
                ax.plot(steps, vals, style, color=color, linewidth=1.5,
                        label=name, alpha=0.8)

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Attn vs MLP: k95", fontsize=13)
    axes[1].set_title("Attn vs MLP: PC1%", fontsize=13)
    plt.tight_layout()
    out3 = out_dir / "perlayer_attn_vs_mlp.png"
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
