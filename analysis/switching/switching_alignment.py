import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Directional probing and trajectory geometry for attractor oscillation analysis.

Analysis 1 — Trajectory PCA (per-block):
  PCA on consecutive parameter diffs  delta_t = theta_{t+1} - theta_t,
  computed per transformer block.  Shows the dimensionality of the
  optimization trajectory and whether switching dynamics live in a
  low-dimensional subspace.

Analysis 2 — Logit lens:
  At each transformer layer, project hidden states through ln_f + lm_head
  and measure probe exact-match accuracy.  Shows WHERE in the network the
  probe answer becomes decodable.

Analysis 3 — Block-by-block directional probe (prob2.md protocol):
  At peak/trough checkpoints, for each transformer block L:
    (a) Compute switching direction d_L = theta_peak - theta_trough (block only).
    (b) Normalize to unit vector: v_L = d_L / ||d_L||
    (c) Generate n_random orthogonal unit-norm null directions (Gram-Schmidt).
    (d) For each eps in sigma_grid:
        - theta_L' = theta_L + eps * RMS(theta_L) * d_L
        - Measure PRE-relax p_ood  (geometric fragility, §10)
        - Relax 300 steps with fresh optimizer
        - Measure POST-relax p_ood (basin depth)
    (e) Compare trajectory vs random: if traj >> random, that block
        carries switching signal.
  Block-global RMS is a single scalar over all block parameters.

Produces:
  fig_trajectory_pca.png:     Explained variance per PC per block
  fig_logit_lens.png:         Layer-wise probe readout at peaks vs troughs
  fig_directional_probe.png:  Basin depth: trajectory vs random per block
  directional_probing.json:   Full numerical results

Usage:
  # Full analysis:
  python directional_probing.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ \\
      --seed 42

  # Quick (skip expensive directional probe):
  python directional_probing.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ \\
      --seed 42 --skip-probe
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from attractor_analysis import load_checkpoint, load_metrics
from pilot import evaluate_probe
from basin_geometry import relax, make_optimizer


# =========================================================================
# Block parameter utilities
# =========================================================================

def get_block_keys(state_dict, block_idx):
    """Sorted parameter keys for a transformer block (excludes buffers)."""
    prefix = f"blocks.{block_idx}."
    # Exclude registered buffers like attn.bias (causal mask)
    return sorted(k for k in state_dict
                  if k.startswith(prefix) and not k.endswith(".attn.bias"))


def flatten_block(state_dict, block_idx):
    """Flatten all parameters of a transformer block into a 1-D tensor."""
    parts = []
    for key in get_block_keys(state_dict, block_idx):
        parts.append(state_dict[key].cpu().reshape(-1).float())
    return torch.cat(parts)


def block_tensor_shapes(state_dict, block_idx):
    """Return ordered list of (key, shape, numel) for a block's params."""
    info = []
    for key in get_block_keys(state_dict, block_idx):
        t = state_dict[key]
        info.append((key, t.shape, t.numel()))
    return info


def rms(t):
    """Root mean square of a tensor."""
    return t.float().pow(2).mean().sqrt().item()


def apply_perturbation(model, block_idx, direction_unit, eps, block_rms):
    """Apply block-global RMS-scaled perturbation to one block, in-place.

    theta_L' = theta_L + eps * RMS(theta_L) * d_L

    where d_L is a unit-norm direction and RMS is a single scalar computed
    over ALL parameters in the block.  Matches the protocol in prob2.md §4.

    eps=1.0 means "one RMS unit" of perturbation.
    """
    prefix = f"blocks.{block_idx}."
    offset = 0

    with torch.no_grad():
        for name, param in sorted(model.named_parameters(),
                                   key=lambda x: x[0]):
            if not name.startswith(prefix):
                continue
            numel = param.numel()
            d_part = direction_unit[offset:offset + numel].reshape(param.shape)
            param.data.add_((eps * block_rms * d_part).to(
                param.device, param.dtype))
            offset += numel


def generate_random_orthogonal(trajectory_unit, n_random, seed=0):
    """Generate unit-norm random directions orthogonal to the trajectory.

    Per prob2.md §3:
        u ~ N(0, I)
        u <- u - (u . v) v       # orthogonalize vs trajectory
        u <- u / ||u||            # unit norm

    Also mutually orthogonalizes successive directions (Gram-Schmidt).
    Returns list of n_random unit-norm tensors.
    """
    D = trajectory_unit.numel()

    rng = torch.Generator()
    rng.manual_seed(seed)

    directions = []
    basis = [trajectory_unit]  # start with the trajectory direction

    for _ in range(n_random):
        v = torch.randn(D, generator=rng)
        # Gram-Schmidt against all existing basis vectors
        for b in basis:
            v = v - (v @ b) * b
        v_norm = v.norm()
        if v_norm < 1e-8:
            continue
        v = v / v_norm  # unit norm
        basis.append(v)
        directions.append(v)

    return directions


# =========================================================================
# Analysis 1: Trajectory PCA per block
# =========================================================================

def trajectory_pca_per_block(run_dir, n_blocks, step_stride=400, max_pcs=10):
    """PCA on consecutive parameter diffs per transformer block."""
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    # Subsample by stride
    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    if len(steps) < 3:
        steps = all_steps

    print(f"  Loading {len(steps)} checkpoints "
          f"(stride={step_stride}, [{steps[0]}..{steps[-1]}])")

    # Accumulate per-block diffs
    block_diffs = {b: [] for b in range(n_blocks)}
    prev_params = None

    for i, step in enumerate(steps):
        ckpt = load_checkpoint(run_dir, step)
        sd = ckpt["model_state_dict"]

        cur_params = {}
        for b in range(n_blocks):
            cur_params[b] = flatten_block(sd, b)

        del ckpt, sd

        if prev_params is not None:
            for b in range(n_blocks):
                block_diffs[b].append(cur_params[b] - prev_params[b])

        prev_params = cur_params
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(steps)} checkpoints loaded")

    # PCA per block
    pca_results = []
    for b in range(n_blocks):
        diffs = block_diffs[b]
        if len(diffs) < 2:
            pca_results.append(None)
            continue

        X = torch.stack(diffs).numpy()  # (N, D_block)
        X -= X.mean(axis=0, keepdims=True)

        n_comp = min(max_pcs, *X.shape)
        _, S, _ = np.linalg.svd(X, full_matrices=False)

        explained = S ** 2 / max((X.shape[0] - 1), 1)
        total = explained.sum()
        ratio = explained / max(total, 1e-12)
        cumul = np.cumsum(ratio)

        pca_results.append({
            "block": b,
            "n_diffs": len(diffs),
            "dim": X.shape[1],
            "explained_ratio": ratio[:n_comp].tolist(),
            "cumulative": cumul[:n_comp].tolist(),
            "singular_values": S[:n_comp].tolist(),
        })

        print(f"    Block {b}: dim={X.shape[1]:,}, "
              f"top-3 explain {cumul[min(2, len(cumul)-1)]:.1%}")

    return pca_results


def plot_trajectory_pca(pca_results, out_dir):
    """Plot explained variance ratios per block."""
    valid = [r for r in pca_results if r is not None]
    if not valid:
        return

    n = len(valid)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for r, c in zip(valid, colors):
        k = len(r["explained_ratio"])
        ax1.plot(range(1, k + 1), r["explained_ratio"], "-o", color=c,
                 markersize=4, label=f'Block {r["block"]}')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Per-block Trajectory PCA")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = axes[1]
    for r, c in zip(valid, colors):
        k = len(r["cumulative"])
        ax2.plot(range(1, k + 1), r["cumulative"], "-o", color=c,
                 markersize=4, label=f'Block {r["block"]}')
    ax2.axhline(0.90, color="gray", ls=":", alpha=0.5, label="90%")
    ax2.axhline(0.95, color="gray", ls="--", alpha=0.3, label="95%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Trajectory Dimensionality per Block")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = Path(out_dir) / "fig_trajectory_pca.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# =========================================================================
# Analysis 2: Logit lens
# =========================================================================

@torch.no_grad()
def forward_logit_lens(model, input_ids):
    """Forward pass returning per-layer logits via logit lens."""
    B, T = input_ids.shape
    pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
    h = model.drop(model.tok_emb(input_ids) + model.pos_emb(pos))

    layer_logits = [model.lm_head(model.ln_f(h))]
    for block in model.blocks:
        h = block(h)
        layer_logits.append(model.lm_head(model.ln_f(h)))
    return layer_logits


@torch.no_grad()
def logit_lens_accuracy(model, probe_dataset, device, batch_size=128):
    """Logit-lens probe accuracy at each layer."""
    model.eval()
    loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
    n_layers = model.n_layer + 1
    correct = [0] * n_layers
    total = 0

    for input_ids, targets, probe_mask in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        probe_mask = probe_mask.to(device).bool()
        has_probe = probe_mask.any(dim=1)

        layer_logits = forward_logit_lens(model, input_ids)

        for l, logits in enumerate(layer_logits):
            preds = logits.argmax(dim=-1)
            match = (preds == targets) | ~probe_mask
            correct[l] += (match.all(dim=1) & has_probe).sum().item()

        total += has_probe.sum().item()

    return [c / max(total, 1) for c in correct]


def plot_logit_lens(results, out_dir):
    """Plot layer-wise probe accuracy at peaks vs troughs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    peak_c, trough_c = "#d62728", "#2ca02c"
    for r in results:
        n = len(r["accuracies"])
        c = peak_c if r["label"] == "peak" else trough_c
        s = "-o" if r["label"] == "peak" else "--s"
        ax.plot(range(n), r["accuracies"], s, color=c, alpha=0.8,
                markersize=5,
                label=f'{r["label"]} @ {r["step"]} (p={r["p_ood"]:.3f})')
    ax.set_xlabel("Layer (0=embed, 1-8=blocks)")
    ax.set_ylabel("Probe exact-match (logit lens)")
    ax.set_title("Logit Lens: Layer-wise Probe Readout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = Path(out_dir) / "fig_logit_lens.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# =========================================================================
# Analysis 3: Block-by-block directional probe
# =========================================================================

def directional_probe_one_block(model_fn, sd_orig, block_idx,
                                traj_dir, sigma_grid, n_random,
                                train_loader, probe_ood, device, seed=42):
    """Directional probe for a single block (prob2.md protocol).

    For each perturbation magnitude eps:
      1. theta_L' = theta_L + eps * RMS(theta_L) * d_L   (d_L unit-norm)
      2. Measure PRE-relax p_ood  (geometric fragility, §10)
      3. Relax 300 steps (fresh optimizer)
      4. Measure POST-relax p_ood (basin depth)

    Compares trajectory direction vs random orthogonal null directions.

    Args:
        model_fn:     callable() -> fresh GPTModel (on CPU)
        sd_orig:      original state dict (CPU, dict of tensors)
        block_idx:    which transformer block to perturb
        traj_dir:     flat 1-D trajectory direction for this block (raw delta)
        sigma_grid:   list of eps values (0.0 = baseline)
        n_random:     number of random orthogonal directions
        train_loader: training data for relaxation
        probe_ood:    probe eval dataset
        device:       computation device
        seed:         random seed for direction generation

    Returns:
        dict with trajectory/random pre- and post-relax depths
    """
    # Block-global RMS (single scalar, prob2.md §4)
    block_params = flatten_block(sd_orig, block_idx)
    block_rms = rms(block_params)

    # Normalize trajectory direction to unit norm (prob2.md §2)
    traj_norm = traj_dir.norm().item()
    v_traj = traj_dir / max(traj_norm, 1e-12)

    del block_params

    # Random orthogonal unit-norm directions (prob2.md §3)
    rand_dirs = generate_random_orthogonal(v_traj, n_random, seed=seed)

    result = {
        "block": block_idx,
        "block_rms": block_rms,
        "traj_norm": traj_norm,
        "sigma_grid": sigma_grid,
        # Each entry: {"pre_relax": float, "post_relax": float}
        "trajectory": [],
        "random": [[] for _ in rand_dirs],
    }

    def _perturb_and_measure(direction_unit, eps):
        """Apply perturbation, measure pre-relax, relax, measure post-relax."""
        model = model_fn()
        model.load_state_dict({k: v.clone() for k, v in sd_orig.items()})
        model.to(device)

        if eps > 0:
            apply_perturbation(model, block_idx, direction_unit, eps, block_rms)

        # Pre-relax p_ood (prob2.md §10 — geometric fragility)
        pre_relax = evaluate_probe(model, probe_ood, device)

        # Relax 300 steps (standard protocol)
        out = relax(model, train_loader, probe_ood, device)
        post_relax = out["p_ood"]

        del model
        if device == "mps":
            torch.mps.empty_cache()

        return {"pre_relax": pre_relax, "post_relax": post_relax}

    # -- Trajectory direction --
    for eps in sigma_grid:
        m = _perturb_and_measure(v_traj, eps)
        result["trajectory"].append(m)

    # -- Random directions --
    for k, rd in enumerate(rand_dirs):
        for eps in sigma_grid:
            m = _perturb_and_measure(rd, eps)
            result["random"][k].append(m)

    return result


def plot_directional_probe(all_probe_results, out_dir):
    """Plot directional probe: pre-relax + post-relax, traj vs random.

    Layout: 2 columns (pre-relax | post-relax) x N rows (one per checkpoint).
    For each panel, x-axis = block, grouped bars for each eps level,
    blue = trajectory, gray = random mean ± std.
    """
    if not all_probe_results:
        return

    # Group by checkpoint
    ckpts = sorted(set(r["ckpt_step"] for r in all_probe_results))
    n_rows = len(ckpts)

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(16, 5 * n_rows),
                             squeeze=False)

    for row, ckpt_step in enumerate(ckpts):
        rows = sorted([r for r in all_probe_results
                       if r["ckpt_step"] == ckpt_step],
                      key=lambda r: r["probe"]["block"])

        blocks = [r["probe"]["block"] for r in rows]
        eps_grid = rows[0]["probe"]["sigma_grid"]
        # Skip eps=0 in bar plots (it's the baseline)
        plot_eps = [(si, e) for si, e in enumerate(eps_grid) if e > 0]
        if not plot_eps:
            plot_eps = list(enumerate(eps_grid))
        n_eps = len(plot_eps)

        for col, phase in enumerate(["pre_relax", "post_relax"]):
            ax = axes[row, col]
            x = np.arange(len(blocks))
            total_width = 0.8
            bar_w = total_width / (n_eps * 2)

            for gi, (si, eps) in enumerate(plot_eps):
                traj_vals = [r["probe"]["trajectory"][si][phase]
                             for r in rows]
                rand_vals_mean = []
                rand_vals_std = []
                for r in rows:
                    vals = [rd[si][phase] for rd in r["probe"]["random"]]
                    rand_vals_mean.append(np.mean(vals) if vals
                                          else float("nan"))
                    rand_vals_std.append(np.std(vals) if vals
                                         else 0.0)

                offset = (gi - n_eps / 2 + 0.5) * bar_w * 2
                cmap_t = plt.cm.Blues(0.3 + 0.6 * gi / max(n_eps - 1, 1))
                cmap_r = plt.cm.Greys(0.3 + 0.5 * gi / max(n_eps - 1, 1))

                ax.bar(x + offset - bar_w / 2, traj_vals, bar_w,
                       color=cmap_t,
                       label=(f"traj e={eps}" if row == 0 and col == 0
                              else None),
                       alpha=0.85, edgecolor="black", linewidth=0.3)
                ax.bar(x + offset + bar_w / 2, rand_vals_mean, bar_w,
                       yerr=rand_vals_std,
                       color=cmap_r,
                       label=(f"rand e={eps}" if row == 0 and col == 0
                              else None),
                       alpha=0.85, edgecolor="black", linewidth=0.3,
                       capsize=2)

            # eps=0 baseline (if present)
            if eps_grid[0] == 0.0:
                bl = rows[0]["probe"]["trajectory"][0][phase]
                ax.axhline(bl, color="orange", ls=":", lw=1.5,
                           label="e=0 baseline" if row == 0 and col == 0
                           else None)

            lbl = rows[0].get("ckpt_label", "ckpt")
            phase_lbl = "PRE-relax" if phase == "pre_relax" else "POST-relax"
            ax.set_title(f"{lbl} @ {ckpt_step} — {phase_lbl}", fontsize=11)
            ax.set_xlabel("Transformer Block")
            ax.set_ylabel(f"p_ood ({phase_lbl})")
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in blocks])
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 1.0)

    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle("Block-by-block Directional Probe: Trajectory vs Random",
                 fontsize=13)
    plt.tight_layout()
    path = Path(out_dir) / "fig_directional_probe.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Directional probing & trajectory geometry"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    # Feature toggles
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-lens", action="store_true")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip directional probe (hours of compute)")

    # PCA options
    parser.add_argument("--pca-stride", type=int, default=400,
                        help="Checkpoint stride for trajectory PCA")

    # Directional probe options
    parser.add_argument("--sigma-grid", type=str, default="0.0,0.5,1.0,1.5,2.0",
                        help="Epsilon grid (comma-separated); 1.0 = one RMS unit")
    parser.add_argument("--n-random", type=int, default=3,
                        help="Random orthogonal directions per block")
    parser.add_argument("--max-pairs", type=int, default=2,
                        help="Max switch-pairs to probe")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    device = get_device()
    sigma_grid = [float(s) for s in args.sigma_grid.split(",")]

    # ── Load manifest ────────────────────────────────────────────────
    manifest_path = (Path(args.manifest) if args.manifest
                     else run_dir / "oscillation_manifest.json")
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)

    peaks = manifest["peaks"]
    troughs = manifest["troughs"]
    switch_pairs = [(sp["peak"], sp["trough"])
                    for sp in manifest["switch_pairs"]]

    print(f"Run dir : {run_dir}")
    print(f"Peaks   : {peaks}")
    print(f"Troughs : {troughs}")
    print(f"Switch  : {switch_pairs}")
    print(f"Device  : {device}")

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = load_metrics(run_dir)
    step_pood = {m["step"]: m["probe_ood_acc"] for m in metrics}

    # ── Build datasets ───────────────────────────────────────────────
    cw_path = run_dir / "codewords.json"
    cfg = Config(
        seed=42,  # codeword seed always 42
        p_probe=0.10, batch_size=args.batch_size,
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )
    data = build_datasets(
        cfg, codewords_path=str(cw_path) if cw_path.exists() else None,
    )
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    probe_ood = data["probe_eval_ood"]
    train_loader = DataLoader(
        data["train_dataset"], batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )

    n_blocks = cfg.n_layer  # 8
    output = {"pca": None, "logit_lens": [], "directional_probe": []}

    # =================================================================
    # 1. Trajectory PCA
    # =================================================================
    if not args.skip_pca:
        print("\n" + "=" * 60)
        print("1. Trajectory PCA per block")
        print("=" * 60)
        pca = trajectory_pca_per_block(
            run_dir, n_blocks, step_stride=args.pca_stride,
        )
        output["pca"] = pca
        plot_trajectory_pca(pca, out_dir)

    # =================================================================
    # 2. Logit lens
    # =================================================================
    if not args.skip_lens:
        print("\n" + "=" * 60)
        print("2. Logit lens at peaks + troughs")
        print("=" * 60)

        for step in sorted(peaks + troughs):
            label = "peak" if step in peaks else "trough"
            print(f"\n  Step {step} ({label})...")

            ckpt = load_checkpoint(run_dir, step, device=device)
            model = GPTModel(
                vocab_size=vocab_size, seq_len=cfg.seq_len,
                d_model=cfg.d_model, n_layer=cfg.n_layer,
                n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            del ckpt

            accs = logit_lens_accuracy(model, probe_ood, device,
                                       args.batch_size)
            p_ood = step_pood.get(step, accs[-1])
            for l, a in enumerate(accs):
                bar = "#" * int(a * 30)
                print(f"    L{l}: {a:.4f} {bar}")

            output["logit_lens"].append({
                "step": step, "label": label,
                "p_ood": p_ood, "accuracies": accs,
            })

            del model
            if device == "mps":
                torch.mps.empty_cache()

        plot_logit_lens(output["logit_lens"], out_dir)

    # =================================================================
    # 3. Block-by-block directional probe
    # =================================================================
    if not args.skip_probe:
        print("\n" + "=" * 60)
        print("3. Block-by-block directional probe")
        print(f"   sigma_grid: {sigma_grid}")
        print(f"   n_random  : {args.n_random}")
        print(f"   relax     : 300 steps, lr=6e-4, lam=4.0")
        print("=" * 60)

        # Use manifest priority_pairs if available (noise-calibrated selection)
        if "representative" in manifest and manifest["representative"].get("priority_pairs"):
            pp = manifest["representative"]["priority_pairs"]
            probe_pairs = [(p["peak"], p["trough"]) for p in pp[:args.max_pairs]]
            print(f"   Using manifest priority_pairs: {probe_pairs}")
        else:
            probe_pairs = switch_pairs[:args.max_pairs]
            print(f"   Using first {args.max_pairs} switch_pairs: {probe_pairs}")

        def make_model():
            return GPTModel(
                vocab_size=vocab_size, seq_len=cfg.seq_len,
                d_model=cfg.d_model, n_layer=cfg.n_layer,
                n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
            )

        for peak_step, trough_step in probe_pairs:
            print(f"\n  ── Switch pair: peak={peak_step} → "
                  f"trough={trough_step} ──")

            # Load both checkpoints (CPU)
            sd_peak = load_checkpoint(
                run_dir, peak_step, device="cpu"
            )["model_state_dict"]
            sd_trough = load_checkpoint(
                run_dir, trough_step, device="cpu"
            )["model_state_dict"]

            # Per-block switching directions
            print("  Computing per-block switching directions...")
            block_dirs = {}
            for b in range(n_blocks):
                bp = flatten_block(sd_peak, b)
                bt = flatten_block(sd_trough, b)
                block_dirs[b] = bp - bt
                print(f"    Block {b}: dim={bp.numel():,}, "
                      f"param_rms={rms(bp):.6f}, "
                      f"traj_rms={rms(block_dirs[b]):.6f}, "
                      f"ratio={rms(block_dirs[b]) / max(rms(bp), 1e-12):.4f}")

            # Probe from the peak checkpoint
            print(f"\n  Probing from PEAK @ {peak_step}...")
            t0_pair = time.time()

            for b in range(n_blocks):
                t0 = time.time()
                print(f"\n    Block {b}:")

                result = directional_probe_one_block(
                    model_fn=make_model,
                    sd_orig=sd_peak,
                    block_idx=b,
                    traj_dir=block_dirs[b],
                    sigma_grid=sigma_grid,
                    n_random=args.n_random,
                    train_loader=train_loader,
                    probe_ood=probe_ood,
                    device=device,
                    seed=args.seed + b * 17,
                )

                # Summary
                elapsed = time.time() - t0
                for si, eps in enumerate(sigma_grid):
                    t_pre = result["trajectory"][si]["pre_relax"]
                    t_post = result["trajectory"][si]["post_relax"]
                    r_posts = [rd[si]["post_relax"]
                               for rd in result["random"]]
                    r_mean = np.mean(r_posts) if r_posts else float("nan")
                    delta = t_post - r_mean
                    tag = ("SPECIAL" if delta < -0.05
                           else "similar" if abs(delta) < 0.05
                           else "inverse")
                    print(f"      e={eps:.1f}: pre={t_pre:.3f} "
                          f"post={t_post:.3f} rand={r_mean:.3f} "
                          f"({tag}, D={delta:+.3f})")
                print(f"      [{elapsed:.0f}s]")

                output["directional_probe"].append({
                    "ckpt_step": peak_step,
                    "ckpt_label": "peak",
                    "trough_step": trough_step,
                    "probe": result,
                })

            elapsed_pair = time.time() - t0_pair
            print(f"\n  Pair done in {elapsed_pair / 60:.1f} min")

            del sd_peak, sd_trough, block_dirs

        plot_directional_probe(output["directional_probe"], out_dir)

    # ── Save JSON ────────────────────────────────────────────────────
    # Strip non-serializable data
    save = {
        "pca": output["pca"],
        "logit_lens": output["logit_lens"],
        "directional_probe": [
            {
                "ckpt_step": r["ckpt_step"],
                "ckpt_label": r["ckpt_label"],
                "trough_step": r["trough_step"],
                "block": r["probe"]["block"],
                "block_rms": r["probe"]["block_rms"],
                "traj_norm": r["probe"]["traj_norm"],
                "sigma_grid": r["probe"]["sigma_grid"],
                "trajectory": r["probe"]["trajectory"],
                "random": r["probe"]["random"],
            }
            for r in output["directional_probe"]
        ],
    }

    path = out_dir / "directional_probing.json"
    with open(path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved {path}")
    print(f"Done! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
