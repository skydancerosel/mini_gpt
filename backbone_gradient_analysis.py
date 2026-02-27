#!/usr/bin/env python3
"""
Backbone-aware gradient analysis (Step 9 additions: 9A, 9B, 9C).

Connects the backbone–residual decomposition to gradient dynamics:

  9A) Gradient–backbone alignment
      c_LM(t) = |⟨ĝ_LM, v_b⟩|  and  c_P(t) = |⟨ĝ_P, v_b⟩|
      Expected: c_LM high and stable, c_P low (probe mostly transverse).

  9B) Gradient energy split (parallel / perpendicular to backbone)
      ‖g∥‖/‖g‖  and  ‖g⊥‖/‖g‖   for LM and probe gradients.

  9C) Switching direction lives in residual
      |⟨v_switch, v_b⟩|  per block   (should be small).
      Optional: capture of v_switch by top-k residual PCs.

Outputs (in <run-dir>/analysis/):
  fig_gradient_backbone_alignment.png   Plot 1 – cos(g, v_b) vs step
  fig_gradient_energy_split.png         Plot 2 – ‖g∥‖/‖g‖ vs step
  fig_switch_backbone_alignment.png     Plot 3 – |⟨v_switch, v_b⟩| bar chart
  backbone_gradient_analysis.json       Full numerical results

Usage:
  python backbone_gradient_analysis.py \\
      --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s271/ --seed 271
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
from directional_probing import flatten_block, get_block_keys


# ═════════════════════════════════════════════════════════════════════════
# Utility: per-block gradient extraction
# ═════════════════════════════════════════════════════════════════════════

def flatten_block_grad(model, block_idx):
    """Extract .grad for all parameters in a transformer block as 1-D tensor.

    Same parameter ordering as flatten_block() so vectors are comparable.
    Params without grad contribute zeros.
    """
    prefix = f"blocks.{block_idx}."
    parts = []
    for name in sorted(n for n, _ in model.named_parameters()):
        if name.startswith(prefix) and not name.endswith(".attn.bias"):
            param = dict(model.named_parameters())[name]
            if param.grad is not None:
                parts.append(param.grad.cpu().reshape(-1).float())
            else:
                parts.append(torch.zeros(param.numel(), dtype=torch.float32))
    return torch.cat(parts)


# ═════════════════════════════════════════════════════════════════════════
# Compute v_backbone and v_switch per block
# ═════════════════════════════════════════════════════════════════════════

def compute_v_backbone_per_block(run_dir, n_blocks, step_stride=200,
                                 max_residual_pcs=6):
    """Compute v_backbone = PC1 of uncentered cumulative drift per block.

    Also returns top-k right singular vectors for optional residual
    PC analysis (9C).

    Returns:
        v_backbone: dict  block_idx → 1-D tensor (unit-norm, D_block)
        Vt_top:     dict  block_idx → (k, D_block) numpy array
        init_params: dict block_idx → 1-D tensor (init parameters)
    """
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[0] not in steps:
        steps.insert(0, all_steps[0])
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    steps = sorted(set(steps))

    print(f"  Computing v_backbone: loading {len(steps)} checkpoints "
          f"(stride={step_stride})")

    # Load init
    sd0 = load_checkpoint(run_dir, steps[0])["model_state_dict"]
    init_params = {b: flatten_block(sd0, b) for b in range(n_blocks)}
    del sd0

    # Accumulate drifts
    block_drifts = {b: [] for b in range(n_blocks)}
    for i, step in enumerate(steps[1:], 1):
        sd = load_checkpoint(run_dir, step)["model_state_dict"]
        for b in range(n_blocks):
            block_drifts[b].append(flatten_block(sd, b) - init_params[b])
        del sd
        if i % 10 == 0:
            print(f"    {i}/{len(steps)-1} loaded")

    # Uncentered SVD per block
    v_backbone = {}
    Vt_top = {}
    for b in range(n_blocks):
        X = torch.stack(block_drifts[b]).numpy()
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        v_backbone[b] = torch.from_numpy(Vt[0].copy()).float()
        k = min(max_residual_pcs, Vt.shape[0])
        Vt_top[b] = Vt[:k].copy()

        var = S ** 2
        pc1_pct = var[0] / max(var.sum(), 1e-12) * 100
        print(f"    Block {b}: PC1={pc1_pct:.1f}%, dim={X.shape[1]:,}")

    return v_backbone, Vt_top, init_params


def compute_v_switch_per_block(run_dir, peak_step, trough_step, n_blocks):
    """Compute unit-norm switching direction per block."""
    sd_peak = load_checkpoint(run_dir, peak_step)["model_state_dict"]
    sd_trough = load_checkpoint(run_dir, trough_step)["model_state_dict"]
    v_switch = {}
    for b in range(n_blocks):
        d = flatten_block(sd_peak, b) - flatten_block(sd_trough, b)
        v_switch[b] = d / d.norm()
    del sd_peak, sd_trough
    return v_switch


# ═════════════════════════════════════════════════════════════════════════
# Checkpoint selection
# ═════════════════════════════════════════════════════════════════════════

def auto_select_checkpoints(manifest, include_init=200):
    """Select 6–8 representative checkpoints from manifest.

    Priority: init, early_peak, early_trough, transition (~λ switch),
    mid_peak, mid_trough, late, final (10000).
    """
    rep = manifest.get("representative", {})
    peaks = set(manifest["peaks"])
    troughs = set(manifest["troughs"])

    candidates = []
    candidates.append(("init", include_init))

    # Early peak + nearest trough
    ep = rep.get("early_peak")
    if ep:
        candidates.append(("early_peak", ep))
        for sp in manifest["switch_pairs"]:
            if sp["peak"] == ep:
                candidates.append(("early_trough", sp["trough"]))
                break

    # Transition (near λ switch at ~4000)
    tt = rep.get("transition_trough")
    if tt:
        candidates.append(("transition", tt))

    # Mid peak + mid trough
    mp = rep.get("mid_peak")
    mt = rep.get("mid_trough")
    if mp:
        candidates.append(("mid_peak", mp))
    if mt:
        candidates.append(("mid_trough", mt))

    # Late
    lp = rep.get("late_peak")
    if lp:
        candidates.append(("late", lp))

    # Final
    candidates.append(("final", 10000))

    # Deduplicate
    seen = set()
    result = []
    for label, step in candidates:
        if step not in seen:
            seen.add(step)
            tag = ("peak" if step in peaks
                   else "trough" if step in troughs
                   else "other")
            result.append({"step": step, "label": label, "type": tag})

    return result


# ═════════════════════════════════════════════════════════════════════════
# 9A + 9B: Gradient–backbone alignment and energy split
# ═════════════════════════════════════════════════════════════════════════

def gradient_backbone_analysis(model, dataloader, device, v_backbone,
                               n_blocks, n_batches=8, lambda_probe=2.0):
    """Compute per-block gradient alignment with backbone direction.

    For each mini-batch, computes separate LM and probe gradients via
    retain_graph, then measures alignment and energy split per block.

    Returns dict with per-block results averaged over batches.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(dataloader)

    # Accumulators per block
    cos_lm = {b: [] for b in range(n_blocks)}
    cos_probe = {b: [] for b in range(n_blocks)}
    par_frac_lm = {b: [] for b in range(n_blocks)}
    perp_frac_lm = {b: [] for b in range(n_blocks)}
    par_frac_probe = {b: [] for b in range(n_blocks)}
    perp_frac_probe = {b: [] for b in range(n_blocks)}
    lm_grad_norms = {b: [] for b in range(n_blocks)}
    probe_grad_norms = {b: [] for b in range(n_blocks)}

    valid_batches = 0

    for i in range(n_batches):
        try:
            input_ids, targets, probe_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, targets, probe_mask = next(data_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)
        probe_mask = probe_mask.to(device)

        # Forward pass
        model.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)

        # Compute per-position losses
        loss_flat = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss_all = loss_flat.view(targets.shape)

        pmask = probe_mask.bool()
        lm_mask = ~pmask & (targets != -100)

        if not lm_mask.any():
            continue

        lm_loss = loss_all[lm_mask].mean()

        has_probe = pmask.any()
        if has_probe:
            p_loss = loss_all[pmask].mean()
        else:
            continue  # skip batches without probe data

        # ── LM backward (retain graph for probe backward) ──────────
        model.zero_grad(set_to_none=True)
        lm_loss.backward(retain_graph=True)

        for b in range(n_blocks):
            g = flatten_block_grad(model, b)
            g_norm = g.norm().item()
            lm_grad_norms[b].append(g_norm)

            if g_norm < 1e-12:
                cos_lm[b].append(0.0)
                par_frac_lm[b].append(0.0)
                perp_frac_lm[b].append(1.0)
                continue

            g_hat = g / g_norm
            vb = v_backbone[b]

            c = abs(float(torch.dot(g_hat, vb)))
            cos_lm[b].append(c)

            par = abs(float(torch.dot(g, vb)))
            par_frac_lm[b].append(par / g_norm)
            perp_frac_lm[b].append(
                float((g - float(torch.dot(g, vb)) * vb).norm()) / g_norm)

        # ── Probe backward ─────────────────────────────────────────
        model.zero_grad(set_to_none=True)
        p_loss.backward()

        for b in range(n_blocks):
            g = flatten_block_grad(model, b)
            g_norm = g.norm().item()
            probe_grad_norms[b].append(g_norm)

            if g_norm < 1e-12:
                cos_probe[b].append(0.0)
                par_frac_probe[b].append(0.0)
                perp_frac_probe[b].append(1.0)
                continue

            g_hat = g / g_norm
            vb = v_backbone[b]

            c = abs(float(torch.dot(g_hat, vb)))
            cos_probe[b].append(c)

            par = abs(float(torch.dot(g, vb)))
            par_frac_probe[b].append(par / g_norm)
            perp_frac_probe[b].append(
                float((g - float(torch.dot(g, vb)) * vb).norm()) / g_norm)

        valid_batches += 1

    if valid_batches == 0:
        print("    WARNING: no valid batches!")
        return {}

    # Average over batches
    result = {}
    for b in range(n_blocks):
        result[b] = {
            "cos_lm_backbone": float(np.mean(cos_lm[b])),
            "cos_lm_backbone_std": float(np.std(cos_lm[b])),
            "cos_probe_backbone": float(np.mean(cos_probe[b])),
            "cos_probe_backbone_std": float(np.std(cos_probe[b])),
            "parallel_frac_lm": float(np.mean(par_frac_lm[b])),
            "perp_frac_lm": float(np.mean(perp_frac_lm[b])),
            "parallel_frac_probe": float(np.mean(par_frac_probe[b])),
            "perp_frac_probe": float(np.mean(perp_frac_probe[b])),
            "lm_grad_norm": float(np.mean(lm_grad_norms[b])),
            "probe_grad_norm": float(np.mean(probe_grad_norms[b])),
        }

    return result


# ═════════════════════════════════════════════════════════════════════════
# 9C: Switching direction lives in residual
# ═════════════════════════════════════════════════════════════════════════

def switch_backbone_alignment(v_switch, v_backbone, Vt_top, n_blocks):
    """Compute |⟨v_switch, v_backbone⟩| per block.

    Also measures how much of v_switch is captured by residual PCs
    (PC2–PC6, i.e. Vt_top[1:]).

    Returns dict per block.
    """
    results = {}
    for b in range(n_blocks):
        vs = v_switch[b].numpy()
        vb = v_backbone[b].numpy()

        cos_sb = abs(float(np.dot(vs, vb)))

        # Capture by residual PCs (PC2 onward)
        Vt = Vt_top[b]  # (k, D)
        residual_capture = {}
        for k in range(2, min(Vt.shape[0] + 1, 7)):
            # Project v_switch onto PCs 2..k (indices 1..k-1)
            V_res = Vt[1:k]  # (k-1, D)
            proj = V_res @ vs  # (k-1,)
            capture = float(np.sum(proj ** 2))  # fraction (vs is unit-norm)
            residual_capture[f"pc2_to_pc{k}"] = capture

        results[b] = {
            "cos_switch_backbone": cos_sb,
            "residual_pc_capture": residual_capture,
        }

    return results


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════

def plot_gradient_backbone_alignment(all_results, out_dir):
    """Plot 1: cos(g_LM, v_b) and cos(g_probe, v_b) vs training step.

    Layout: one panel per block (2×4 grid).
    """
    steps = [r["step"] for r in all_results]
    n_blocks = max(int(k) for r in all_results
                   for k in r["per_block"].keys()) + 1

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for b in range(n_blocks):
        ax = axes[b]
        c_lm = [r["per_block"][str(b)]["cos_lm_backbone"]
                 for r in all_results]
        c_lm_std = [r["per_block"][str(b)]["cos_lm_backbone_std"]
                     for r in all_results]
        c_pr = [r["per_block"][str(b)]["cos_probe_backbone"]
                 for r in all_results]
        c_pr_std = [r["per_block"][str(b)]["cos_probe_backbone_std"]
                     for r in all_results]

        ax.errorbar(steps, c_lm, yerr=c_lm_std, fmt="-o", color="steelblue",
                     markersize=5, linewidth=1.5, capsize=3,
                     label=r"$|\cos(\hat g_{LM}, v_b)|$")
        ax.errorbar(steps, c_pr, yerr=c_pr_std, fmt="-s", color="indianred",
                     markersize=5, linewidth=1.5, capsize=3,
                     label=r"$|\cos(\hat g_P, v_b)|$")

        ax.set_title(f"Block {b}", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if b == 0:
            ax.legend(fontsize=8)

    for ax in axes[4:]:
        ax.set_xlabel("Training step", fontsize=10)
    for ax in axes[::4]:
        ax.set_ylabel("Alignment", fontsize=10)

    fig.suptitle("Gradient–Backbone Alignment (9A)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = Path(out_dir) / "fig_gradient_backbone_alignment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gradient_energy_split(all_results, out_dir):
    """Plot 2: ‖g∥‖/‖g‖ for LM and probe vs step.

    Layout: 2 rows (LM top, Probe bottom), colored lines per block.
    """
    steps = [r["step"] for r in all_results]
    n_blocks = max(int(k) for r in all_results
                   for k in r["per_block"].keys()) + 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: LM gradient parallel fraction
    ax = axes[0]
    for b in range(n_blocks):
        vals = [r["per_block"][str(b)]["parallel_frac_lm"]
                for r in all_results]
        ax.plot(steps, vals, "-o", color=colors[b], markersize=4,
                linewidth=1.2, label=f"Block {b}")
    ax.set_ylabel(r"$\|\mathbf{g}^{\parallel}\| / \|\mathbf{g}\|$",
                   fontsize=12)
    ax.set_title(r"LM gradient — backbone fraction $\|\mathbf{g}^{\parallel}_{LM}\| / \|\mathbf{g}_{LM}\|$",
                  fontsize=12)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom: Probe gradient parallel fraction
    ax = axes[1]
    for b in range(n_blocks):
        vals = [r["per_block"][str(b)]["parallel_frac_probe"]
                for r in all_results]
        ax.plot(steps, vals, "-o", color=colors[b], markersize=4,
                linewidth=1.2, label=f"Block {b}")
    ax.set_ylabel(r"$\|\mathbf{g}^{\parallel}\| / \|\mathbf{g}\|$",
                   fontsize=12)
    ax.set_title(r"Probe gradient — backbone fraction $\|\mathbf{g}^{\parallel}_{P}\| / \|\mathbf{g}_{P}\|$",
                  fontsize=12)
    ax.set_xlabel("Training step", fontsize=11)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient Energy Split: Backbone vs Transverse (9B)",
                  fontsize=14, y=1.01)
    fig.tight_layout()
    path = Path(out_dir) / "fig_gradient_energy_split.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_switch_backbone_bar(switch_results, out_dir):
    """Plot 3: bar chart of |⟨v_switch, v_b⟩| per block + residual PC capture."""
    n_blocks = len(switch_results)
    blocks = sorted(switch_results.keys())
    cos_vals = [switch_results[b]["cos_switch_backbone"] for b in blocks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: backbone alignment bar chart
    ax = axes[0]
    x = np.arange(n_blocks)
    bars = ax.bar(x, cos_vals, color="steelblue", alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    # Random baseline: 1/sqrt(D) ≈ 0.0006 for D=3.1M
    ax.axhline(0.001, color="gray", ls=":", alpha=0.5,
               label="random baseline (~0.001)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel(r"$|\langle v_{switch}, v_{backbone} \rangle|$")
    ax.set_title("Switch–Backbone Alignment")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, cos_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Right: residual PC capture
    ax = axes[1]
    for b in blocks:
        cap = switch_results[b]["residual_pc_capture"]
        ks = sorted(cap.keys())
        if ks:
            labels = [k.replace("pc2_to_", "PC2–") for k in ks]
            vals = [cap[k] for k in ks]
            ax.plot(labels, vals, "-o", markersize=5, linewidth=1.2,
                    label=f"Block {b}")
    ax.set_xlabel("Residual PCs used")
    ax.set_ylabel(r"Captured fraction of $v_{switch}$")
    ax.set_title(r"$v_{switch}$ captured by residual PCs")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Switching Direction Lives in Residual (9C)",
                  fontsize=14, y=1.01)
    fig.tight_layout()
    path = Path(out_dir) / "fig_switch_backbone_alignment.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backbone-aware gradient analysis (Step 9: 9A, 9B, 9C)"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (codeword seed is always 42)")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Explicit comma-separated steps (overrides auto)")
    parser.add_argument("--n-batches", type=int, default=8,
                        help="Gradient samples per checkpoint (default: 8)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-probe", type=float, default=2.0)
    parser.add_argument("--pca-stride", type=int, default=200,
                        help="Checkpoint stride for v_backbone computation")
    parser.add_argument("--switch-pair", type=str, default=None,
                        help="Explicit peak:trough (e.g. '2200:3000')")
    parser.add_argument("--skip-9a", action="store_true")
    parser.add_argument("--skip-9b", action="store_true")
    parser.add_argument("--skip-9c", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    device = get_device()
    n_blocks = 8

    print(f"Backbone Gradient Analysis")
    print(f"  run_dir:    {run_dir}")
    print(f"  device:     {device}")
    print(f"  n_batches:  {args.n_batches}")
    print(f"  batch_size: {args.batch_size}")

    # ── Load manifest ────────────────────────────────────────────────
    manifest_path = (Path(args.manifest) if args.manifest
                     else run_dir / "oscillation_manifest.json")
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # ── Select checkpoints ───────────────────────────────────────────
    if args.checkpoints:
        ckpt_steps = [int(s.strip()) for s in args.checkpoints.split(",")]
        peaks = set(manifest["peaks"])
        troughs = set(manifest["troughs"])
        checkpoint_info = []
        for s in sorted(ckpt_steps):
            tag = ("peak" if s in peaks
                   else "trough" if s in troughs
                   else "other")
            checkpoint_info.append({"step": s, "label": tag, "type": tag})
    else:
        checkpoint_info = auto_select_checkpoints(manifest)

    print(f"\n  Checkpoints ({len(checkpoint_info)}):")
    for ci in checkpoint_info:
        print(f"    step {ci['step']:>6d}  ({ci['label']})")

    # ── Determine switch pair ────────────────────────────────────────
    if args.switch_pair:
        pk, tr = args.switch_pair.split(":")
        switch_pair = (int(pk), int(tr))
    else:
        rep = manifest.get("representative", {})
        pp = rep.get("priority_pairs", manifest.get("switch_pairs", []))
        if pp:
            switch_pair = (pp[0]["peak"], pp[0]["trough"])
        else:
            print("ERROR: no switch pairs found", file=sys.stderr)
            sys.exit(1)
    print(f"  Switch pair: peak={switch_pair[0]} → trough={switch_pair[1]}")

    # ── Compute v_backbone per block ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing v_backbone per block (uncentered SVD)")
    print(f"{'='*60}")
    v_backbone, Vt_top, _ = compute_v_backbone_per_block(
        run_dir, n_blocks, step_stride=args.pca_stride)

    # ── Compute v_switch per block ───────────────────────────────────
    print(f"\nComputing v_switch: peak={switch_pair[0]} → "
          f"trough={switch_pair[1]}")
    v_switch = compute_v_switch_per_block(
        run_dir, switch_pair[0], switch_pair[1], n_blocks)

    # ── Build datasets ───────────────────────────────────────────────
    print(f"\nBuilding datasets...")
    cfg = Config(
        seed=42,  # codeword seed always 42
        p_probe=0.10, batch_size=args.batch_size,
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )
    cw_path = run_dir / "codewords.json"
    data = build_datasets(
        cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    train_loader = DataLoader(
        data["train_dataset"], batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0)

    # ── 9C: switch-backbone alignment (no model needed) ──────────────
    switch_result = None
    if not args.skip_9c:
        print(f"\n{'='*60}")
        print("9C: Switch–backbone alignment")
        print(f"{'='*60}")
        switch_result = switch_backbone_alignment(
            v_switch, v_backbone, Vt_top, n_blocks)
        for b in sorted(switch_result.keys()):
            r = switch_result[b]
            print(f"  Block {b}: |⟨v_switch, v_b⟩| = "
                  f"{r['cos_switch_backbone']:.6f}")
        plot_switch_backbone_bar(switch_result, out_dir)

    # ── 9A + 9B: gradient analysis at each checkpoint ────────────────
    all_results = []
    if not (args.skip_9a and args.skip_9b):
        print(f"\n{'='*60}")
        print("9A + 9B: Gradient–backbone alignment & energy split")
        print(f"{'='*60}")

        total_t0 = time.time()

        for ci in checkpoint_info:
            step = ci["step"]
            label = ci["label"]
            print(f"\n  Step {step} ({label})")

            t0 = time.time()
            ckpt = load_checkpoint(run_dir, step, device=device)
            model = GPTModel(
                vocab_size=vocab_size, seq_len=cfg.seq_len,
                d_model=cfg.d_model, n_layer=cfg.n_layer,
                n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            del ckpt

            per_block = gradient_backbone_analysis(
                model, train_loader, device, v_backbone,
                n_blocks, n_batches=args.n_batches,
                lambda_probe=args.lambda_probe)

            elapsed = time.time() - t0

            entry = {
                "step": step,
                "label": label,
                "type": ci["type"],
                "per_block": {str(b): v for b, v in per_block.items()},
                "elapsed_sec": round(elapsed, 1),
            }
            all_results.append(entry)

            # Print summary
            for b in range(n_blocks):
                r = per_block.get(b, {})
                print(f"    Block {b}: cos_LM={r.get('cos_lm_backbone', 0):.4f}  "
                      f"cos_P={r.get('cos_probe_backbone', 0):.4f}  "
                      f"par_LM={r.get('parallel_frac_lm', 0):.4f}  "
                      f"par_P={r.get('parallel_frac_probe', 0):.4f}")
            print(f"    [{elapsed:.1f}s]")

            del model
            if device == "mps":
                torch.mps.empty_cache()

        total_elapsed = time.time() - total_t0
        print(f"\n  Total gradient analysis: {total_elapsed:.1f}s")

        # Plots
        if not args.skip_9a:
            plot_gradient_backbone_alignment(all_results, out_dir)
        if not args.skip_9b:
            plot_gradient_energy_split(all_results, out_dir)

    # ── Save JSON ────────────────────────────────────────────────────
    output = {
        "config": {
            "run_dir": str(run_dir),
            "seed": args.seed,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "lambda_probe": args.lambda_probe,
            "pca_stride": args.pca_stride,
            "switch_pair": list(switch_pair),
            "checkpoints": [ci["step"] for ci in checkpoint_info],
        },
        "gradient_backbone_alignment": all_results,
        "switch_backbone_alignment": (
            {str(b): v for b, v in switch_result.items()}
            if switch_result else None),
    }

    json_path = out_dir / "backbone_gradient_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")
    print(f"Done! Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
