import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Three closing analyses for the backbone story:

1) Update-direction alignment
   u_t = θ(t) − θ(t−Δ)  (actual parameter change including AdamW
   preconditioner + weight decay, accumulated over Δ=200 steps).
   Report |cos(u_t, v_b)| and signed ⟨û_t, v_b⟩  per block.

2) Signed gradient projection bias
   b_t = ⟨g_t, v_b⟩  (signed, per mini-batch).
   Distribution at each checkpoint + cumulative sum over training.

3) Residualized switching direction
   v_switch⊥ = v_switch − ⟨v_switch, v_b⟩ v_b,  renormalised.
   Redo residual PC capture on v_switch⊥.

Outputs (in <run-dir>/analysis/):
  fig_update_backbone_alignment.png
  fig_signed_gradient_bias.png
  fig_switch_residualized.png
  backbone_update_analysis.json

Usage:
  python backbone_update_analysis.py \
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
# Shared: v_backbone, v_switch, checkpoint selection
# ═════════════════════════════════════════════════════════════════════════

def compute_v_backbone_and_pcs(run_dir, n_blocks, step_stride=200, max_pcs=6):
    """Uncentered SVD on cumulative drift → v_backbone + top-k PCs per block."""
    ckpt_dir = Path(run_dir)
    ckpt_files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)

    steps = [s for s in all_steps if s % step_stride == 0]
    if all_steps[0] not in steps:
        steps.insert(0, all_steps[0])
    if all_steps[-1] not in steps:
        steps.append(all_steps[-1])
    steps = sorted(set(steps))

    print(f"  Loading {len(steps)} checkpoints for v_backbone (stride={step_stride})")
    sd0 = load_checkpoint(run_dir, steps[0])["model_state_dict"]
    init_params = {b: flatten_block(sd0, b) for b in range(n_blocks)}
    del sd0

    block_drifts = {b: [] for b in range(n_blocks)}
    for i, step in enumerate(steps[1:], 1):
        sd = load_checkpoint(run_dir, step)["model_state_dict"]
        for b in range(n_blocks):
            block_drifts[b].append(flatten_block(sd, b) - init_params[b])
        del sd
        if i % 10 == 0:
            print(f"    {i}/{len(steps)-1}")

    v_backbone = {}
    Vt_top = {}
    for b in range(n_blocks):
        X = torch.stack(block_drifts[b]).numpy()
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        v_backbone[b] = torch.from_numpy(Vt[0].copy()).float()
        k = min(max_pcs, Vt.shape[0])
        Vt_top[b] = Vt[:k].copy()
        var = S ** 2
        print(f"    Block {b}: PC1={var[0]/var.sum()*100:.1f}%")

    return v_backbone, Vt_top


def compute_v_switch(run_dir, peak, trough, n_blocks):
    sd_p = load_checkpoint(run_dir, peak)["model_state_dict"]
    sd_t = load_checkpoint(run_dir, trough)["model_state_dict"]
    v = {}
    for b in range(n_blocks):
        d = flatten_block(sd_p, b) - flatten_block(sd_t, b)
        v[b] = d / d.norm()
    return v


def auto_select_checkpoints(manifest, include_init=200):
    rep = manifest.get("representative", {})
    peaks = set(manifest["peaks"])
    troughs = set(manifest["troughs"])
    cands = [("init", include_init)]
    ep = rep.get("early_peak")
    if ep:
        cands.append(("early_peak", ep))
        for sp in manifest["switch_pairs"]:
            if sp["peak"] == ep:
                cands.append(("early_trough", sp["trough"]))
                break
    tt = rep.get("transition_trough")
    if tt:
        cands.append(("transition", tt))
    mp, mt = rep.get("mid_peak"), rep.get("mid_trough")
    if mp:
        cands.append(("mid_peak", mp))
    if mt:
        cands.append(("mid_trough", mt))
    lp = rep.get("late_peak")
    if lp:
        cands.append(("late", lp))
    final_step = rep.get("final_step", manifest.get("late_lm", [10000])[-1])
    cands.append(("final", final_step))
    seen, out = set(), []
    for lbl, s in cands:
        if s not in seen:
            seen.add(s)
            tag = "peak" if s in peaks else ("trough" if s in troughs else "other")
            out.append({"step": s, "label": lbl, "type": tag})
    return out


# ═════════════════════════════════════════════════════════════════════════
# Per-block gradient extraction
# ═════════════════════════════════════════════════════════════════════════

def flatten_block_grad(model, block_idx):
    prefix = f"blocks.{block_idx}."
    parts = []
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        if name.startswith(prefix) and not name.endswith(".attn.bias"):
            if param.grad is not None:
                parts.append(param.grad.cpu().reshape(-1).float())
            else:
                parts.append(torch.zeros(param.numel(), dtype=torch.float32))
    return torch.cat(parts)


# ═════════════════════════════════════════════════════════════════════════
# Analysis 1: Update-direction alignment
# ═════════════════════════════════════════════════════════════════════════

def update_backbone_alignment(run_dir, n_blocks, v_backbone, step_stride=200):
    """Compute |cos(u_t, v_b)| and signed cos for actual parameter updates.

    u_t = θ(t) − θ(t−Δ)  is the cumulative AdamW update over Δ=stride steps.
    This includes preconditioner, momentum, weight decay, gradient clipping —
    everything the optimizer does.
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

    print(f"  Loading {len(steps)} checkpoints for update alignment")

    # Load all block params
    all_params = {}
    for i, step in enumerate(steps):
        sd = load_checkpoint(run_dir, step)["model_state_dict"]
        all_params[step] = {b: flatten_block(sd, b) for b in range(n_blocks)}
        del sd
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(steps)}")
    print(f"    {len(steps)}/{len(steps)} done")

    # Compute update directions between consecutive checkpoints
    results = {b: {"steps": [], "abs_cos": [], "signed_cos": [],
                    "update_norm": []}
               for b in range(n_blocks)}

    for i in range(1, len(steps)):
        s_prev, s_curr = steps[i - 1], steps[i]
        mid_step = (s_prev + s_curr) // 2  # label at midpoint

        for b in range(n_blocks):
            u = all_params[s_curr][b] - all_params[s_prev][b]
            u_norm = u.norm().item()
            if u_norm < 1e-12:
                results[b]["steps"].append(mid_step)
                results[b]["abs_cos"].append(0.0)
                results[b]["signed_cos"].append(0.0)
                results[b]["update_norm"].append(0.0)
                continue

            vb = v_backbone[b]
            cos_val = float(torch.dot(u, vb)) / u_norm
            results[b]["steps"].append(mid_step)
            results[b]["abs_cos"].append(abs(cos_val))
            results[b]["signed_cos"].append(cos_val)
            results[b]["update_norm"].append(u_norm)

    return results


def plot_update_alignment(update_results, p_ood_series, out_dir):
    """Plot |cos(u_t, v_b)| and signed cos(u_t, v_b) vs step, per block."""
    n_blocks = len(update_results)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: |cos(u_t, v_b)| — absolute alignment
    ax = axes[0]
    for b in range(n_blocks):
        r = update_results[b]
        ax.plot(r["steps"], r["abs_cos"], "-o", color=colors[b],
                markersize=3, linewidth=1.2, label=f"Block {b}")
    ax.set_ylabel(r"$|\cos(u_t, v_b)|$", fontsize=12)
    ax.set_title(r"Update–Backbone Alignment: $|\cos(u_t, v_b)|$", fontsize=13)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Overlay p_ood on twin axis
    if p_ood_series:
        ax2 = ax.twinx()
        ax2.plot(p_ood_series["steps"], p_ood_series["values"],
                 "k--", linewidth=1.5, alpha=0.4, label="$p_{ood}$")
        ax2.set_ylabel("$p_{ood}$", fontsize=11)
        ax2.legend(loc="lower right", fontsize=9)

    # Bottom: signed cos(u_t, v_b)
    ax = axes[1]
    for b in range(n_blocks):
        r = update_results[b]
        ax.plot(r["steps"], r["signed_cos"], "-o", color=colors[b],
                markersize=3, linewidth=1.2, label=f"Block {b}")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_ylabel(r"$\cos(u_t, v_b)$ (signed)", fontsize=12)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_title(r"Signed Update–Backbone Alignment", fontsize=13)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    if p_ood_series:
        ax2 = ax.twinx()
        ax2.plot(p_ood_series["steps"], p_ood_series["values"],
                 "k--", linewidth=1.5, alpha=0.4, label="$p_{ood}$")
        ax2.set_ylabel("$p_{ood}$", fontsize=11)

    fig.tight_layout()
    path = Path(out_dir) / "fig_update_backbone_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
# Analysis 2: Signed gradient projection bias
# ═════════════════════════════════════════════════════════════════════════

def signed_gradient_projection(model, dataloader, device, v_backbone,
                               n_blocks, n_batches=16, lambda_probe=2.0):
    """Compute b = ⟨g, v_b⟩ (signed) for each batch, per block.

    Returns dict per block with list of signed projections.
    Uses combined loss gradient (L_LM + λ L_probe) to match training.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(dataloader)

    projections = {b: [] for b in range(n_blocks)}
    grad_norms = {b: [] for b in range(n_blocks)}

    for i in range(n_batches):
        try:
            input_ids, targets, probe_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, targets, probe_mask = next(data_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)
        probe_mask = probe_mask.to(device)

        model.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)

        loss_flat = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss_all = loss_flat.view(targets.shape)
        pmask = probe_mask.bool()
        lm_mask = ~pmask & (targets != -100)

        lm_loss = (loss_all[lm_mask].mean()
                   if lm_mask.any()
                   else torch.tensor(0.0, device=device))
        p_loss = (loss_all[pmask].mean()
                  if pmask.any()
                  else torch.tensor(0.0, device=device))
        loss = lm_loss + lambda_probe * p_loss

        loss.backward()

        for b in range(n_blocks):
            g = flatten_block_grad(model, b)
            g_norm = g.norm().item()
            grad_norms[b].append(g_norm)
            # Signed projection: no absolute value
            proj = float(torch.dot(g, v_backbone[b]))
            projections[b].append(proj)

    return projections, grad_norms


def plot_signed_gradient_bias(all_projections, out_dir):
    """Plot signed gradient projection distribution + cumulative sum.

    Left: boxplot of ⟨g, v_b⟩ per checkpoint per block (averaged over blocks).
    Right: cumulative sum of mean ⟨g, v_b⟩ over training.
    """
    steps = [r["step"] for r in all_projections]
    n_blocks = max(int(k) for r in all_projections
                   for k in r["projections"].keys()) + 1
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Top: mean signed projection vs step, per block
    ax = axes[0]
    for b in range(n_blocks):
        means = [np.mean(r["projections"][str(b)]) for r in all_projections]
        stds = [np.std(r["projections"][str(b)]) for r in all_projections]
        ax.errorbar(steps, means, yerr=stds, fmt="-o", color=colors[b],
                     markersize=4, linewidth=1.2, capsize=3,
                     label=f"Block {b}")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_ylabel(r"$\langle g_t, v_b \rangle$ (signed)", fontsize=12)
    ax.set_title(r"Signed Gradient Projection onto Backbone: $b_t = \langle g_t, v_b \rangle$",
                  fontsize=13)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Training step", fontsize=11)

    # Bottom: cumulative sum of mean projection across checkpoints
    ax = axes[1]
    for b in range(n_blocks):
        means = [np.mean(r["projections"][str(b)]) for r in all_projections]
        cumsum = np.cumsum(means)
        ax.plot(steps, cumsum, "-o", color=colors[b],
                markersize=4, linewidth=1.5, label=f"Block {b}")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_ylabel(r"$\sum b_t$ (cumulative)", fontsize=12)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_title(r"Cumulative Signed Gradient Bias along Backbone", fontsize=13)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "fig_signed_gradient_bias.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
# Analysis 3: Residualized switching direction
# ═════════════════════════════════════════════════════════════════════════

def residualized_switch_capture(v_switch, v_backbone, Vt_top, n_blocks):
    """Orthogonalize v_switch against v_backbone, then measure capture by
    residual PCs (PC2–PC6).

    v_switch⊥ = v_switch − ⟨v_switch, v_b⟩ v_b,  then renormalize.
    """
    results = {}
    for b in range(n_blocks):
        vs = v_switch[b]
        vb = v_backbone[b]

        # Original alignment
        cos_orig = float(torch.dot(vs, vb))

        # Residualize
        vs_perp = vs - cos_orig * vb
        vs_perp_norm = vs_perp.norm().item()
        if vs_perp_norm < 1e-12:
            results[b] = {
                "cos_switch_backbone": abs(cos_orig),
                "vs_perp_norm_fraction": 0.0,
                "capture_original": {},
                "capture_residualized": {},
            }
            continue

        vs_perp = vs_perp / vs_perp_norm
        vs_perp_np = vs_perp.numpy()
        vs_np = vs.numpy()

        Vt = Vt_top[b]  # (k, D)

        # Capture of ORIGINAL v_switch by residual PCs
        capture_orig = {}
        for k in range(2, min(Vt.shape[0] + 1, 7)):
            V_res = Vt[1:k]  # PC2..PCk
            proj = V_res @ vs_np
            capture_orig[f"pc2_to_pc{k}"] = float(np.sum(proj ** 2))

        # Capture of RESIDUALIZED v_switch⊥ by residual PCs
        capture_resid = {}
        for k in range(2, min(Vt.shape[0] + 1, 7)):
            V_res = Vt[1:k]
            proj = V_res @ vs_perp_np
            capture_resid[f"pc2_to_pc{k}"] = float(np.sum(proj ** 2))

        results[b] = {
            "cos_switch_backbone": abs(cos_orig),
            "vs_perp_norm_fraction": vs_perp_norm,
            "capture_original": capture_orig,
            "capture_residualized": capture_resid,
        }

    return results


def plot_residualized_switch(switch_results, out_dir):
    """Plot residual PC capture: original vs residualized v_switch."""
    n_blocks = len(switch_results)
    blocks = sorted(switch_results.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: bar chart of |cos(v_switch, v_b)| (same as before, for reference)
    ax = axes[0]
    cos_vals = [switch_results[b]["cos_switch_backbone"] for b in blocks]
    perp_frac = [switch_results[b]["vs_perp_norm_fraction"] for b in blocks]
    x = np.arange(n_blocks)
    ax.bar(x - 0.15, cos_vals, 0.3, color="steelblue", alpha=0.8,
           label=r"$|\langle v_{sw}, v_b\rangle|$")
    ax.bar(x + 0.15, perp_frac, 0.3, color="indianred", alpha=0.8,
           label=r"$\|v_{sw}^\perp\|$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("Block")
    ax.set_ylabel("Magnitude")
    ax.set_title("Switch Direction Decomposition")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: original v_switch capture by residual PCs
    ax = axes[1]
    for b in blocks:
        cap = switch_results[b]["capture_original"]
        if cap:
            ks = sorted(cap.keys())
            labels = [k.replace("pc2_to_", "PC2–") for k in ks]
            vals = [cap[k] for k in ks]
            ax.plot(labels, vals, "-o", color=colors[b], markersize=5,
                    linewidth=1.2, label=f"Block {b}")
    ax.set_xlabel("Residual PCs used")
    ax.set_ylabel("Captured fraction")
    ax.set_title(r"$v_{switch}$ captured by residual PCs")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Right: residualized v_switch⊥ capture by residual PCs
    ax = axes[2]
    for b in blocks:
        cap = switch_results[b]["capture_residualized"]
        if cap:
            ks = sorted(cap.keys())
            labels = [k.replace("pc2_to_", "PC2–") for k in ks]
            vals = [cap[k] for k in ks]
            ax.plot(labels, vals, "-o", color=colors[b], markersize=5,
                    linewidth=1.2, label=f"Block {b}")
    ax.set_xlabel("Residual PCs used")
    ax.set_ylabel("Captured fraction")
    ax.set_title(r"$v_{switch}^\perp$ captured by residual PCs")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Residualized Switching Direction Analysis", fontsize=14, y=1.01)
    fig.tight_layout()
    path = Path(out_dir) / "fig_switch_residualized.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backbone update + signed gradient + residualized switch")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--n-batches-grad", type=int, default=16,
                        help="Gradient samples per checkpoint (default: 16)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-stride", type=int, default=200)
    parser.add_argument("--update-stride", type=int, default=200,
                        help="Stride for update alignment (default: 200)")
    parser.add_argument("--switch-pair", type=str, default=None)
    parser.add_argument("--lambda-probe", type=float, default=2.0,
                        help="Base lambda for probe loss (default: 2.0)")
    parser.add_argument("--lambda-probe2", type=float, default=None,
                        help="Second-phase lambda (after --lambda-step)")
    parser.add_argument("--lambda-step", type=int, default=4000,
                        help="Step at which to switch lambda (default: 4000)")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--skip-switch", action="store_true")
    args = parser.parse_args()

    lambda_base = args.lambda_probe
    lambda_phase2 = args.lambda_probe2 if args.lambda_probe2 is not None else lambda_base
    lambda_step = args.lambda_step

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    device = get_device()
    n_blocks = 8

    print(f"Backbone Update Analysis")
    print(f"  run_dir: {run_dir}")
    print(f"  device:  {device}")

    # ── Load manifest ────────────────────────────────────────────────
    manifest_path = (Path(args.manifest) if args.manifest
                     else run_dir / "oscillation_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # ── Switch pair ──────────────────────────────────────────────────
    if args.switch_pair:
        pk, tr = args.switch_pair.split(":")
        switch_pair = (int(pk), int(tr))
    else:
        rep = manifest.get("representative", {})
        pp = rep.get("priority_pairs", manifest.get("switch_pairs", []))
        switch_pair = (pp[0]["peak"], pp[0]["trough"])
    print(f"  Switch pair: {switch_pair[0]} → {switch_pair[1]}")

    # ── Compute directions ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing v_backbone + residual PCs per block")
    print(f"{'='*60}")
    v_backbone, Vt_top = compute_v_backbone_and_pcs(
        run_dir, n_blocks, step_stride=args.pca_stride)

    print(f"\nComputing v_switch")
    v_switch = compute_v_switch(
        run_dir, switch_pair[0], switch_pair[1], n_blocks)

    # ── Load metrics for p_ood overlay ───────────────────────────────
    metrics = load_metrics(run_dir)
    p_ood_series = {
        "steps": [m["step"] for m in metrics],
        "values": [m["probe_ood_acc"] for m in metrics],
    }

    output = {"config": {
        "run_dir": str(run_dir), "seed": args.seed,
        "switch_pair": list(switch_pair),
        "update_stride": args.update_stride,
        "n_batches_grad": args.n_batches_grad,
    }}

    # ═════════════════════════════════════════════════════════════════
    # 1. Update-direction alignment
    # ═════════════════════════════════════════════════════════════════
    if not args.skip_update:
        print(f"\n{'='*60}")
        print(f"1. Update–backbone alignment (stride={args.update_stride})")
        print(f"{'='*60}")

        update_results = update_backbone_alignment(
            run_dir, n_blocks, v_backbone,
            step_stride=args.update_stride)

        for b in range(n_blocks):
            r = update_results[b]
            mean_abs = np.mean(r["abs_cos"])
            mean_sgn = np.mean(r["signed_cos"])
            print(f"  Block {b}: mean |cos|={mean_abs:.4f}, "
                  f"mean signed={mean_sgn:.4f}")

        output["update_alignment"] = {
            str(b): {k: v if not isinstance(v, list) else v
                     for k, v in update_results[b].items()}
            for b in range(n_blocks)
        }
        plot_update_alignment(update_results, p_ood_series, out_dir)

    # ═════════════════════════════════════════════════════════════════
    # 2. Signed gradient projection bias
    # ═════════════════════════════════════════════════════════════════
    if not args.skip_gradient:
        print(f"\n{'='*60}")
        print(f"2. Signed gradient projection bias")
        print(f"{'='*60}")

        checkpoint_info = auto_select_checkpoints(manifest)
        print(f"  Checkpoints: {[c['step'] for c in checkpoint_info]}")

        # Build datasets
        cfg = Config(seed=42, p_probe=0.10, batch_size=args.batch_size,
                     n_layer=8, d_model=512, n_head=16, d_ff=2048)
        cw_path = run_dir / "codewords.json"
        data = build_datasets(
            cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
        vocab_size = len(data["tokenizer"])
        train_loader = DataLoader(
            data["train_dataset"], batch_size=args.batch_size,
            shuffle=True, drop_last=True, num_workers=0)

        print(f"  Lambda schedule: base={lambda_base}, phase2={lambda_phase2}, switch@{lambda_step}")

        all_proj_results = []
        for ci in checkpoint_info:
            step = ci["step"]
            cur_lambda = lambda_phase2 if step >= lambda_step else lambda_base
            print(f"\n  Step {step} ({ci['label']}, λ={cur_lambda})")
            ckpt = load_checkpoint(run_dir, step, device=device)
            model = GPTModel(
                vocab_size=vocab_size, seq_len=cfg.seq_len,
                d_model=cfg.d_model, n_layer=cfg.n_layer,
                n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            del ckpt

            projections, grad_norms = signed_gradient_projection(
                model, train_loader, device, v_backbone,
                n_blocks, n_batches=args.n_batches_grad,
                lambda_probe=cur_lambda)

            entry = {
                "step": step,
                "label": ci["label"],
                "projections": {str(b): projections[b]
                                for b in range(n_blocks)},
                "grad_norms": {str(b): grad_norms[b]
                               for b in range(n_blocks)},
            }
            all_proj_results.append(entry)

            for b in range(n_blocks):
                m = np.mean(projections[b])
                s = np.std(projections[b])
                print(f"    Block {b}: ⟨g, v_b⟩ = {m:.6f} ± {s:.6f}")

            del model
            if device == "mps":
                torch.mps.empty_cache()

        output["signed_gradient_projection"] = all_proj_results
        plot_signed_gradient_bias(all_proj_results, out_dir)

    # ═════════════════════════════════════════════════════════════════
    # 3. Residualized switching direction
    # ═════════════════════════════════════════════════════════════════
    if not args.skip_switch:
        print(f"\n{'='*60}")
        print("3. Residualized switching direction capture")
        print(f"{'='*60}")

        switch_results = residualized_switch_capture(
            v_switch, v_backbone, Vt_top, n_blocks)

        for b in range(n_blocks):
            r = switch_results[b]
            cap_o = r["capture_original"].get("pc2_to_pc6", 0)
            cap_r = r["capture_residualized"].get("pc2_to_pc6", 0)
            print(f"  Block {b}: |⟨v_sw, v_b⟩|={r['cos_switch_backbone']:.4f}  "
                  f"‖v⊥‖={r['vs_perp_norm_fraction']:.4f}  "
                  f"orig_cap(2-6)={cap_o:.4f}  resid_cap(2-6)={cap_r:.4f}")

        output["residualized_switch"] = {
            str(b): switch_results[b] for b in range(n_blocks)}
        plot_residualized_switch(switch_results, out_dir)

    # ── Save JSON ────────────────────────────────────────────────────
    json_path = out_dir / "backbone_update_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
