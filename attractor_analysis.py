#!/usr/bin/env python3
"""
Attractor switching analysis for the TinyStories probe experiment.

Produces:
  Figure B1: Time series of p_ood, nll_ood, lm_ood
  Figure B2: Scatter lm_ood vs nll_ood (tradeoff curve)
  Figure B3: Switching-direction alignment (cosine of Δθ_t with Δ_switch)
  Table B1:  Basin test — steps-to-threshold from peak vs trough

Usage:
  python attractor_analysis.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/
"""

import argparse
import json
import math
import random
import re
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
from pilot import evaluate_probe, evaluate_probe_nll


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_metrics(run_dir):
    """Load pilot_metrics.json from run directory."""
    p = Path(run_dir) / "pilot_metrics.json"
    with open(p) as f:
        return json.load(f)


def load_checkpoint(run_dir, step, device="cpu"):
    """Load model state_dict from a checkpoint."""
    p = Path(run_dir) / f"ckpt_{step:06d}.pt"
    return torch.load(p, map_location=device, weights_only=True)


def identify_peaks_troughs(metrics, key="probe_ood_acc", min_prominence=0.05):
    """Find local maxima (peaks) and minima (troughs) of a metric.

    Returns list of (step_idx, step, value, label) tuples.
    """
    vals = [m[key] for m in metrics]
    steps = [m["step"] for m in metrics]
    markers = []

    for i in range(1, len(vals) - 1):
        if vals[i] > vals[i-1] and vals[i] > vals[i+1]:
            # Check prominence
            left_min = min(vals[max(0,i-3):i])
            right_min = min(vals[i+1:min(len(vals),i+4)])
            prom = vals[i] - max(left_min, right_min)
            if prom >= min_prominence:
                markers.append((i, steps[i], vals[i], "peak"))
        elif vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            left_max = max(vals[max(0,i-3):i])
            right_max = max(vals[i+1:min(len(vals),i+4)])
            prom = min(left_max, right_max) - vals[i]
            if prom >= min_prominence:
                markers.append((i, steps[i], vals[i], "trough"))

    return markers


def get_peak_trough_pairs(markers, n_pairs=3):
    """Extract peak-trough pairs from markers list."""
    pairs = []
    peaks = [(i, s, v) for i, s, v, l in markers if l == "peak"]
    troughs = [(i, s, v) for i, s, v, l in markers if l == "trough"]

    # Match each peak to nearest subsequent trough
    for pi, ps, pv in peaks:
        for ti, ts, tv in troughs:
            if ts > ps:
                pairs.append({"peak_step": ps, "peak_val": pv,
                              "trough_step": ts, "trough_val": tv})
                break

    return pairs[:n_pairs]


# Trunk-only parameter filter for geometry analysis
TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)

def flatten_state_dict_filtered(state_dict):
    """Flatten trunk-only params (attn + MLP weights) into a single 1-D tensor.

    Excludes: tok_emb, lm_head (tied), pos_emb, attn.bias (causal mask), all LN.
    Includes: blocks.*.attn.qkv.weight, attn.out_proj.weight, mlp.w_up/w_down.weight.
    Returns float32 tensor of ~25.2M elements.
    """
    parts = []
    for key in sorted(state_dict.keys()):
        if TRUNK_PATTERN.match(key):
            parts.append(state_dict[key].cpu().reshape(-1).float())
    if not parts:
        raise ValueError("No parameters matched TRUNK_PATTERN — check state_dict keys")
    return torch.cat(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Figure B1: Time series
# ═══════════════════════════════════════════════════════════════════════════

def plot_timeseries(metrics, markers, out_dir):
    """Plot p_ood, nll_ood, lm_ood over steps."""
    steps = [m["step"] for m in metrics]
    p_ood = [m["probe_ood_acc"] for m in metrics]
    nll_ood = [m["nll_ood"] for m in metrics]
    lm_ood = [m["lm_ood"] for m in metrics]
    p_in = [m["probe_in_acc"] for m in metrics]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: p_ood and p_in
    ax1 = axes[0]
    ax1.plot(steps, p_ood, "b-", linewidth=1.5, label="p_ood (exact)")
    ax1.plot(steps, p_in, "b--", linewidth=1, alpha=0.5, label="p_in (exact)")
    for _, s, v, label in markers:
        color = "red" if label == "peak" else "green"
        marker = "^" if label == "peak" else "v"
        ax1.scatter(s, v, c=color, marker=marker, s=80, zorder=5)
    ax1.set_ylabel("Exact-match accuracy")
    ax1.legend(loc="upper left")
    ax1.set_title("Figure B1: Attractor Oscillation")
    ax1.grid(True, alpha=0.3)

    # Panel 2: nll_ood (probe answer NLL)
    ax2 = axes[1]
    ax2.plot(steps, nll_ood, "r-", linewidth=1.5, label="nll_ood (probe)")
    ax2.set_ylabel("Probe answer NLL")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # Panel 3: lm_ood (LM context NLL on probe examples)
    ax3 = axes[2]
    ax3.plot(steps, lm_ood, "g-", linewidth=1.5, label="lm_ood (context LM)")
    ax3.set_ylabel("LM NLL on probe examples")
    ax3.set_xlabel("Training step")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(out_dir) / "fig_B1_timeseries.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure B2: Scatter tradeoff
# ═══════════════════════════════════════════════════════════════════════════

def plot_scatter(metrics, out_dir):
    """Scatter plot: lm_ood vs nll_ood, colored by step."""
    # Skip step 1 (random init outlier)
    ms = [m for m in metrics if m["step"] > 1]
    lm_vals = [m["lm_ood"] for m in ms]
    nll_vals = [m["nll_ood"] for m in ms]
    steps = [m["step"] for m in ms]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(lm_vals, nll_vals, c=steps, cmap="viridis",
                    s=40, alpha=0.8, edgecolors="none")
    plt.colorbar(sc, label="Training step")

    # Correlation
    r = np.corrcoef(lm_vals, nll_vals)[0, 1]
    ax.set_xlabel("LM NLL on probe examples (lm_ood)")
    ax.set_ylabel("Probe answer NLL (nll_ood)")
    ax.set_title(f"Figure B2: Objective Competition (r = {r:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(out_dir) / "fig_B2_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path} (correlation r={r:.3f})")
    return r


# ═══════════════════════════════════════════════════════════════════════════
# Figure B3: Switching-direction alignment
# ═══════════════════════════════════════════════════════════════════════════

def compute_switching_alignment(run_dir, metrics, peak_step=2800, trough_step=2000):
    """Cosine alignment of each consecutive weight delta with the switching direction.

    Switching direction: Δ_switch = θ_peak − θ_trough  (trunk-only params).
    For each consecutive pair (t, t+Δ):
        a_t = cos(θ_{t+Δ} − θ_t, Δ_switch)

    Positive a_t → weight update moves toward probe-dominant regime.
    Negative a_t → weight update moves toward LM-dominant regime.
    """
    print(f"  Switching direction: θ_{peak_step} − θ_{trough_step} (trunk-only)")

    # Load reference checkpoints
    ckpt_peak = load_checkpoint(run_dir, peak_step)
    ckpt_trough = load_checkpoint(run_dir, trough_step)
    theta_peak = flatten_state_dict_filtered(ckpt_peak["model_state_dict"])
    theta_trough = flatten_state_dict_filtered(ckpt_trough["model_state_dict"])
    delta_switch = theta_peak - theta_trough
    delta_switch_norm = delta_switch / (delta_switch.norm() + 1e-12)
    d_trunk = delta_switch.numel()
    print(f"    |Δ_switch| = {delta_switch.norm():.4f}  ({d_trunk:,} trunk params)")
    del ckpt_peak, ckpt_trough, theta_peak, theta_trough

    # Sequential pass: compute alignment for each consecutive delta
    steps = [m["step"] for m in metrics]
    mid_steps = []
    alignments = []

    prev_theta = None
    for i, step in enumerate(steps):
        ckpt = load_checkpoint(run_dir, step)
        theta = flatten_state_dict_filtered(ckpt["model_state_dict"])
        del ckpt

        if prev_theta is not None:
            delta_t = theta - prev_theta
            norm_t = delta_t.norm()
            if norm_t > 1e-12:
                cos_a = float((delta_t @ delta_switch_norm) / norm_t)
            else:
                cos_a = 0.0
            alignments.append(cos_a)
            mid_steps.append((steps[i - 1] + step) / 2)

        prev_theta = theta
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(steps)} checkpoints")

    print(f"    {len(alignments)} alignment values computed")
    return {
        "mid_steps": mid_steps,
        "alignments": alignments,
        "peak_step": peak_step,
        "trough_step": trough_step,
        "delta_switch_norm": delta_switch_norm,
    }


def compute_pairwise_cosines(sa_list):
    """Cosine similarities between all pairs of switching directions."""
    n = len(sa_list)
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = float(
                sa_list[i]["delta_switch_norm"] @ sa_list[j]["delta_switch_norm"]
            )
    return cos_matrix


def plot_multi_switching_alignment(sa_list, cos_matrix, metrics, markers, out_dir):
    """Plot switching-direction alignment for multiple directions alongside p_ood."""
    steps = [m["step"] for m in metrics]
    p_ood = [m["probe_ood_acc"] for m in metrics]
    n_dirs = len(sa_list)

    fig, axes = plt.subplots(n_dirs + 1, 1, figsize=(12, 3.5 * (n_dirs + 1)), sharex=True)
    if n_dirs + 1 == 2:
        axes = [axes[0], axes[1]]  # ensure list

    # Panel 1: p_ood
    ax1 = axes[0]
    ax1.plot(steps, p_ood, "b-", linewidth=1.5, label="p_ood")
    for _, s, v, label in markers:
        color = "red" if label == "peak" else "green"
        msym = "^" if label == "peak" else "v"
        ax1.scatter(s, v, c=color, marker=msym, s=80, zorder=5)
    ax1.set_ylabel("p_ood (exact-match)")
    ax1.set_title("Figure B3: Multi-direction switching alignment")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Per-direction panels
    delta_p = [p_ood[i + 1] - p_ood[i] for i in range(len(p_ood) - 1)]
    dir_colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]  # distinct colors per direction

    for k, sa in enumerate(sa_list):
        ax = axes[k + 1]
        mid = sa["mid_steps"]
        ali = sa["alignments"]
        dc = dir_colors[k % len(dir_colors)]

        ax.plot(mid, ali, color=dc, linewidth=1, alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(mid, 0, ali,
                        where=[a > 0 for a in ali],
                        color="red", alpha=0.15)
        ax.fill_between(mid, 0, ali,
                        where=[a <= 0 for a in ali],
                        color="blue", alpha=0.15)

        # Mark the reference window
        pk, tr = sa["peak_step"], sa["trough_step"]
        lo, hi = min(pk, tr), max(pk, tr)
        ax.axvspan(lo, hi, color=dc, alpha=0.08)

        ax.set_ylabel(f"a$^{{({k+1})}}$_t")
        label = f"Δ$^{{({k+1})}}$ = θ_{pk} − θ_{tr}"
        ax.legend([label], loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Correlation
        if len(delta_p) == len(ali):
            r = np.corrcoef(ali, delta_p)[0, 1]
            print(f"  Direction {k+1} ({pk}→{tr}): corr(a_t, Δp_ood) = {r:.3f}")

    axes[-1].set_xlabel("Training step")

    plt.tight_layout()
    path = Path(out_dir) / "fig_B3_switching_alignment.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # Print pairwise cosine table
    if cos_matrix is not None and n_dirs > 1:
        print(f"\n  Pairwise cosines between switching directions:")
        header = "        " + "  ".join(f"Δ({j+1})" for j in range(n_dirs))
        print(f"  {header}")
        for i in range(n_dirs):
            row = f"  Δ({i+1})  " + "  ".join(f"{cos_matrix[i,j]:+.4f}" for j in range(n_dirs))
            print(row)


# ═══════════════════════════════════════════════════════════════════════════
# Table B1: Basin test
# ═══════════════════════════════════════════════════════════════════════════

def basin_test(run_dir, pairs, n_finetune_steps=1000, finetune_lambda=8.0,
               eval_every=100, target_p_ood=0.6, codewords_path=None, seed=42):
    """Fine-tune from peak and trough checkpoints, measure recovery speed."""
    device = get_device()

    # Build datasets once
    cfg = Config(
        seed=seed, p_probe=0.10, lr=1e-3, batch_size=64,
        total_steps=n_finetune_steps, eval_every=eval_every,
        warmup_steps=0,  # no warmup for fine-tune
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )
    # Use saved codewords if available (auto-detect from run_dir)
    if codewords_path is None:
        default_cw = Path(run_dir) / "codewords.json"
        if default_cw.exists():
            codewords_path = str(default_cw)
    data = build_datasets(cfg, codewords_path=codewords_path)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    probe_ood = data["probe_eval_ood"]

    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )

    ce_none = nn.CrossEntropyLoss(reduction='none')

    results = []

    for pair in pairs:
        for ckpt_type in ["peak", "trough"]:
            step = pair[f"{ckpt_type}_step"]
            start_p_ood = pair[f"{ckpt_type}_val"]

            print(f"\n  Basin test: {ckpt_type} @ step {step} (start p_ood={start_p_ood:.3f})")

            # Load checkpoint
            ckpt = load_checkpoint(run_dir, step, device=device)
            model = GPTModel(
                vocab_size=vocab_size, seq_len=cfg.seq_len,
                d_model=cfg.d_model, n_layer=cfg.n_layer,
                n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            # Optimizer (fresh)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.5)
            data_iter = iter(train_loader)

            # Fine-tune with high lambda
            trace = []
            steps_to_target = None

            for ft_step in range(1, n_finetune_steps + 1):
                model.train()
                opt.zero_grad(set_to_none=True)

                try:
                    input_ids, targets, probe_mask = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    input_ids, targets, probe_mask = next(data_iter)

                input_ids = input_ids.to(device)
                targets = targets.to(device)
                probe_mask = probe_mask.to(device)

                logits, _ = model(input_ids)
                loss_flat = ce_none(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_all = loss_flat.view(targets.shape)
                pmask = probe_mask.bool()
                lm_mask = ~pmask & (targets != -100)
                lm_loss = loss_all[lm_mask].mean() if lm_mask.any() else torch.tensor(0.0, device=device)
                p_loss = loss_all[pmask].mean() if pmask.any() else torch.tensor(0.0, device=device)
                loss = lm_loss + finetune_lambda * p_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if ft_step % eval_every == 0:
                    p_acc = evaluate_probe(model, probe_ood, device)
                    trace.append((ft_step, p_acc))
                    if p_acc >= target_p_ood and steps_to_target is None:
                        steps_to_target = ft_step

            max_achieved = max(v for _, v in trace) if trace else 0
            results.append({
                "type": ckpt_type,
                "source_step": step,
                "start_p_ood": start_p_ood,
                "steps_to_target": steps_to_target,
                "max_achieved": max_achieved,
                "trace": trace,
            })
            status = f"reached {target_p_ood:.0%} at step {steps_to_target}" if steps_to_target else f"max={max_achieved:.3f}"
            print(f"    Result: {status}")

    return results


def print_basin_table(results, target):
    """Print Table B1."""
    print(f"\n  Table B1: Basin test (target p_ood >= {target:.0%})")
    print(f"  {'Type':<8s} {'Source':>7s} {'Start':>7s} {'Steps':>7s} {'Max':>7s}")
    print(f"  {'-'*40}")
    for r in results:
        steps_str = str(r["steps_to_target"]) if r["steps_to_target"] else "never"
        print(f"  {r['type']:<8s} {r['source_step']:>7d} {r['start_p_ood']:>7.3f} "
              f"{steps_str:>7s} {r['max_achieved']:>7.3f}")


def plot_basin_traces(results, out_dir):
    """Plot basin test recovery curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        steps = [s for s, _ in r["trace"]]
        accs = [a for _, a in r["trace"]]
        color = "red" if r["type"] == "peak" else "blue"
        style = "-" if r["type"] == "peak" else "--"
        label = f"{r['type']} @ {r['source_step']} (start={r['start_p_ood']:.2f})"
        ax.plot(steps, accs, style, color=color, linewidth=1.5, alpha=0.7, label=label)

    ax.axhline(y=0.6, color="gray", linestyle=":", alpha=0.5, label="target=0.6")
    ax.set_xlabel("Fine-tune steps (λ=8.0)")
    ax.set_ylabel("p_ood (exact-match)")
    ax.set_title("Table B1: Basin test — recovery from peak vs trough")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(out_dir) / "fig_B1_basin_test.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure B4: Reheating test
# ═══════════════════════════════════════════════════════════════════════════

def plot_reheating(orig_metrics, reheat_metrics, out_dir):
    """Plot original run + reheating continuation."""
    # Original run data
    o_steps = [m["step"] for m in orig_metrics]
    o_pood = [m["probe_ood_acc"] for m in orig_metrics]
    o_nll = [m["nll_ood"] for m in orig_metrics]

    # Reheating data — shift x-axis to continue from original end
    orig_end_step = o_steps[-1]
    r_steps = [orig_end_step + m["step"] for m in reheat_metrics]
    r_pood = [m["probe_ood_acc"] for m in reheat_metrics]
    r_nll = [m["nll_ood"] for m in reheat_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: p_ood
    ax1 = axes[0]
    ax1.plot(o_steps, o_pood, "b-", linewidth=1, alpha=0.6, label="original run")
    ax1.plot(r_steps, r_pood, "r-", linewidth=2, label="reheating (LR reset)")
    ax1.axvline(x=orig_end_step, color="orange", linestyle="--", linewidth=2,
                alpha=0.8, label="LR reset point")
    ax1.set_ylabel("p_ood (exact-match)")
    ax1.legend(loc="upper left")
    ax1.set_title("Figure B4: Reheating Test — LR freeze-out vs basin geometry")
    ax1.grid(True, alpha=0.3)

    # Panel 2: nll_ood
    ax2 = axes[1]
    ax2.plot(o_steps, o_nll, "b-", linewidth=1, alpha=0.6, label="original run")
    ax2.plot(r_steps, r_nll, "r-", linewidth=2, label="reheating (LR reset)")
    ax2.axvline(x=orig_end_step, color="orange", linestyle="--", linewidth=2, alpha=0.8)
    ax2.set_ylabel("Probe answer NLL (nll_ood)")
    ax2.set_xlabel("Training step")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    path = Path(out_dir) / "fig_B4_reheating.png"
    plt.savefig(path, dpi=150)
    plt.close()

    # Verdict
    reheat_max_pood = max(r_pood) if r_pood else 0
    reheat_start_pood = r_pood[0] if r_pood else 0
    print(f"  Saved {path}")
    print(f"  Reheating: start p_ood={reheat_start_pood:.3f}, max p_ood={reheat_max_pood:.3f}")
    if reheat_max_pood >= 0.50:
        print(f"  VERDICT: Oscillation returned (max={reheat_max_pood:.1%}) → LR freeze-out was the main cause")
    else:
        print(f"  VERDICT: Oscillation did NOT return (max={reheat_max_pood:.1%}) → basin geometry changed")
    return reheat_max_pood


# ═══════════════════════════════════════════════════════════════════════════
# Figure B5: LR threshold map
# ═══════════════════════════════════════════════════════════════════════════

def plot_lr_threshold(reheat_dirs, out_dir):
    """Plot LR-threshold map: max p_ood and time-to-escape for each LR.

    Args:
        reheat_dirs: list of (lr_value, run_dir_path) tuples
        out_dir: output directory for the figure
    """
    escape_threshold = 0.4

    results = []
    for lr, rdir in reheat_dirs:
        ms = load_metrics(rdir)
        p_vals = [m["probe_ood_acc"] for m in ms]
        s_vals = [m["step"] for m in ms]

        max_p = max(p_vals)
        max_step = s_vals[p_vals.index(max_p)]

        escape_step = None
        for m in ms:
            if m["probe_ood_acc"] > escape_threshold:
                escape_step = m["step"]
                break

        results.append({
            "lr": lr, "max_pood": max_p, "max_step": max_step,
            "escape_step": escape_step, "steps": s_vals, "p_ood": p_vals,
        })
        esc = f"step {escape_step}" if escape_step else "never"
        print(f"  LR={lr:.0e}: max p_ood={max_p:.3f} @ step {max_step}, "
              f"escape (>{escape_threshold}) at {esc}")

    results.sort(key=lambda r: r["lr"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    # Panel 1: p_ood traces
    ax1 = axes[0]
    for r, c in zip(results, colors):
        ax1.plot(r["steps"], r["p_ood"], color=c, linewidth=1.5,
                 label=f"LR={r['lr']:.0e}")
    ax1.axhline(y=escape_threshold, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Reheating step")
    ax1.set_ylabel("p_ood")
    ax1.set_title("Recovery traces")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: max p_ood bar chart
    ax2 = axes[1]
    lrs = [r["lr"] for r in results]
    maxps = [r["max_pood"] for r in results]
    ax2.bar(range(len(lrs)), maxps, color=colors,
            tick_label=[f"{lr:.0e}" for lr in lrs])
    ax2.set_xlabel("Peak LR")
    ax2.set_ylabel("Max p_ood achieved")
    ax2.set_title("Max p_ood")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: time-to-escape bar chart
    ax3 = axes[2]
    escs = [r["escape_step"] if r["escape_step"] else 0 for r in results]
    bar_colors = [c if r["escape_step"] else (0.8, 0.8, 0.8, 1.0)
                  for r, c in zip(results, colors)]
    ax3.bar(range(len(lrs)), escs, color=bar_colors,
            tick_label=[f"{lr:.0e}" for lr in lrs])
    ax3.set_xlabel("Peak LR")
    ax3.set_ylabel(f"Steps to p_ood > {escape_threshold}")
    ax3.set_title("Time to escape")
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure B5: LR Threshold Map for Basin Escape", fontsize=14)
    plt.tight_layout()
    path = Path(out_dir) / "fig_B5_lr_threshold.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Attractor switching analysis")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to pilot run directory with checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset construction (must match training seed)")
    parser.add_argument("--skip-basin", action="store_true",
                        help="Skip basin test (saves time)")
    parser.add_argument("--skip-subspace", action="store_true",
                        help="Skip switching-direction alignment analysis")
    parser.add_argument("--reheat-dir", type=str, default=None,
                        help="Path to reheating test run directory")
    parser.add_argument("--reheat-dirs", type=str, default=None,
                        help="Comma-separated LR:path pairs for LR threshold map, "
                             "e.g. '3e-4:runs/dir1,6e-4:runs/dir2,1e-3:runs/dir3'")
    parser.add_argument("--switch-pairs", type=str, default=None,
                        help="Comma-separated peak:trough pairs for switching directions, "
                             "e.g. '2800:2000,5000:5400,6400:6800'. "
                             "If not given, auto-detects from first detected peak-trough pair.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading metrics from {run_dir}...")
    metrics = load_metrics(run_dir)
    print(f"  {len(metrics)} eval points")

    # Identify peaks and troughs
    markers = identify_peaks_troughs(metrics, min_prominence=0.03)
    peaks = [(i, s, v) for i, s, v, l in markers if l == "peak"]
    troughs = [(i, s, v) for i, s, v, l in markers if l == "trough"]
    print(f"  Found {len(peaks)} peaks, {len(troughs)} troughs")
    for _, s, v, l in markers:
        print(f"    {l:>6s} @ step {s}: p_ood={v:.3f}")

    pairs = get_peak_trough_pairs(markers)
    print(f"  {len(pairs)} peak-trough pairs")

    # Figure B1
    print("\nFigure B1: Time series...")
    plot_timeseries(metrics, markers, out_dir)

    # Figure B2
    print("\nFigure B2: Scatter tradeoff...")
    r = plot_scatter(metrics, out_dir)

    # Figure B3: Switching-direction alignment (multi-direction)
    if not args.skip_subspace:
        # Resolve switch pairs: explicit CLI > auto-detect from markers
        if args.switch_pairs is not None:
            switch_pairs = []
            for entry in args.switch_pairs.split(","):
                peak_s, trough_s = entry.strip().split(":")
                switch_pairs.append((int(peak_s), int(trough_s)))
        else:
            # Auto-detect from markers
            if len(peaks) == 0 or len(troughs) == 0:
                print("\nERROR: --switch-pairs not given and no peaks/troughs detected.",
                      file=sys.stderr)
                print("Cannot compute switching alignment without peak-trough pairs.",
                      file=sys.stderr)
                sys.exit(1)
            switch_pairs = []
            for pair in pairs:
                switch_pairs.append((pair["peak_step"], pair["trough_step"]))
            if not switch_pairs:
                print("\nERROR: No peak-trough pairs could be formed from detected markers.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  Auto-detected switch-pairs: {switch_pairs}")

        print(f"\nFigure B3: Switching-direction alignment ({len(switch_pairs)} directions)...")

        # Compute alignment for each direction
        sa_list = []
        for i, (pk, tr) in enumerate(switch_pairs):
            print(f"\n  Direction {i+1}: θ_{pk} − θ_{tr}")
            sa = compute_switching_alignment(run_dir, metrics, peak_step=pk, trough_step=tr)
            sa_list.append(sa)

        # Pairwise cosines between switching directions
        cos_mat = None
        if len(sa_list) > 1:
            cos_mat = compute_pairwise_cosines(sa_list)

        # Plot
        plot_multi_switching_alignment(sa_list, cos_mat, metrics, markers, out_dir)

        # Save JSON (strip torch tensors)
        save_data = {
            "directions": [
                {
                    "peak_step": sa["peak_step"],
                    "trough_step": sa["trough_step"],
                    "mid_steps": sa["mid_steps"],
                    "alignments": sa["alignments"],
                }
                for sa in sa_list
            ],
        }
        if cos_mat is not None:
            save_data["pairwise_cosines"] = cos_mat.tolist()
        with open(out_dir / "switching_alignment.json", "w") as f:
            json.dump(save_data, f, indent=2)

    # Table B1: Basin test
    if not args.skip_basin and len(pairs) > 0:
        print("\nTable B1: Basin test...")
        target = 0.6
        basin_results = basin_test(run_dir, pairs, n_finetune_steps=1000,
                                   finetune_lambda=8.0, eval_every=50,
                                   target_p_ood=target, seed=args.seed)
        print_basin_table(basin_results, target)
        plot_basin_traces(basin_results, out_dir)
        # Save
        with open(out_dir / "basin_test.json", "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "trace"} for r in basin_results],
                      f, indent=2)
    elif len(pairs) == 0:
        print("\n  Skipping basin test: no peak-trough pairs found")

    # Figure B4: Reheating test
    if args.reheat_dir:
        print("\nFigure B4: Reheating test...")
        reheat_metrics = load_metrics(args.reheat_dir)
        print(f"  {len(reheat_metrics)} reheating eval points")
        plot_reheating(metrics, reheat_metrics, out_dir)

    # Figure B5: LR threshold map
    if args.reheat_dirs:
        print("\nFigure B5: LR threshold map...")
        reheat_lr_pairs = []
        for entry in args.reheat_dirs.split(","):
            lr_str, rdir = entry.strip().split(":")
            reheat_lr_pairs.append((float(lr_str), rdir.strip()))
        plot_lr_threshold(reheat_lr_pairs, out_dir)

    print(f"\nDone! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
