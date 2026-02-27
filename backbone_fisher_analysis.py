#!/usr/bin/env python3
"""
Backbone-aware Fisher analysis (Step 10 additions: 10A, 10B, 10C).

Connects backbone geometry to loss-landscape curvature:

  10A) Rayleigh quotients along backbone, switch, and PC2 directions
       q(v) = v^T F v = (1/M) ‖Gv‖²
       Expected: q(v_b) large and stable (backbone = stiff/slow direction).

  10B) Anisotropy ratio
       anisotropy = q(v_b) / E_{u ⊥ v_b}[q(u)]
       Expected: ≫ 1 (backbone is special).

  10C) Fisher–backbone eigenvector overlap
       |⟨u_1, v_b⟩|  where u_1 = leading Fisher eigenvector
       Expected: high (backbone ≈ top curvature direction).

All computations use trunk-only parameters (TRUNK_PATTERN, ~25M params),
consistent with the existing Fisher analysis.

Outputs (in <run-dir>/analysis/):
  fig_rayleigh_quotients.png         Rayleigh quotients + anisotropy + overlap
  backbone_fisher_analysis.json      Full numerical results

Usage:
  python backbone_fisher_analysis.py \\
      --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s271/ --seed 271
"""

import argparse
import json
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
from attractor_analysis import (load_checkpoint, load_metrics,
                                 TRUNK_PATTERN, flatten_state_dict_filtered)
from fisher_analysis import collect_trunk_gradient, count_trunk_params


# ═════════════════════════════════════════════════════════════════════════
# Checkpoint selection (shared with backbone_gradient_analysis)
# ═════════════════════════════════════════════════════════════════════════

def auto_select_checkpoints(manifest, include_init=200):
    """Select 6–8 representative checkpoints from manifest."""
    rep = manifest.get("representative", {})
    peaks = set(manifest["peaks"])
    troughs = set(manifest["troughs"])

    candidates = []
    candidates.append(("init", include_init))

    ep = rep.get("early_peak")
    if ep:
        candidates.append(("early_peak", ep))
        for sp in manifest["switch_pairs"]:
            if sp["peak"] == ep:
                candidates.append(("early_trough", sp["trough"]))
                break

    tt = rep.get("transition_trough")
    if tt:
        candidates.append(("transition", tt))

    mp = rep.get("mid_peak")
    mt = rep.get("mid_trough")
    if mp:
        candidates.append(("mid_peak", mp))
    if mt:
        candidates.append(("mid_trough", mt))

    lp = rep.get("late_peak")
    if lp:
        candidates.append(("late", lp))

    candidates.append(("final", 10000))

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
# Trunk-level direction computation
# ═════════════════════════════════════════════════════════════════════════

def compute_trunk_directions(run_dir, manifest, step_stride=200):
    """Compute v_backbone, v_pc2, v_switch in trunk-only parameter space.

    Uses flatten_state_dict_filtered (TRUNK_PATTERN) for consistency
    with Fisher gradient collection.

    Returns dict with 'v_backbone', 'v_pc2', 'v_switch' (all unit-norm,
    D_trunk ≈ 25M), plus metadata.
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

    print(f"  Computing trunk directions: loading {len(steps)} checkpoints")

    # Load init
    sd0 = load_checkpoint(run_dir, steps[0])["model_state_dict"]
    theta0 = flatten_state_dict_filtered(sd0)
    D = theta0.numel()
    del sd0

    # Accumulate trunk-only drifts
    drifts = []
    for i, step in enumerate(steps[1:], 1):
        sd = load_checkpoint(run_dir, step)["model_state_dict"]
        theta_t = flatten_state_dict_filtered(sd)
        drifts.append(theta_t - theta0)
        del sd
        if i % 10 == 0:
            print(f"    {i}/{len(steps)-1} loaded")
    print(f"    {len(steps)-1}/{len(steps)-1} loaded")

    # Uncentered SVD
    X = torch.stack(drifts).numpy()  # (T-1, D)
    print(f"  SVD on ({X.shape[0]}, {X.shape[1]:,}) matrix...")
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    v_backbone = torch.from_numpy(Vt[0].copy()).float()
    v_pc2 = torch.from_numpy(Vt[1].copy()).float()

    var = S ** 2
    total_var = var.sum()
    pc1_pct = var[0] / max(total_var, 1e-12) * 100
    pc2_pct = var[1] / max(total_var, 1e-12) * 100
    print(f"  Trunk PC1={pc1_pct:.1f}%, PC2={pc2_pct:.1f}%, "
          f"dim={D:,}")

    # v_switch from manifest priority pair
    rep = manifest.get("representative", {})
    pp = rep.get("priority_pairs", manifest.get("switch_pairs", []))
    if pp:
        peak_step = pp[0]["peak"]
        trough_step = pp[0]["trough"]
    else:
        sp = manifest.get("switch_pairs", [])
        peak_step = sp[0]["peak"]
        trough_step = sp[0]["trough"]

    sd_peak = load_checkpoint(run_dir, peak_step)["model_state_dict"]
    sd_trough = load_checkpoint(run_dir, trough_step)["model_state_dict"]
    d_switch = (flatten_state_dict_filtered(sd_peak) -
                flatten_state_dict_filtered(sd_trough))
    v_switch = d_switch / d_switch.norm()
    del sd_peak, sd_trough

    cos_bb_sw = abs(float(torch.dot(v_backbone, v_switch)))
    print(f"  Switch pair: peak={peak_step} → trough={trough_step}")
    print(f"  |⟨v_backbone, v_switch⟩| = {cos_bb_sw:.6f}")

    return {
        "v_backbone": v_backbone,
        "v_pc2": v_pc2,
        "v_switch": v_switch,
        "n_params": D,
        "pc1_var_ratio": pc1_pct / 100,
        "pc2_var_ratio": pc2_pct / 100,
        "switch_pair": (peak_step, trough_step),
        "cos_backbone_switch": cos_bb_sw,
    }


# ═════════════════════════════════════════════════════════════════════════
# Gradient matrix collection
# ═════════════════════════════════════════════════════════════════════════

def collect_gradient_matrix(model, dataloader, device, n_batches,
                            lambda_probe):
    """Collect M trunk gradient vectors into a matrix G (M, D).

    Same loss as fisher_analysis.py: L = L_LM + lambda_probe * L_probe.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(dataloader)
    gradients = []

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
        g = collect_trunk_gradient(model)
        gradients.append(g)

        if (i + 1) % 8 == 0 or (i + 1) == n_batches:
            print(f"      Batch {i+1}/{n_batches}: loss={loss.item():.4f}")

    G = torch.stack(gradients)  # (M, D)
    return G


# ═════════════════════════════════════════════════════════════════════════
# 10A: Rayleigh quotients
# ═════════════════════════════════════════════════════════════════════════

def compute_rayleigh_quotient(G, v):
    """Rayleigh quotient q(v) = v^T F v = (1/M) ‖Gv‖².

    Args:
        G: (M, D) gradient matrix on CPU.
        v: (D,) unit-norm direction on CPU.
    Returns:
        float: q(v)
    """
    Gv = G @ v  # (M,)
    return float((Gv ** 2).sum() / G.shape[0])


# ═════════════════════════════════════════════════════════════════════════
# 10B: Anisotropy ratio
# ═════════════════════════════════════════════════════════════════════════

def compute_anisotropy_ratio(G, v_backbone, n_random=10, seed=42):
    """Anisotropy = q(v_backbone) / E[q(u) for u ⊥ v_backbone].

    Denominator approximated with n_random random orthogonal directions.
    """
    q_backbone = compute_rayleigh_quotient(G, v_backbone)

    D = v_backbone.numel()
    rng = torch.Generator().manual_seed(seed)
    q_randoms = []

    for _ in range(n_random):
        u = torch.randn(D, generator=rng)
        # Orthogonalize against v_backbone
        u = u - torch.dot(u, v_backbone) * v_backbone
        u = u / u.norm()
        q_randoms.append(compute_rayleigh_quotient(G, u))

    mean_q_random = float(np.mean(q_randoms))
    std_q_random = float(np.std(q_randoms))

    anisotropy = q_backbone / max(mean_q_random, 1e-30)

    return {
        "anisotropy_ratio": anisotropy,
        "q_backbone": q_backbone,
        "q_random_mean": mean_q_random,
        "q_random_std": std_q_random,
        "q_randoms": q_randoms,
    }


# ═════════════════════════════════════════════════════════════════════════
# 10C: Fisher–backbone eigenvector overlap
# ═════════════════════════════════════════════════════════════════════════

def compute_fisher_backbone_overlap(G, v_backbone):
    """Recover leading Fisher eigenvector u_1 and measure |⟨u_1, v_b⟩|.

    Uses Gram matrix trick:
      (1/M) G G^T has eigenvectors w_i (M-dim).
      u_i = G^T w_i / ‖G^T w_i‖  (D-dim Fisher eigenvector).
    """
    M, D = G.shape
    gram = (G @ G.T) / M  # (M, M)
    eigvals, eigvecs = torch.linalg.eigh(gram)  # ascending order

    # Top eigenvalue/vector
    lambda_1 = float(eigvals[-1])
    w_1 = eigvecs[:, -1]  # (M,)

    # Recover D-dimensional eigenvector
    u_1 = G.T @ w_1  # (D,)
    u_1 = u_1 / u_1.norm()

    overlap = abs(float(torch.dot(u_1, v_backbone)))

    # Also get trace for reference
    trace = float(eigvals.sum())

    return {
        "overlap_u1_backbone": overlap,
        "lambda_1": lambda_1,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════

def plot_backbone_fisher(results, out_dir):
    """2–3 panel figure: Rayleigh quotients, anisotropy, overlap vs step."""
    steps = [r["step"] for r in results]
    labels = [r["label"] for r in results]
    types = [r["type"] for r in results]

    has_rayleigh = "rayleigh_backbone" in results[0]
    has_aniso = "anisotropy_ratio" in results[0]
    has_overlap = "overlap_u1_backbone" in results[0]

    n_panels = sum([has_rayleigh, has_aniso, has_overlap])
    if n_panels == 0:
        print("  No data to plot — skipping Fisher figure.")
        return

    q_bb = [r.get("rayleigh_backbone", 0) for r in results] if has_rayleigh else []
    q_sw = [r.get("rayleigh_switch", 0) for r in results] if has_rayleigh else []
    q_p2 = [r.get("rayleigh_pc2", 0) for r in results] if has_rayleigh else []
    aniso = [r.get("anisotropy_ratio", 0) for r in results] if has_aniso else []
    overlap = [r.get("overlap_u1_backbone", 0) for r in results] if has_overlap else []

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4*n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # ── Panel: Rayleigh quotients ──────────────────────────────────
    if has_rayleigh:
        ax = axes[ax_idx]; ax_idx += 1
        ax.semilogy(steps, q_bb, "-o", color="steelblue", linewidth=2,
                    markersize=6, label=r"$q(v_{backbone})$")
        ax.semilogy(steps, q_sw, "-s", color="indianred", linewidth=2,
                    markersize=6, label=r"$q(v_{switch})$")
        ax.semilogy(steps, q_p2, "-^", color="seagreen", linewidth=2,
                    markersize=6, label=r"$q(v_{PC2})$")
        for i, (s, t) in enumerate(zip(steps, types)):
            if t == "peak":
                ax.axvline(s, color="red", alpha=0.15, lw=6)
            elif t == "trough":
                ax.axvline(s, color="green", alpha=0.15, lw=6)
        ax.set_ylabel(r"Rayleigh quotient $v^\top F v$", fontsize=12)
        ax.set_title("Rayleigh Quotients Along Key Directions (10A)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ── Panel: Anisotropy ratio ────────────────────────────────────
    if has_aniso:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(steps, aniso, "-o", color="darkorange", linewidth=2,
                markersize=6)
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="isotropic (=1)")
        ax.set_ylabel("Anisotropy ratio", fontsize=12)
        ax.set_title(r"Anisotropy: $q(v_b) \,/\, \mathbb{E}[q(u_\perp)]$ (10B)",
                      fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        for i, (s, t) in enumerate(zip(steps, types)):
            if t == "peak":
                ax.axvline(s, color="red", alpha=0.15, lw=6)
            elif t == "trough":
                ax.axvline(s, color="green", alpha=0.15, lw=6)

    # ── Panel: Fisher–backbone overlap ─────────────────────────────
    if has_overlap:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(steps, overlap, "-o", color="purple", linewidth=2, markersize=6)
        ax.set_ylabel(r"$|\langle u_1, v_{backbone} \rangle|$", fontsize=12)
        ax.set_title(r"Fisher Top Eigenvector–Backbone Overlap (10C)", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        for i, (s, t) in enumerate(zip(steps, types)):
            if t == "peak":
                ax.axvline(s, color="red", alpha=0.15, lw=6)
            elif t == "trough":
                ax.axvline(s, color="green", alpha=0.15, lw=6)

    axes[-1].set_xlabel("Training step", fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / "fig_rayleigh_quotients.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backbone-aware Fisher analysis (Step 10: 10A, 10B, 10C)"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Explicit comma-separated steps (overrides auto)")
    parser.add_argument("--n-batches", type=int, default=32,
                        help="Gradient samples M per checkpoint (default: 32)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-probe", type=float, default=2.0)
    parser.add_argument("--lambda-probe2", type=float, default=None,
                        help="Second-phase lambda (after --lambda-step)")
    parser.add_argument("--lambda-step", type=int, default=4000,
                        help="Step at which to switch lambda (default: 4000)")
    parser.add_argument("--pca-stride", type=int, default=200)
    parser.add_argument("--n-random-aniso", type=int, default=10,
                        help="Random directions for anisotropy (default: 10)")
    parser.add_argument("--skip-10a", action="store_true")
    parser.add_argument("--skip-10b", action="store_true")
    parser.add_argument("--skip-10c", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    device = get_device()

    print(f"Backbone Fisher Analysis")
    print(f"  run_dir:      {run_dir}")
    print(f"  device:       {device}")
    print(f"  n_batches:    {args.n_batches} (M)")
    print(f"  batch_size:   {args.batch_size}")
    lambda_base = args.lambda_probe
    lambda_phase2 = args.lambda_probe2 if args.lambda_probe2 is not None else lambda_base
    lambda_step = args.lambda_step
    print(f"  lambda:       base={lambda_base}, phase2={lambda_phase2}, switch@{lambda_step}")

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

    # ── Compute trunk-level directions ───────────────────────────────
    print(f"\n{'='*60}")
    print("Computing trunk-level directions (v_backbone, v_switch, v_pc2)")
    print(f"{'='*60}")
    dirs = compute_trunk_directions(
        run_dir, manifest, step_stride=args.pca_stride)

    v_backbone = dirs["v_backbone"]
    v_switch = dirs["v_switch"]
    v_pc2 = dirs["v_pc2"]

    # ── Build datasets ───────────────────────────────────────────────
    print(f"\nBuilding datasets...")
    cfg = Config(
        seed=42, p_probe=0.10, batch_size=args.batch_size,
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

    # ── Process each checkpoint ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing Rayleigh quotients, anisotropy, and Fisher overlap")
    print(f"{'='*60}")

    results = []
    total_t0 = time.time()

    for ci in checkpoint_info:
        step = ci["step"]
        label = ci["label"]
        print(f"\n  Step {step} ({label})")
        print(f"  {'─'*50}")

        t0 = time.time()

        # Load model
        ckpt = load_checkpoint(run_dir, step, device=device)
        model = GPTModel(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            d_model=cfg.d_model, n_layer=cfg.n_layer,
            n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt

        if not results:
            n_trunk = count_trunk_params(model)
            print(f"    Trunk params (D): {n_trunk:,}")

        # Collect gradient matrix G (M, D)
        cur_lambda = lambda_phase2 if step >= lambda_step else lambda_base
        print(f"    Collecting {args.n_batches} gradient samples (λ={cur_lambda})...")
        G = collect_gradient_matrix(
            model, train_loader, device,
            args.n_batches, cur_lambda)

        entry = {
            "step": step,
            "label": label,
            "type": ci["type"],
        }

        # 10A: Rayleigh quotients
        if not args.skip_10a:
            q_bb = compute_rayleigh_quotient(G, v_backbone)
            q_sw = compute_rayleigh_quotient(G, v_switch)
            q_p2 = compute_rayleigh_quotient(G, v_pc2)
            entry["rayleigh_backbone"] = q_bb
            entry["rayleigh_switch"] = q_sw
            entry["rayleigh_pc2"] = q_p2
            print(f"    q(v_bb)={q_bb:.6e}  q(v_sw)={q_sw:.6e}  "
                  f"q(v_pc2)={q_p2:.6e}")

        # 10B: Anisotropy ratio
        if not args.skip_10b:
            aniso = compute_anisotropy_ratio(
                G, v_backbone, n_random=args.n_random_aniso)
            entry["anisotropy_ratio"] = aniso["anisotropy_ratio"]
            entry["anisotropy_q_backbone"] = aniso["q_backbone"]
            entry["anisotropy_q_random_mean"] = aniso["q_random_mean"]
            entry["anisotropy_q_random_std"] = aniso["q_random_std"]
            print(f"    anisotropy = {aniso['anisotropy_ratio']:.1f}×  "
                  f"(q_bb={aniso['q_backbone']:.4e}, "
                  f"q_rand={aniso['q_random_mean']:.4e})")

        # 10C: Fisher–backbone eigenvector overlap
        if not args.skip_10c:
            fisher_ov = compute_fisher_backbone_overlap(G, v_backbone)
            entry["overlap_u1_backbone"] = fisher_ov["overlap_u1_backbone"]
            entry["lambda_1"] = fisher_ov["lambda_1"]
            entry["trace"] = fisher_ov["trace"]
            print(f"    |⟨u_1, v_bb⟩| = {fisher_ov['overlap_u1_backbone']:.4f}  "
                  f"λ_1 = {fisher_ov['lambda_1']:.4e}  "
                  f"trace = {fisher_ov['trace']:.4e}")

        elapsed = time.time() - t0
        entry["elapsed_sec"] = round(elapsed, 1)
        print(f"    [{elapsed:.1f}s]")
        results.append(entry)

        del model, G
        if device == "mps":
            torch.mps.empty_cache()

    total_elapsed = time.time() - total_t0
    print(f"\n  Total: {total_elapsed:.1f}s")

    # ── Plot ─────────────────────────────────────────────────────────
    if results and not (args.skip_10a and args.skip_10b and args.skip_10c):
        plot_backbone_fisher(results, out_dir)

    # ── Save JSON ────────────────────────────────────────────────────
    output = {
        "config": {
            "run_dir": str(run_dir),
            "seed": args.seed,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "lambda_probe": args.lambda_probe,
            "pca_stride": args.pca_stride,
            "n_random_aniso": args.n_random_aniso,
            "device": device,
        },
        "trunk_directions": {
            "n_params": dirs["n_params"],
            "pc1_var_ratio": dirs["pc1_var_ratio"],
            "pc2_var_ratio": dirs["pc2_var_ratio"],
            "switch_pair": list(dirs["switch_pair"]),
            "cos_backbone_switch": dirs["cos_backbone_switch"],
        },
        "results": results,
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    json_path = out_dir / "backbone_fisher_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")
    print(f"Done! Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
