#!/usr/bin/env python3
"""
Basin geometry analysis for the TinyStories probe experiment.

Reports continuous basin depth D_σ = mean p_ood after 300-step relaxation
(not binary classification, since fresh-optimizer relaxation drops p_ood ~0.25).

Produces:
  Table:     Trunk-only location drift norms and ratios (Step 2)
  Figure B6: Basin depth curves D_σ for peaks and late-LM (Step 3)
  Table:     Basin depth D₀ and half-depth σ½ per checkpoint (Step 3)
  Figure B7: Switching manifold dimensionality via Gram-Schmidt residuals (Step 4)

Usage:
  # Quick sanity test (~2h)
  python basin_geometry.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ \\
      --sigma-grid 0,0.10 --n-trials 2 --skip-manifold

  # Full phase 1 (peaks + LM, ~10h, run as nohup)
  nohup python -u basin_geometry.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ \\
      --phase 1 --n-trials 4 > basin_geometry.log 2>&1 &

  # Resume after interruption
  python basin_geometry.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ --resume
"""

import argparse
import json
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
from attractor_analysis import (
    TRUNK_PATTERN,
    flatten_state_dict_filtered,
    load_checkpoint,
    load_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

PEAKS   = [2800, 5000, 6400]
TROUGHS = [2000, 5400, 6800]
LATE_LM = [10000]

# Relaxation protocol R (locked)
RELAX_STEPS     = 300
RELAX_LR        = 6e-4    # constant, no cosine/warmup
RELAX_WD        = 0.5
RELAX_LAMBDA    = 4.0
RELAX_GRAD_CLIP = 1.0
RELAX_BATCH     = 64

# Legacy regime classifier thresholds (kept for informational labels only)
PROBE_P_THRESH   = 0.60
PROBE_NLL_THRESH = 2.0

# Perturbation defaults
DEFAULT_SIGMAS = [0.0, 0.01, 0.03, 0.10, 0.30]
DEFAULT_TRIALS = 8

# Extended peak-trough pairs for manifold dimensionality (chronological order)
EXTENDED_PAIRS = [
    (1800, 2000), (2400, 2600), (2800, 2000), (3800, 4000),
    (4200, 4400), (4600, 4800), (5000, 5400), (5600, 5800),
    (6000, 6200), (6400, 6800),
]


# ═══════════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

def classify_regime(p_ood, nll_ood):
    """Binary regime classifier: 'probe' or 'lm'."""
    if p_ood >= PROBE_P_THRESH and nll_ood <= PROBE_NLL_THRESH:
        return "probe"
    return "lm"


def make_optimizer(model, lr, wd):
    """Create AdamW with standard param groups (LN/bias → wd=0)."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


def perturb_trunk(state_dict, sigma, rng_seed):
    """Apply trunk-only RMS-scaled Gaussian noise.

    For keys matching TRUNK_PATTERN: W' = W + σ·rms(W)·Z, Z~N(0,I).
    Other keys: cloned unchanged.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(rng_seed)

    new_sd = {}
    for key, val in state_dict.items():
        if sigma > 0 and TRUNK_PATTERN.match(key):
            w = val.float()
            rms = w.pow(2).mean().sqrt()
            noise = torch.randn(w.shape, generator=rng, dtype=torch.float32)
            new_sd[key] = (w + sigma * rms * noise).to(val.dtype)
        else:
            new_sd[key] = val.clone()
    return new_sd


def format_eta(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# Relaxation protocol
# ═══════════════════════════════════════════════════════════════════════════

def relax(model, train_loader, probe_ood, device,
          n_steps=RELAX_STEPS, lr=RELAX_LR, lam=RELAX_LAMBDA,
          wd=RELAX_WD, grad_clip=RELAX_GRAD_CLIP):
    """Run standardized relaxation protocol R.

    Constant LR, no warmup, no cosine. grad_accum=1.
    Loss = lm_loss + λ·p_loss via probe_mask.

    Model weights must already be loaded before calling.

    Returns dict with p_ood, nll_ood, lm_ood, regime.
    """
    opt = make_optimizer(model, lr, wd)
    ce_none = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(train_loader)

    model.train()
    for step in range(n_steps):
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
        loss = lm_loss + lam * p_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    # Evaluate final state
    p_ood = evaluate_probe(model, probe_ood, device)
    nll_ood, lm_ood, _ = evaluate_probe_nll(model, probe_ood, device)
    regime = classify_regime(p_ood, nll_ood)

    return {
        "p_ood": round(p_ood, 4),
        "nll_ood": round(nll_ood, 4),
        "lm_ood": round(lm_ood, 4),
        "regime": regime,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Trunk-only location drift
# ═══════════════════════════════════════════════════════════════════════════

def compute_drift_table(run_dir, peaks, troughs, late_lm):
    """Compute trunk-only location drift between regime checkpoints.

    Args:
        peaks: list of peak checkpoint steps
        troughs: list of trough checkpoint steps
        late_lm: list of late LM checkpoint steps
    """
    run_dir = Path(run_dir)

    # Load and flatten all anchors
    all_steps = sorted(set(peaks + troughs + late_lm))
    thetas = {}
    for step in all_steps:
        ckpt = load_checkpoint(run_dir, step)
        thetas[step] = flatten_state_dict_filtered(ckpt["model_state_dict"])
        del ckpt
        print(f"    Loaded step {step}")

    # Regime centers
    C_P = torch.stack([thetas[s] for s in peaks]).mean(dim=0)
    C_T = torch.stack([thetas[s] for s in troughs]).mean(dim=0)

    def norm(a, b):
        return float((thetas[a] - thetas[b]).norm())

    # Within-peak drifts (consecutive pairs)
    within_peak = []
    for i in range(1, len(peaks)):
        within_peak.append({
            "pair": f"{peaks[i]}-{peaks[i-1]}",
            "norm": norm(peaks[i], peaks[i-1]),
        })
    # Within-trough drifts (consecutive pairs)
    within_trough = []
    for i in range(1, len(troughs)):
        within_trough.append({
            "pair": f"{troughs[i]}-{troughs[i-1]}",
            "norm": norm(troughs[i], troughs[i-1]),
        })

    # Between-regime separation: pair each peak with nearest trough
    between = []
    for ps in peaks:
        closest_ts = min(troughs, key=lambda t: abs(t - ps))
        between.append({
            "pair": f"{ps}-{closest_ts}",
            "norm": norm(ps, closest_ts),
        })

    center_sep = float((C_P - C_T).norm())
    mean_within = float(np.mean([d["norm"] for d in within_peak + within_trough])) if (within_peak or within_trough) else 0.0
    mean_between = float(np.mean([d["norm"] for d in between])) if between else 0.0

    # Distance from late LM to each center
    lm_step = late_lm[0]
    dist_lm_to_cp = float((thetas[lm_step] - C_P).norm())
    dist_lm_to_ct = float((thetas[lm_step] - C_T).norm())

    result = {
        "within_peak": within_peak,
        "within_trough": within_trough,
        "between": between,
        "center_separation": center_sep,
        "mean_within": mean_within,
        "mean_between": mean_between,
        "drift_ratio_between_over_within": float(mean_between / mean_within) if mean_within > 0 else float("inf"),
        f"dist_lm{lm_step}_to_C_P": dist_lm_to_cp,
        f"dist_lm{lm_step}_to_C_T": dist_lm_to_ct,
    }
    return result


def print_drift_table(drift):
    """Print formatted drift table."""
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║           Trunk-only Location Drift Table           ║")
    print("  ╠══════════════════════════════════════════════════════╣")

    print("  ║  Within-peak drifts:                                ║")
    for d in drift["within_peak"]:
        print(f"  ║    ‖θ_{d['pair']}‖ = {d['norm']:>10.2f}                    ║")
    print("  ║  Within-trough drifts:                              ║")
    for d in drift["within_trough"]:
        print(f"  ║    ‖θ_{d['pair']}‖ = {d['norm']:>10.2f}                    ║")

    print("  ║  Between-regime separation:                         ║")
    for d in drift["between"]:
        print(f"  ║    ‖θ_{d['pair']}‖ = {d['norm']:>10.2f}                    ║")

    print(f"  ║  ‖C_P − C_T‖          = {drift['center_separation']:>10.2f}                    ║")
    print(f"  ║  Mean within           = {drift['mean_within']:>10.2f}                    ║")
    print(f"  ║  Mean between          = {drift['mean_between']:>10.2f}                    ║")
    print(f"  ║  Ratio (between/within)= {drift['drift_ratio_between_over_within']:>10.4f}                    ║")
    # Find the lm distance keys dynamically
    lm_cp_key = [k for k in drift if k.startswith("dist_lm") and k.endswith("_to_C_P")]
    lm_ct_key = [k for k in drift if k.startswith("dist_lm") and k.endswith("_to_C_T")]
    if lm_cp_key:
        lm_step = lm_cp_key[0].replace("dist_lm", "").replace("_to_C_P", "")
        print(f"  ║  ‖θ_{lm_step} − C_P‖      = {drift[lm_cp_key[0]]:>10.2f}                    ║")
    if lm_ct_key:
        print(f"  ║  ‖θ_{lm_step} − C_T‖      = {drift[lm_ct_key[0]]:>10.2f}                    ║")
    print("  ╚══════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Basin volume curves
# ═══════════════════════════════════════════════════════════════════════════

def run_perturbation_trial(base_sd, model, train_loader, probe_ood, device,
                           sigma, trial_seed):
    """Single perturbation + relaxation trial."""
    perturbed_sd = perturb_trunk(base_sd, sigma, trial_seed)
    model.load_state_dict(perturbed_sd)
    del perturbed_sd
    result = relax(model, train_loader, probe_ood, device)
    result["sigma"] = sigma
    result["trial_seed"] = trial_seed
    return result


def compute_basin_volume(run_dir, checkpoints, sigmas, n_trials,
                         model, train_loader, probe_ood, device,
                         out_dir, resume=False):
    """Compute basin volume curves V_σ for a set of checkpoints.

    Saves incremental progress to basin_volume_progress.json.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    progress_path = out_dir / "basin_volume_progress.json"

    # Load existing progress
    completed = {}
    if resume and progress_path.exists():
        with open(progress_path) as f:
            completed = json.load(f)
        print(f"  Resuming: {sum(len(v) for v in completed.values())} (ckpt, sigma) blocks done")

    total_blocks = len(checkpoints) * len(sigmas)
    total_trials = total_blocks * n_trials
    trials_done = 0
    t0 = time.time()

    # Count already-done trials for progress
    for ckpt_key in completed:
        for sig_key in completed[ckpt_key]:
            trials_done += len(completed[ckpt_key][sig_key]["trials"])

    for ckpt_step in checkpoints:
        ckpt_key = str(ckpt_step)
        if ckpt_key not in completed:
            completed[ckpt_key] = {}

        # Load base state_dict once per checkpoint
        ckpt = load_checkpoint(run_dir, ckpt_step)
        base_sd = ckpt["model_state_dict"]
        del ckpt
        print(f"\n  Checkpoint {ckpt_step}:")

        for sigma in sigmas:
            sig_key = f"{sigma:.4f}"
            if sig_key in completed.get(ckpt_key, {}):
                print(f"    σ={sigma:.4f}: already done, skipping")
                continue

            trials = []
            for trial_idx in range(n_trials):
                trial_seed = (ckpt_step * 10000 + int(sigma * 10000) * 100 + trial_idx)

                result = run_perturbation_trial(
                    base_sd, model, train_loader, probe_ood, device,
                    sigma, trial_seed,
                )
                trials.append(result)
                trials_done += 1

                elapsed = time.time() - t0
                eta = (elapsed / max(trials_done, 1)) * (total_trials - trials_done)
                print(f"    [{trials_done}/{total_trials}] σ={sigma:.4f} "
                      f"trial={trial_idx+1}/{n_trials} "
                      f"regime={result['regime']} p_ood={result['p_ood']:.3f} "
                      f"elapsed={format_eta(elapsed)} ETA={format_eta(eta)}")

            # Compute continuous basin depth + legacy binary volume
            p_vals = [t["p_ood"] for t in trials]
            nll_vals = [t["nll_ood"] for t in trials]
            n_probe = sum(1 for t in trials if t["regime"] == "probe")
            v_probe = n_probe / len(trials)

            completed[ckpt_key][sig_key] = {
                "mean_p_ood": round(float(np.mean(p_vals)), 4),
                "std_p_ood": round(float(np.std(p_vals)), 4),
                "mean_nll_ood": round(float(np.mean(nll_vals)), 4),
                "v_probe": v_probe,
                "n_probe": n_probe,
                "n_trials": len(trials),
                "trials": trials,
            }

            # Incremental save
            with open(progress_path, "w") as f:
                json.dump(completed, f, indent=2)

        del base_sd

    return completed


def compute_half_depth_sigma(volume_data):
    """Find σ½ = smallest σ where mean p_ood < D₀/2 (linear interpolation).

    D₀ = mean p_ood at σ=0 (basin depth at checkpoint).
    σ½ = noise tolerance: how much perturbation the probe basin can absorb
    before losing half its depth.

    Returns (sigma_half, D0).
    """
    points = []
    for sig_key, data in volume_data.items():
        p = data.get("mean_p_ood")
        if p is None:
            # Backward compatibility with old files
            p = np.mean([t["p_ood"] for t in data["trials"]])
        points.append((float(sig_key), p))
    points.sort()

    if not points:
        return float("inf"), 0.0

    D0 = points[0][1]  # basin depth at σ=0
    if D0 <= 0:
        return 0.0, D0

    threshold = D0 / 2.0

    # Never drops below threshold
    if all(v >= threshold for _, v in points):
        return float("inf"), D0

    # Already below at σ=0
    if points[0][1] < threshold:
        return 0.0, D0

    # Linear interpolation
    for i in range(len(points) - 1):
        s0, v0 = points[i]
        s1, v1 = points[i + 1]
        if v0 >= threshold and v1 < threshold:
            frac = (threshold - v0) / (v1 - v0)
            return s0 + frac * (s1 - s0), D0

    return float("inf"), D0


def print_depth_table(completed, checkpoints):
    """Print basin depth summary table."""
    print("\n  Basin depth table (D_σ = mean p_ood after 300-step relaxation):")
    print(f"  {'Step':>8s}  {'D₀':>8s}  {'±':>6s}  {'σ½':>8s}  {'regime':>8s}")
    print(f"  {'-'*46}")
    for step in checkpoints:
        ckpt_key = str(step)
        if ckpt_key not in completed:
            continue
        vol_data = completed[ckpt_key]
        sig_half, D0 = compute_half_depth_sigma(vol_data)

        # Get std at σ=0
        z_data = vol_data.get("0.0000", vol_data.get("0.0", {}))
        std0 = z_data.get("std_p_ood")
        if std0 is None:
            std0 = float(np.std([t["p_ood"] for t in z_data["trials"]]))

        regime = "probe" if D0 >= PROBE_P_THRESH else "lm"
        sh_str = f"{sig_half:.4f}" if sig_half < float("inf") else ">max_σ"
        print(f"  {step:>8d}  {D0:>8.3f}  {std0:>6.3f}  {sh_str:>8s}  {regime:>8s}")


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Switching manifold dimensionality
# ═══════════════════════════════════════════════════════════════════════════

def compute_manifold_dimensionality(run_dir, pairs):
    """Compute switching manifold dimensionality via Gram-Schmidt residuals.

    For each event displacement Δ_e = θ_peak − θ_trough (trunk-only),
    compute residual after projecting out the span of previous directions.
    """
    run_dir = Path(run_dir)

    # Load all direction vectors
    print(f"  Loading {len(pairs)} displacement vectors...")
    deltas = []
    delta_norms = []
    for pk, tr in pairs:
        ckpt_pk = load_checkpoint(run_dir, pk)
        ckpt_tr = load_checkpoint(run_dir, tr)
        theta_pk = flatten_state_dict_filtered(ckpt_pk["model_state_dict"])
        theta_tr = flatten_state_dict_filtered(ckpt_tr["model_state_dict"])
        d = theta_pk - theta_tr
        delta_norms.append(float(d.norm()))
        deltas.append(d)
        del ckpt_pk, ckpt_tr, theta_pk, theta_tr
        print(f"    ({pk},{tr}): ‖Δ‖ = {delta_norms[-1]:.2f}")

    n = len(deltas)

    # Full cosine gram matrix
    print("  Computing cosine gram matrix...")
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gram[i, j] = float(
                (deltas[i] @ deltas[j]) / (deltas[i].norm() * deltas[j].norm() + 1e-12)
            )

    # Gram-Schmidt residuals
    print("  Computing Gram-Schmidt residuals...")
    # residuals[e] = list of r_e(k) for k = 0, 1, ..., n-1
    # r_e(0) = 1.0 (no projection), r_e(k) = residual after projecting out first k basis vectors
    basis = []  # orthonormal basis vectors
    residuals = []

    for e in range(n):
        d = deltas[e].clone()
        d_norm = d.norm()
        r_vals = [1.0]  # k=0: no projection

        for k in range(len(basis)):
            # Project out basis[k]
            # (we accumulate projection: proj = sum_j <d, basis_j> * basis_j)
            pass

        # Compute r_e(k) for each k = 1, ..., current_basis_size
        # More efficient: project out all basis vectors one by one
        proj_d = d.clone()
        for k in range(len(basis)):
            coeff = float(proj_d @ basis[k])
            proj_d = proj_d - coeff * basis[k]
            r_vals.append(float(proj_d.norm() / (d_norm + 1e-12)))

        residuals.append(r_vals)

        # Add this direction to the basis (after orthogonalizing)
        new_basis_vec = d.clone()
        for b in basis:
            new_basis_vec = new_basis_vec - (new_basis_vec @ b) * b
        new_norm = new_basis_vec.norm()
        if new_norm > 1e-8:
            basis.append(new_basis_vec / new_norm)
            print(f"    Direction {e+1} ({pairs[e]}): "
                  f"residual after {len(basis)-1} dims = {r_vals[-1]:.4f}, "
                  f"added to basis (dim={len(basis)})")
        else:
            print(f"    Direction {e+1} ({pairs[e]}): "
                  f"linearly dependent (residual = {float(new_norm / (d_norm + 1e-12)):.6f})")

    return {
        "pairs": [{"peak": pk, "trough": tr} for pk, tr in pairs],
        "delta_norms": delta_norms,
        "residuals": residuals,
        "gram_matrix": gram.tolist(),
        "effective_dim": len(basis),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_basin_depth(completed, checkpoints_peaks, checkpoints_troughs,
                     checkpoints_late, out_dir):
    """Plot Figure B6: Basin depth curves D_σ = mean p_ood after relaxation."""
    out_dir = Path(out_dir)

    has_troughs = len(checkpoints_troughs) > 0
    ncols = 2 if has_troughs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    def plot_panel(ax, ckpt_list, title):
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ckpt_list)))
        for step, c in zip(ckpt_list, colors):
            ckpt_key = str(step)
            if ckpt_key not in completed:
                continue
            vol_data = completed[ckpt_key]
            sigmas_vals = []
            means = []
            stds = []
            for sig_key in sorted(vol_data.keys(), key=lambda x: float(x)):
                block = vol_data[sig_key]
                sigmas_vals.append(float(sig_key))
                m = block.get("mean_p_ood")
                s = block.get("std_p_ood")
                if m is None:
                    p_vals = [t["p_ood"] for t in block["trials"]]
                    m = float(np.mean(p_vals))
                    s = float(np.std(p_vals))
                means.append(m)
                stds.append(s if s is not None else 0.0)

            means = np.array(means)
            stds = np.array(stds)
            se = stds / max(np.sqrt(block["n_trials"]), 1)  # stderr

            ax.plot(sigmas_vals, means, "o-", color=c, linewidth=1.5,
                    markersize=5, label=f"step {step}")
            ax.fill_between(sigmas_vals, means - se, means + se,
                            color=c, alpha=0.15)

        ax.set_xlabel("σ (noise scale)")
        ax.set_ylabel("D_σ (mean p_ood after relaxation)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.80)

    # Left panel: peaks + late LM
    plot_panel(axes[0], checkpoints_peaks + checkpoints_late, "Peaks + Late LM")

    # Right panel: troughs + late LM (if phase 2)
    if has_troughs:
        plot_panel(axes[1], checkpoints_troughs + checkpoints_late, "Troughs + Late LM")

    fig.suptitle("Figure B6: Basin Depth Curves D_σ = mean p_ood after relaxation",
                 fontsize=13)
    plt.tight_layout()
    path = out_dir / "fig_B6_basin_depth.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_manifold_dimensionality(md_results, out_dir):
    """Plot Figure B7: Gram-Schmidt residual curves."""
    out_dir = Path(out_dir)
    pairs = md_results["pairs"]
    residuals = md_results["residuals"]
    n = len(pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: r_e(k) for each direction
    ax1 = axes[0]
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, n))
    for e in range(n):
        r_vals = residuals[e]
        ks = list(range(len(r_vals)))
        pk, tr = pairs[e]["peak"], pairs[e]["trough"]
        ax1.plot(ks, r_vals, "o-", color=colors[e], linewidth=1.2,
                 markersize=4, alpha=0.8, label=f"({pk},{tr})")
    ax1.set_xlabel("k (number of basis vectors)")
    ax1.set_ylabel("r_e(k) (relative residual)")
    ax1.set_title("Gram-Schmidt residuals per direction")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Right: Cosine gram matrix as heatmap
    ax2 = axes[1]
    gram = np.array(md_results["gram_matrix"])
    im = ax2.imshow(gram, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    labels = [f"({p['peak']},{p['trough']})" for p in pairs]
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_title("Cosine similarity (Gram matrix)")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    fig.suptitle(f"Figure B7: Switching Manifold Dimensionality "
                 f"(effective dim ≈ {md_results['effective_dim']})", fontsize=14)
    plt.tight_layout()
    path = out_dir / "fig_B7_manifold_dim.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def load_manifest(manifest_path):
    """Load oscillation manifest and extract step markers.

    Returns (peaks, troughs, late_lm, extended_pairs) or raises on error.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run detect_oscillations.py first, or provide explicit --peaks/--troughs."
        )
    with open(manifest_path) as f:
        m = json.load(f)

    peaks = m["peaks"]
    troughs = m["troughs"]
    late_lm = m["late_lm"]
    extended_pairs = [(p["peak"], p["trough"]) for p in m["extended_pairs"]]

    if not peaks:
        raise ValueError(f"Manifest {manifest_path} has empty peaks list")
    if not troughs:
        raise ValueError(f"Manifest {manifest_path} has empty troughs list")

    return peaks, troughs, late_lm, extended_pairs


def main():
    parser = argparse.ArgumentParser(description="Basin geometry analysis")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to pilot run directory with checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset construction (must match training seed)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1=peaks+LM only, 2=peaks+troughs+LM")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Trials per (checkpoint, sigma) (default: {DEFAULT_TRIALS})")
    parser.add_argument("--sigma-grid", type=str, default=None,
                        help="Comma-separated sigma values, e.g. '0,0.03,0.10,0.30'")
    parser.add_argument("--peaks", type=str, default=None,
                        help="Comma-separated peak checkpoint steps, e.g. '2800,5000,6400'")
    parser.add_argument("--troughs", type=str, default=None,
                        help="Comma-separated trough checkpoint steps, e.g. '2000,5400,6800'")
    parser.add_argument("--late-lm", type=str, default=None,
                        help="Comma-separated late LM steps, e.g. '10000'")
    parser.add_argument("--extended-pairs", type=str, default=None,
                        help="Comma-separated peak:trough pairs for manifold dim, "
                             "e.g. '2800:2000,5000:5400'")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to oscillation_manifest.json (overrides --peaks/--troughs/etc)")
    parser.add_argument("--dump-config", action="store_true",
                        help="Print resolved config to stdout and save to analysis/resolved_config.json")
    parser.add_argument("--skip-drift", action="store_true",
                        help="Skip drift table (Step 2)")
    parser.add_argument("--skip-volume", action="store_true",
                        help="Skip basin volume curves (Step 3)")
    parser.add_argument("--skip-manifold", action="store_true",
                        help="Skip manifold dimensionality (Step 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume volume computation from progress file")
    parser.add_argument("--codewords", type=str, default=None,
                        help="Path to codewords JSON (default: <run-dir>/codewords.json if exists)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    seed = args.seed

    # Parse sigma grid
    if args.sigma_grid:
        sigmas = [float(s.strip()) for s in args.sigma_grid.split(",")]
    else:
        sigmas = DEFAULT_SIGMAS

    # ── Resolve checkpoint markers ─────────────────────────────────────
    # Priority: --manifest > explicit CLI args > module-level defaults
    extended_pairs_resolved = None

    if args.manifest:
        # Manifest mode: load everything from manifest (fail loudly if missing)
        ckpt_peaks, ckpt_troughs, late_lm, extended_pairs_resolved = load_manifest(args.manifest)
        print(f"  Loaded manifest from {args.manifest}")

        # Use representative subset if available (noise-calibrated)
        with open(args.manifest) as _mf:
            _manifest_full = json.load(_mf)
        rep = _manifest_full.get("representative", {})
        if rep.get("basin_depth_steps"):
            rep_steps = set(rep["basin_depth_steps"])
            ckpt_peaks = sorted(s for s in ckpt_peaks if s in rep_steps)
            ckpt_troughs = sorted(s for s in ckpt_troughs if s in rep_steps)
            late_lm = sorted(s for s in late_lm if s in rep_steps)
            # Add any rep steps not in peaks/troughs as late_lm
            accounted = set(ckpt_peaks) | set(ckpt_troughs) | set(late_lm)
            for s in rep_steps - accounted:
                late_lm.append(s)
            late_lm = sorted(set(late_lm))
            print(f"  Using representative subset: peaks={ckpt_peaks}, "
                  f"troughs={ckpt_troughs}, late_lm={late_lm}")
        del _manifest_full
    else:
        # Explicit CLI args or module-level defaults
        if args.peaks:
            ckpt_peaks = [int(s.strip()) for s in args.peaks.split(",")]
        else:
            ckpt_peaks = PEAKS

        if args.troughs:
            ckpt_troughs_all = [int(s.strip()) for s in args.troughs.split(",")]
        else:
            ckpt_troughs_all = TROUGHS

        if args.late_lm:
            late_lm = [int(s.strip()) for s in args.late_lm.split(",")]
        else:
            late_lm = LATE_LM

        if args.extended_pairs:
            extended_pairs_resolved = []
            for entry in args.extended_pairs.split(","):
                pk, tr = entry.strip().split(":")
                extended_pairs_resolved.append((int(pk), int(tr)))
        # else: stays None, will use EXTENDED_PAIRS default below

        ckpt_troughs = ckpt_troughs_all

    # Build checkpoint list based on phase
    if args.phase == 1:
        checkpoints = ckpt_peaks + late_lm
        ckpt_troughs_for_volume = []
    else:
        checkpoints = ckpt_peaks + ckpt_troughs + late_lm
        ckpt_troughs_for_volume = ckpt_troughs

    # Resolve extended_pairs for manifold analysis
    if extended_pairs_resolved is None:
        extended_pairs_resolved = EXTENDED_PAIRS

    print("=" * 70)
    print("  BASIN GEOMETRY ANALYSIS")
    print(f"  Run dir:      {run_dir}")
    print(f"  Seed:         {seed}")
    print(f"  Phase:        {args.phase} ({'peaks+LM' if args.phase == 1 else 'all'})")
    print(f"  Peaks:        {ckpt_peaks}")
    print(f"  Troughs:      {ckpt_troughs}")
    print(f"  Late LM:      {late_lm}")
    print(f"  Checkpoints:  {checkpoints}")
    print(f"  Sigma grid:   {sigmas}")
    print(f"  Trials/point: {args.n_trials}")
    n_runs = len(checkpoints) * len(sigmas) * args.n_trials
    print(f"  Total runs:   {n_runs}")
    if args.manifest:
        print(f"  Manifest:     {args.manifest}")
    print("=" * 70)

    # ── --dump-config: save resolved parameters ────────────────────────
    if args.dump_config:
        resolved = {
            "seed": seed,
            "run_dir": str(run_dir),
            "phase": args.phase,
            "peaks": ckpt_peaks,
            "troughs": ckpt_troughs,
            "late_lm": late_lm,
            "checkpoints": checkpoints,
            "sigmas": sigmas,
            "n_trials": args.n_trials,
            "extended_pairs": [list(p) for p in extended_pairs_resolved],
            "manifest": args.manifest,
            "relax_steps": RELAX_STEPS,
            "relax_lr": RELAX_LR,
            "relax_wd": RELAX_WD,
            "relax_lambda": RELAX_LAMBDA,
        }
        config_path = out_dir / "resolved_config.json"
        with open(config_path, "w") as f:
            json.dump(resolved, f, indent=2)
        print(f"\n  Resolved config saved to {config_path}")
        print(json.dumps(resolved, indent=2))

    # ── Setup (shared across all steps) ────────────────────────────────
    device = get_device()
    print(f"\n  Device: {device}")

    # Build datasets once
    cfg = Config(
        seed=seed, p_probe=0.10, batch_size=RELAX_BATCH,
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )
    # Resolve codewords path: explicit arg > run_dir/codewords.json > generate fresh
    cw_path = args.codewords
    if cw_path is None:
        default_cw = run_dir / "codewords.json"
        if default_cw.exists():
            cw_path = str(default_cw)
    print("  Building datasets...")
    data = build_datasets(cfg, codewords_path=cw_path)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    probe_ood = data["probe_eval_ood"]

    train_loader = DataLoader(
        data["train_dataset"], batch_size=RELAX_BATCH,
        shuffle=True, drop_last=True, num_workers=0,
    )

    # Create model once
    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Step 2: Drift table ────────────────────────────────────────────
    if not args.skip_drift:
        print("\n" + "─" * 70)
        print("  Step 2: Location drift table")
        print("─" * 70)
        drift = compute_drift_table(run_dir, ckpt_peaks, ckpt_troughs, late_lm)
        print_drift_table(drift)
        with open(out_dir / "drift_table.json", "w") as f:
            json.dump(drift, f, indent=2)
        print(f"  Saved {out_dir / 'drift_table.json'}")

    # ── Step 3: Basin volume curves ────────────────────────────────────
    if not args.skip_volume:
        print("\n" + "─" * 70)
        print("  Step 3: Basin volume curves")
        print("─" * 70)

        completed = compute_basin_volume(
            run_dir, checkpoints, sigmas, args.n_trials,
            model, train_loader, probe_ood, device,
            out_dir, resume=args.resume,
        )

        print_depth_table(completed, checkpoints)
        plot_basin_depth(completed, ckpt_peaks, ckpt_troughs_for_volume, late_lm, out_dir)

        # Save final results
        with open(out_dir / "basin_geometry_results.json", "w") as f:
            json.dump(completed, f, indent=2)
        print(f"  Saved {out_dir / 'basin_geometry_results.json'}")

    # ── Step 4: Manifold dimensionality ────────────────────────────────
    if not args.skip_manifold:
        print("\n" + "─" * 70)
        print("  Step 4: Switching manifold dimensionality")
        print("─" * 70)

        md = compute_manifold_dimensionality(run_dir, extended_pairs_resolved)
        plot_manifold_dimensionality(md, out_dir)

        with open(out_dir / "manifold_dimensionality.json", "w") as f:
            json.dump(md, f, indent=2)
        print(f"  Saved {out_dir / 'manifold_dimensionality.json'}")

    print(f"\nDone! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
