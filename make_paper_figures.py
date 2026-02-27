#!/usr/bin/env python3
"""
Generate publication-quality figures for the backbone paper.
Produces 4 figures: A (decomposition), B (alignment), C (Fisher), D (reheating).
"""

import json, sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

RUN = Path("runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY")
ANALYSIS = RUN / "analysis"
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

LAMBDA_STEP = 4000
COLORS = {
    "backbone": "#2166ac",   # blue
    "residual": "#b2182b",   # red
    "pood":     "#666666",   # grey
    "gradient": "#999999",   # light grey
    "update":   "#2166ac",   # blue
    "fisher_bb": "#2166ac",
    "fisher_sw": "#b2182b",
    "fisher_pc2": "#4dac26",
    "fisher_rnd": "#999999",
    "lr1":  "#d62728",       # red
    "lr2":  "#2166ac",       # blue
    "lr3":  "#4dac26",       # green
}


# ═══════════════════════════════════════════════════════════════════════
# FIGURE A: Backbone decomposition — a(t), ||r(t)||, p_ood
# ═══════════════════════════════════════════════════════════════════════
def figure_a():
    print("Figure A: Backbone decomposition...")
    with open(ANALYSIS / "trajectory_pca_uncentered.json") as f:
        data = json.load(f)

    br = data["backbone_residual"]
    steps = np.array(br["steps"])
    p_ood = np.array(br["p_ood"])

    # Use block 0 (representative)
    blk = br["blocks"]["0"]
    a_t = np.array(blk["a_t"])
    r_norm = np.array(blk["r_norm"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True,
                                     gridspec_kw={"hspace": 0.08})

    # Top panel: a(t) — backbone coordinate
    ax1.plot(steps, a_t, color=COLORS["backbone"], linewidth=1.8,
             label=r"$a(t)$ (backbone coord.)")
    ax1.set_ylabel(r"$a(t) = \langle\theta(t)-\theta(0),\, \mathbf{v}_\mathrm{b}\rangle$",
                    color=COLORS["backbone"])
    ax1.tick_params(axis="y", labelcolor=COLORS["backbone"])
    ax1.axvline(LAMBDA_STEP, color="0.75", ls="--", lw=0.8, zorder=0)
    ax1.text(LAMBDA_STEP + 100, ax1.get_ylim()[1] * 0.9,
             r"$\lambda$: 2→4", fontsize=8, color="0.5")

    # Overlay p_ood on right axis
    ax1r = ax1.twinx()
    ax1r.plot(steps, p_ood, color=COLORS["pood"], linewidth=0.9, alpha=0.6,
              ls="--", label=r"$p_\mathrm{ood}$")
    ax1r.set_ylabel(r"$p_\mathrm{ood}$", color=COLORS["pood"])
    ax1r.tick_params(axis="y", labelcolor=COLORS["pood"])
    ax1r.set_ylim(-0.05, 0.95)

    # Bottom panel: ||r(t)|| — residual norm
    ax2.plot(steps, r_norm, color=COLORS["residual"], linewidth=1.8,
             label=r"$\|\mathbf{r}(t)\|$ (residual)")
    ax2.set_ylabel(r"$\|\mathbf{r}(t)\|$", color=COLORS["residual"])
    ax2.tick_params(axis="y", labelcolor=COLORS["residual"])
    ax2.set_xlabel("Training step")
    ax2.axvline(LAMBDA_STEP, color="0.75", ls="--", lw=0.8, zorder=0)

    ax2r = ax2.twinx()
    ax2r.plot(steps, p_ood, color=COLORS["pood"], linewidth=0.9, alpha=0.6,
              ls="--")
    ax2r.set_ylabel(r"$p_\mathrm{ood}$", color=COLORS["pood"])
    ax2r.tick_params(axis="y", labelcolor=COLORS["pood"])
    ax2r.set_ylim(-0.05, 0.95)

    # Annotations
    ax1.annotate("monotone drift", xy=(8000, a_t[-5]),
                 fontsize=8, color=COLORS["backbone"], ha="center")
    ax2.annotate("oscillatory", xy=(3000, max(r_norm)*0.85),
                 fontsize=8, color=COLORS["residual"], ha="center")

    for ax in [ax1, ax2]:
        ax.set_xlim(0, steps[-1])

    fig.suptitle("Backbone–Residual Decomposition (Block 0, seed 42)",
                 fontsize=13, y=0.98)
    out = OUTDIR / "fig_A_backbone_decomposition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE B: Update alignment vs gradient alignment (the linchpin)
# Requires re-computation from checkpoints
# ═══════════════════════════════════════════════════════════════════════
def figure_b():
    """
    Compute |cos(g_t, v_b)| and |cos(u_t, v_b)| from checkpoints,
    then plot them side by side.
    """
    print("Figure B: Update vs gradient alignment...")
    import torch
    from config import Config, get_device
    from dataset import build_datasets
    from model import GPTModel as GPT

    device = get_device()
    cfg = Config(
        seed=42, n_layer=8, d_model=512, n_head=16, d_ff=2048,
        weight_decay=0.5, lr=0.001, total_steps=10000, eval_every=200,
        warmup_steps=1500, batch_size=64, grad_accum_steps=2, p_probe=0.10,
    )
    data = build_datasets(cfg, codewords_path=str(RUN / "codewords.json"))
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)

    # Load checkpoints
    ckpts = sorted(RUN.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"  Found {len(ckpts)} checkpoints")

    # ── Step 1: Compute backbone via uncentered PCA on block 0 ──────
    def extract_block0(state_dict):
        """Extract flattened block-0 parameters (matches flatten_block filter)."""
        parts = []
        for k in sorted(state_dict.keys()):
            if k.startswith("blocks.0.") and not k.endswith(".attn.bias"):
                parts.append(state_dict[k].cpu().float().flatten())
        return torch.cat(parts)

    theta0 = extract_block0(torch.load(ckpts[0], map_location="cpu")["model_state_dict"])
    D = theta0.numel()
    print(f"  Block 0 dim: {D:,}")

    # Build drift matrix
    drifts = []
    steps_all = []
    for cp in ckpts:
        sd = torch.load(cp, map_location="cpu")["model_state_dict"]
        theta_t = extract_block0(sd)
        drifts.append((theta_t - theta0).numpy())
        steps_all.append(int(cp.stem.split("_")[1]))  # step number from filename

    X = np.stack(drifts)  # (T, D)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    v_b = Vt[0]  # backbone direction
    print(f"  PC1 variance ratio: {S[0]**2 / (S**2).sum():.3f}")

    # ── Step 2: Compute update alignment |cos(u_t, v_b)| ───────────
    update_steps = []
    update_cos_abs = []
    update_cos_signed = []

    for i in range(1, len(ckpts)):
        u = drifts[i] - drifts[i - 1]  # 200-step update
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            continue
        cos_val = np.dot(u, v_b) / u_norm
        update_steps.append(steps_all[i])
        update_cos_abs.append(abs(cos_val))
        update_cos_signed.append(cos_val)

    # ── Step 3: Compute gradient alignment |cos(g_t, v_b)| ─────────
    # Sample 4 checkpoints across training, compute per-batch gradient alignment
    sample_indices = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
    sample_indices = [i for i in sample_indices if i < len(ckpts)]

    grad_steps = []
    grad_cos_mean = []
    grad_cos_std = []

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )

    v_b_torch = torch.tensor(v_b, dtype=torch.float32, device=device)

    for idx in sample_indices:
        cp = ckpts[idx]
        step = steps_all[idx]
        sd = torch.load(cp, map_location="cpu")
        model = GPT(vocab_size, cfg.seq_len, cfg.d_model, cfg.n_layer,
                    cfg.n_head, cfg.d_ff, cfg.dropout).to(device)
        model.load_state_dict(sd["model_state_dict"])
        model.train()

        # Determine lambda for this step
        cur_lambda = 4.0 if step >= 4000 else 2.0

        cos_vals = []
        n_batches = 8
        batch_iter = iter(train_loader)
        for _ in range(n_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)

            # Dataset returns (input_ids, targets, probe_mask) tuples
            ids, targets, probe_mask_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            model.zero_grad()
            logits, _ = model(ids)  # [B, seq_len, V]
            # targets already shifted: tokens[1:seq_len+1], so logits[t] predicts targets[t]
            loss_all = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), reduction="none"
            ).view(targets.shape)  # [B, seq_len]

            # probe_mask_b is [B, seq_len] bool — True on probe tokens
            pm = probe_mask_b.float()
            lm_w = 1.0 - pm
            lm_loss = (loss_all * lm_w).sum() / lm_w.sum().clamp(min=1)
            probe_loss = (loss_all * pm).sum() / pm.sum().clamp(min=1) if pm.sum() > 0 else torch.tensor(0.0, device=device)
            loss = lm_loss + cur_lambda * probe_loss
            loss.backward()

            # Extract block-0 gradient — must match flatten_block filter
            # (all blocks.0.* params EXCEPT attn.bias buffer)
            g_parts = []
            for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                if name.startswith("blocks.0.") and not name.endswith(".attn.bias"):
                    if param.grad is not None:
                        g_parts.append(param.grad.flatten())
                    else:
                        g_parts.append(torch.zeros(param.numel(), device=device))
            g = torch.cat(g_parts)
            g_norm = g.norm()
            if g_norm > 1e-12:
                cos_g = (g @ v_b_torch) / g_norm
                cos_vals.append(abs(cos_g.item()))

        grad_steps.append(step)
        grad_cos_mean.append(np.mean(cos_vals))
        grad_cos_std.append(np.std(cos_vals))

        del model
        if device == "mps":
            torch.mps.empty_cache()

        print(f"    step {step}: grad |cos|={np.mean(cos_vals):.4f}")

    # ── Step 4: Plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Gradient alignment (with error bars)
    ax.errorbar(grad_steps, grad_cos_mean, yerr=grad_cos_std,
                fmt="s", color=COLORS["gradient"], markersize=5,
                capsize=3, linewidth=1.0, markeredgecolor="0.4",
                label=r"$|\cos(\mathbf{g}_t,\, \mathbf{v}_\mathrm{b})|$ (per-batch gradient)",
                zorder=3)

    # Update alignment
    ax.plot(update_steps, update_cos_abs, "o-", color=COLORS["update"],
            markersize=3, linewidth=1.5,
            label=r"$|\cos(\mathbf{u}_t,\, \mathbf{v}_\mathrm{b})|$ (200-step update)",
            zorder=4)

    # Noise floor
    D_eff = D
    noise_floor = math.sqrt(2 / (math.pi * D_eff))
    ax.axhline(noise_floor, color="0.8", ls=":", lw=0.8, zorder=1)
    ax.text(500, noise_floor + 0.003, f"random noise floor ({noise_floor:.1e})",
            fontsize=7, color="0.6")

    # λ transition
    ax.axvline(LAMBDA_STEP, color="0.75", ls="--", lw=0.8, zorder=0)
    ax.text(LAMBDA_STEP + 100, 0.33, r"$\lambda$: 2→4", fontsize=8, color="0.5")

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$|\cos|$")
    ax.set_xlim(0, 10200)
    ax.set_ylim(-0.01, 0.38)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Gradient vs. Optimizer-Update Alignment with Backbone (Block 0, seed 42)")

    out = OUTDIR / "fig_B_alignment_linchpin.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE C: Fisher stiffening over time
# ═══════════════════════════════════════════════════════════════════════
def figure_c():
    print("Figure C: Fisher stiffening...")
    with open(ANALYSIS / "backbone_fisher_analysis.json") as f:
        data = json.load(f)

    results = data["results"]
    steps = [r["step"] for r in results]
    q_bb = [r["rayleigh_backbone"] for r in results]
    q_sw = [r["rayleigh_switch"] for r in results]
    q_pc2 = [r["rayleigh_pc2"] for r in results]
    q_rnd = [r["anisotropy_q_random_mean"] for r in results]
    aniso = [r["anisotropy_ratio"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True,
                                     gridspec_kw={"hspace": 0.10,
                                                  "height_ratios": [2, 1]})

    # Top: Rayleigh quotients (log scale)
    ax1.semilogy(steps, q_bb, "o-", color=COLORS["fisher_bb"], markersize=6,
                 linewidth=1.8, label=r"$q(\mathbf{v}_\mathrm{b})$ (backbone)")
    ax1.semilogy(steps, q_sw, "s-", color=COLORS["fisher_sw"], markersize=5,
                 linewidth=1.2, label=r"$q(\mathbf{v}_\mathrm{sw})$ (switch)")
    ax1.semilogy(steps, q_pc2, "^-", color=COLORS["fisher_pc2"], markersize=5,
                 linewidth=1.2, label=r"$q(\mathbf{v}_\mathrm{PC2})$")
    ax1.semilogy(steps, q_rnd, "D-", color=COLORS["fisher_rnd"], markersize=4,
                 linewidth=1.0, label=r"$\mathbb{E}[q(\mathbf{w}_\perp)]$ (random)")

    ax1.axvline(LAMBDA_STEP, color="0.75", ls="--", lw=0.8, zorder=0)
    ax1.set_ylabel(r"Rayleigh quotient $q(\mathbf{v})$")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax1.set_title("Fisher Curvature Along Key Directions (seed 42)")

    # Annotate 3 orders of magnitude
    ax1.annotate("", xy=(steps[-1], q_bb[-1]), xytext=(steps[-1], q_bb[0]),
                 arrowprops=dict(arrowstyle="<->", color=COLORS["fisher_bb"],
                                 lw=1.5))
    ax1.text(steps[-1] + 200, math.sqrt(q_bb[0] * q_bb[-1]),
             r"${\sim}10^3\times$", fontsize=9, color=COLORS["fisher_bb"],
             va="center")

    # Bottom: Anisotropy ratio
    ax2.plot(steps, aniso, "o-", color="#e66101", markersize=6, linewidth=1.8)
    ax2.axhline(1.0, color="0.8", ls=":", lw=0.8)
    ax2.axvline(LAMBDA_STEP, color="0.75", ls="--", lw=0.8, zorder=0)
    ax2.set_ylabel(r"Anisotropy $\alpha$")
    ax2.set_xlabel("Training step")
    ax2.text(LAMBDA_STEP + 100, max(aniso) * 0.85, r"$\lambda$: 2→4",
             fontsize=8, color="0.5")

    # Mark peak
    peak_idx = np.argmax(aniso)
    ax2.annotate(f"{aniso[peak_idx]:.1f}×",
                 xy=(steps[peak_idx], aniso[peak_idx]),
                 xytext=(steps[peak_idx] + 800, aniso[peak_idx] - 1),
                 fontsize=9, color="#e66101",
                 arrowprops=dict(arrowstyle="->", color="#e66101", lw=1.0))

    out = OUTDIR / "fig_C_fisher_stiffening.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE D: Reheating trajectories (3 LRs)
# ═══════════════════════════════════════════════════════════════════════
def figure_d():
    print("Figure D: Reheating trajectories...")

    # Load original training p_ood for context
    with open(ANALYSIS / "trajectory_pca_uncentered.json") as f:
        pca_data = json.load(f)
    train_steps = pca_data["backbone_residual"]["steps"]
    train_pood = pca_data["backbone_residual"]["p_ood"]

    # Load reheating data for 3 LRs
    lr_configs = [
        ("1e-3",  "runs/pilot_wd0.5_lr0.001_lp4.0_s42",  COLORS["lr1"]),
        ("6e-4",  "runs/pilot_wd0.5_lr0.0006_lp4.0_s42", COLORS["lr2"]),
        ("3e-4",  "runs/pilot_wd0.5_lr0.0003_lp4.0_s42", COLORS["lr3"]),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Training trajectory (faded)
    ax.plot(train_steps, train_pood, color="0.7", linewidth=1.0, alpha=0.7,
            label="Training", zorder=1)

    # Reheating vertical line
    ax.axvline(10000, color="0.5", ls="--", lw=1.0, zorder=0)
    ax.text(10050, 0.85, "reheating\nstart", fontsize=8, color="0.5",
            ha="left", va="top")

    for lr_label, run_dir, color in lr_configs:
        with open(f"{run_dir}/pilot_metrics.json") as f:
            metrics = json.load(f)
        reheat_steps = [m["step"] for m in metrics]
        reheat_pood = [m["probe_ood_acc"] for m in metrics]

        # Offset to training timeline
        reheat_steps_abs = [10000 + s for s in reheat_steps]

        ax.plot(reheat_steps_abs, reheat_pood, "o-", color=color,
                markersize=3, linewidth=1.5,
                label=rf"$\eta = {lr_label}$", zorder=3)

        # Mark peak
        peak_idx = np.argmax(reheat_pood)
        ax.plot(reheat_steps_abs[peak_idx], reheat_pood[peak_idx],
                "*", color=color, markersize=10, zorder=5)

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$p_\mathrm{ood}$ (probe accuracy)")
    ax.set_xlim(0, 12200)
    ax.set_ylim(-0.05, 0.90)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_title("Reheating: Probe Basin Re-Entry (seed 42)")

    out = OUTDIR / "fig_D_reheating.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.chdir("/Users/yongzhongxu/mini_gpt")

    which = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B", "C", "D"]

    if "A" in which: figure_a()
    if "B" in which: figure_b()
    if "C" in which: figure_c()
    if "D" in which: figure_d()

    print("\nAll requested figures generated in figures/")
