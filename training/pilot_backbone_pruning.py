import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Backbone-Anchor Pruning: 3-phase training strategy for mini-GPT.

Phase 1 (Steps 0–freeze_step):  Free training. Accumulates trajectory
    snapshots every `snapshot_every` steps for online SVD computation.
Phase 2 (freeze_step+):  Freeze backbone projection a(t) for Block 0 & 1.
    Locks the slow/stable LM drift, reducing gradient noise on the ridge.
Phase 3 (prune_step+):  Aggressive transverse pruning on all blocks.
    Zeros out low-energy residual components orthogonal to backbone.

Usage:
    python pilot_backbone_pruning.py --seed 42 --lambda-probe 42 --steps 10000
    python pilot_backbone_pruning.py --seed 42 --lambda-probe 42 --steps 10000 \
        --prune-threshold 0.2 --freeze-step 4000 --prune-step 6000
"""

import argparse
import math
import time
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets


# ═══════════════════════════════════════════════════════════════════════════
# Trunk parameter utilities (matches backbone_decomposition.py convention)
# ═══════════════════════════════════════════════════════════════════════════

TRUNK_PATTERN = re.compile(
    r"blocks\.\d+\."
    r"(attn\.qkv\.weight|attn\.out_proj\.weight|mlp\.w_up\.weight|mlp\.w_down\.weight)"
)


def get_trunk_keys(model, block_idx):
    """Sorted trunk parameter names for a block (attn+MLP weights only)."""
    prefix = f"blocks.{block_idx}."
    return sorted(
        name for name, _ in model.named_parameters()
        if name.startswith(prefix) and TRUNK_PATTERN.match(name)
    )


@torch.no_grad()
def flatten_trunk(model, block_idx):
    """Flatten trunk params of a block into 1-D float32 tensor (on param device)."""
    parts = []
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        if name.startswith(f"blocks.{block_idx}.") and TRUNK_PATTERN.match(name):
            parts.append(param.data.reshape(-1).float())
    return torch.cat(parts)


@torch.no_grad()
def apply_trunk_delta(model, block_idx, delta):
    """Add a delta vector to trunk parameters of a block in-place."""
    offset = 0
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        if name.startswith(f"blocks.{block_idx}.") and TRUNK_PATTERN.match(name):
            numel = param.numel()
            d = delta[offset:offset + numel].reshape(param.shape).to(param.dtype)
            param.add_(d)
            offset += numel


# ═══════════════════════════════════════════════════════════════════════════
# Backbone computation and pruning
# ═══════════════════════════════════════════════════════════════════════════

def compute_backbone_vectors(snapshot_deltas, n_blocks):
    """Compute v_backbone per block via row-normalized uncentered SVD.

    Args:
        snapshot_deltas: dict block_idx -> list of 1-D numpy arrays (delta from init)
        n_blocks: number of transformer blocks

    Returns:
        v_backbone: dict block_idx -> 1-D torch tensor (unit-norm PC1)
    """
    v_backbone = {}
    for b in range(n_blocks):
        deltas = snapshot_deltas[b]
        if len(deltas) < 2:
            d = len(deltas[0]) if deltas else 1
            v = np.random.randn(d).astype(np.float32)
            v /= np.linalg.norm(v)
            v_backbone[b] = torch.from_numpy(v)
            print(f"    Block {b}: <2 snapshots, using random direction")
            continue

        X = np.stack(deltas, axis=0)  # (T, D)

        # Row-normalize for direction-only SVD
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X_rn = X / norms

        # Uncentered SVD
        _, S, Vt = np.linalg.svd(X_rn, full_matrices=False)
        vb = Vt[0].copy()

        # Sign fix: ensure final delta projects positively
        if np.dot(deltas[-1], vb) < 0:
            vb = -vb

        var_explained = S[0] ** 2 / max(np.sum(S ** 2), 1e-12)
        print(f"    Block {b}: PC1 explains {var_explained * 100:.1f}% "
              f"(dim={X.shape[1]:,}, snapshots={X.shape[0]})")

        v_backbone[b] = torch.from_numpy(vb.astype(np.float32))

    return v_backbone


def backbone_residual_pruning(weights, initial_weights, backbone_v, threshold=0.1):
    """Prune transverse residual while preserving backbone rail.

    Args:
        weights: Current parameter vector (flattened)
        initial_weights: Parameter vector at step 0
        backbone_v: PC1 unit vector (the backbone direction)
        threshold: Fraction of max residual magnitude below which to zero out
    """
    drift = weights - initial_weights
    a_t = torch.dot(drift, backbone_v)
    backbone_component = a_t * backbone_v
    residual = drift - backbone_component

    residual_mask = torch.abs(residual) > (torch.max(torch.abs(residual)) * threshold)
    pruned_residual = residual * residual_mask

    new_weights = initial_weights + backbone_component + pruned_residual
    return new_weights


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities (from pilot.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_lr(step, cfg):
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr * max(coeff, 0.1)


@torch.no_grad()
def evaluate_lm(model, dataloader, device, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (input_ids, targets, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        _, loss = model(input_ids, targets)
        mask = targets != -100
        n_tokens = mask.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_probe(model, probe_dataset, device, batch_size=128):
    """Exact-match accuracy on codeword tokens."""
    model.eval()
    loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for input_ids, targets, probe_mask in loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        probe_mask = probe_mask.to(device)
        logits, _ = model(input_ids)
        preds = logits.argmax(dim=-1)
        match = (preds == targets) | ~probe_mask
        has_probe = probe_mask.any(dim=1)
        all_match = match.all(dim=1)
        correct += (all_match & has_probe).sum().item()
        total += has_probe.sum().item()
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_probe_nll(model, probe_dataset, device, batch_size=128):
    """Mean NLL at probe positions, LM NLL at non-probe positions, and ratio."""
    model.eval()
    loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
    probe_loss = 0.0
    probe_tokens = 0
    lm_loss = 0.0
    lm_tokens = 0
    ce = nn.CrossEntropyLoss(reduction='none')
    for input_ids, targets, probe_mask in loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        probe_mask = probe_mask.to(device)
        logits, _ = model(input_ids)
        loss_per_pos = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss_per_pos = loss_per_pos.view(targets.shape)
        pmask = probe_mask.bool()
        lm_mask = ~pmask & (targets != -100)
        probe_loss += loss_per_pos[pmask].sum().item()
        probe_tokens += pmask.sum().item()
        lm_loss += loss_per_pos[lm_mask].sum().item()
        lm_tokens += lm_mask.sum().item()
    model.train()
    p_nll = probe_loss / max(probe_tokens, 1)
    l_nll = lm_loss / max(lm_tokens, 1)
    ratio = p_nll / max(l_nll, 1e-8)
    return p_nll, l_nll, ratio


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backbone-Anchor Pruning Training")
    # Standard training args (defaults match last seed-42 run)
    parser.add_argument("--wd", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-probe", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--lambda-probe", type=float, default=2.0,
                        help="Probe loss weight: L = L_LM + lambda * L_probe")
    parser.add_argument("--lambda-probe2", type=float, default=None,
                        help="Second-phase lambda (after --lambda-step)")
    parser.add_argument("--lambda-step", type=int, default=4000,
                        help="Step to switch from lambda-probe to lambda-probe2")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    # Pruning-specific args
    parser.add_argument("--freeze-step", type=int, default=4000,
                        help="Step to freeze backbone projection (Phase 2)")
    parser.add_argument("--prune-step", type=int, default=6000,
                        help="Step to start transverse pruning (Phase 3)")
    parser.add_argument("--prune-threshold", type=float, default=0.1,
                        help="Residual pruning threshold (0.1 = zero out bottom 10%%)")
    parser.add_argument("--prune-every", type=int, default=1,
                        help="Apply pruning every N optimizer steps")
    parser.add_argument("--freeze-blocks", type=str, default="0,1",
                        help="Comma-separated block indices for backbone freezing")
    parser.add_argument("--prune-blocks", type=str, default="all",
                        help="Blocks for transverse pruning ('all' or comma-separated)")
    parser.add_argument("--snapshot-every", type=int, default=200,
                        help="Snapshot interval for trajectory SVD")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Checkpoint to continue from (preserves LR schedule)")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Step-0 checkpoint for theta_0 reference "
                             "(required with --continue-from)")
    args = parser.parse_args()

    device = get_device()
    cfg = Config(
        seed=args.seed,
        weight_decay=args.wd,
        total_steps=args.steps,
        batch_size=args.batch_size,
        p_probe=args.p_probe,
        eval_every=args.eval_every,
        lr=args.lr,
        warmup_steps=args.warmup,
    )
    if args.beta2 is not None:
        cfg.adam_beta2 = args.beta2
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.n_head is not None:
        cfg.n_head = args.n_head
    if args.d_ff is not None:
        cfg.d_ff = args.d_ff

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    freeze_blocks = [int(b) for b in args.freeze_blocks.split(",")]
    if args.prune_blocks == "all":
        prune_blocks = list(range(cfg.n_layer))
    else:
        prune_blocks = [int(b) for b in args.prune_blocks.split(",")]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (Path("runs") /
                   f"backbone_pruning_lp{args.lambda_probe}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  BACKBONE-ANCHOR PRUNING")
    print(f"  seed={args.seed}, wd={args.wd}, lr={args.lr}, "
          f"beta2={cfg.adam_beta2}, lambda={args.lambda_probe}")
    print(f"  Phase 1: Free training (steps 0-{args.freeze_step})")
    print(f"  Phase 2: Freeze backbone blocks {freeze_blocks} "
          f"(step {args.freeze_step}+)")
    print(f"  Phase 3: Transverse pruning threshold={args.prune_threshold} "
          f"(step {args.prune_step}+)")
    print(f"  Model: {cfg.n_layer}L, d={cfg.d_model}, h={cfg.n_head}, "
          f"ff={cfg.d_ff}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}\n")

    # ── Dataset ──
    data = build_datasets(cfg)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)

    cw_path = out_dir / "codewords.json"
    if not cw_path.exists():
        with open(cw_path, "w") as f:
            json.dump({"codewords": data["codewords"],
                       "count": len(data["codewords"])}, f, indent=2)

    # Save full config including pruning params
    cfg_dict = cfg.to_dict()
    cfg_dict["pruning"] = {
        "strategy": "backbone-anchor",
        "freeze_step": args.freeze_step,
        "prune_step": args.prune_step,
        "prune_threshold": args.prune_threshold,
        "prune_every": args.prune_every,
        "freeze_blocks": freeze_blocks,
        "prune_blocks": prune_blocks,
        "snapshot_every": args.snapshot_every,
        "lambda_probe": args.lambda_probe,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # ── Model ──
    model = GPTModel(
        vocab_size=vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    n_params = model.count_params()
    print(f"Model: {n_params:,} params")

    # ── Save initial weights theta_0 per block (trunk only) ──
    # When continuing, theta_0 must come from the original init checkpoint
    if args.init_ckpt:
        init_sd = torch.load(args.init_ckpt, map_location="cpu",
                             weights_only=True)["model_state_dict"]
        init_model = GPTModel(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            d_model=cfg.d_model, n_layer=cfg.n_layer,
            n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=cfg.dropout,
        )
        init_model.load_state_dict(init_sd)
        init_params = {}
        for b in range(cfg.n_layer):
            init_params[b] = flatten_trunk(init_model, b).cpu().clone()
        del init_model, init_sd
        print(f"Loaded theta_0 from {args.init_ckpt}")
    else:
        # Capture from freshly-initialized model (before any checkpoint load)
        init_params = {}
        for b in range(cfg.n_layer):
            init_params[b] = flatten_trunk(model, b).cpu().clone()
    trunk_dim = init_params[0].numel()
    print(f"Saved theta_0 for {cfg.n_layer} blocks "
          f"(trunk dim={trunk_dim:,} per block)")

    # ── Resume from checkpoint ──
    start_step = 0
    if args.continue_from:
        ckpt = torch.load(args.continue_from, map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt["step"]
        print(f"  Continuing from {args.continue_from} "
              f"(step {start_step}), LR schedule preserved")
        del ckpt

    # Save step-0 checkpoint (only if starting fresh)
    if start_step == 0:
        torch.save({"step": 0, "model_state_dict": model.state_dict()},
                   out_dir / "ckpt_000000.pt")

    # ── Optimizer ──
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)

    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        data["val_dataset"], batch_size=cfg.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )

    probe_in = data["probe_eval_in"]
    probe_ood = data["probe_eval_ood"]

    lambda_base = args.lambda_probe
    lambda_phase2 = (args.lambda_probe2 if args.lambda_probe2 is not None
                     else lambda_base)
    lambda_step = args.lambda_step
    ce_none = nn.CrossEntropyLoss(reduction='none')

    # ── Trajectory snapshot accumulators (Phase 1) ──
    snapshot_deltas = {b: [] for b in range(cfg.n_layer)}

    # ── Backbone state (populated at freeze_step) ──
    v_backbone = None       # dict: block_idx -> unit-norm PC1 tensor
    frozen_a = {}            # dict: block_idx -> scalar (a(t) at freeze_step)
    init_params_dev = None   # init_params moved to device (lazy)

    # If resuming at or past freeze_step, compute backbone immediately
    # from existing checkpoints in the continue-from directory
    if start_step >= args.freeze_step and args.continue_from:
        src_dir = Path(args.continue_from).parent
        ckpt_files = sorted(src_dir.glob("ckpt_*.pt"))
        avail_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)
        snap_steps = [s for s in avail_steps
                      if 0 < s <= args.freeze_step
                      and s % args.snapshot_every == 0]
        if snap_steps:
            print(f"\n  Loading {len(snap_steps)} snapshots from {src_dir} "
                  f"for backbone SVD...")
            for s in snap_steps:
                sd = torch.load(src_dir / f"ckpt_{s:06d}.pt",
                                map_location="cpu",
                                weights_only=True)["model_state_dict"]
                tmp_model = GPTModel(
                    vocab_size=vocab_size, seq_len=cfg.seq_len,
                    d_model=cfg.d_model, n_layer=cfg.n_layer,
                    n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=cfg.dropout,
                )
                tmp_model.load_state_dict(sd)
                for b in range(cfg.n_layer):
                    cur = flatten_trunk(tmp_model, b).cpu()
                    snapshot_deltas[b].append(
                        (cur - init_params[b]).numpy())
                del tmp_model, sd

            print(f"  Computing backbone vectors "
                  f"(SVD on {len(snap_steps)} snapshots)...")
            v_backbone = compute_backbone_vectors(
                snapshot_deltas, cfg.n_layer)
            for b in v_backbone:
                v_backbone[b] = v_backbone[b].to(device)
            init_params_dev = {b: init_params[b].to(device)
                               for b in init_params}

            with torch.no_grad():
                for b in freeze_blocks:
                    current = flatten_trunk(model, b)
                    drift = current - init_params_dev[b]
                    frozen_a[b] = torch.dot(
                        drift, v_backbone[b]).item()
                    print(f"    Block {b}: a_frozen = {frozen_a[b]:.6f}")

            del snapshot_deltas
            snapshot_deltas = None
            print(f"  Backbone ready. Resuming from step {start_step}.\n")
        else:
            print(f"  WARNING: No snapshots found in {src_dir} for SVD. "
                  f"Will compute on-the-fly if steps < freeze_step.")

    metrics = []
    if start_step > 0:
        metrics_path = out_dir / "pilot_metrics.json"
        if metrics_path.exists():
            all_prev = json.load(open(metrics_path))
            metrics = [m for m in all_prev if m["step"] <= start_step]
            print(f"  Loaded {len(metrics)} metric entries up to "
                  f"step {start_step}")
    data_iter = iter(train_loader)
    train_loss_accum = 0.0
    n_accum = 0
    t0 = time.time()
    best_ood = max((m["probe_ood_acc"] for m in metrics), default=0.0)
    pruning_log = []

    print(f"\n{'step':>7s}  {'train':>7s}  {'val':>7s}  "
          f"{'p_in':>6s}  {'p_ood':>6s}  {'nll_in':>7s}  {'nll_ood':>7s}  "
          f"{'lm_ood':>7s}  {'lam':>4s}  {'phase':>6s}  {'lr':>9s}  "
          f"{'min':>5s}")
    print("-" * 105)

    for step in range(1, cfg.total_steps + 1):
        if step <= start_step:
            continue
        model.train()

        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        cur_lambda = lambda_phase2 if step >= lambda_step else lambda_base

        # ── Forward / backward ──
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            try:
                input_ids, targets, probe_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets, probe_mask = next(data_iter)

            input_ids, targets = input_ids.to(device), targets.to(device)

            if cur_lambda > 0:
                probe_mask = probe_mask.to(device)
                logits, _ = model(input_ids)
                loss_flat = ce_none(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_all = loss_flat.view(targets.shape)
                pmask = probe_mask.bool()
                lm_mask = ~pmask & (targets != -100)
                lm_loss = (loss_all[lm_mask].mean() if lm_mask.any()
                           else torch.tensor(0.0, device=device))
                p_loss = (loss_all[pmask].mean() if pmask.any()
                          else torch.tensor(0.0, device=device))
                loss = lm_loss + cur_lambda * p_loss
            else:
                _, loss = model(input_ids, targets)

            loss = loss / cfg.grad_accum_steps
            loss.backward()
            train_loss_accum += loss.item() * cfg.grad_accum_steps
            n_accum += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # ════════════════════════════════════════════════════════════════
        # Backbone-Anchor Pruning Logic
        # ════════════════════════════════════════════════════════════════

        phase = "free"

        # Phase 1: Accumulate trajectory snapshots for SVD
        if (v_backbone is None
                and snapshot_deltas is not None
                and step < args.freeze_step
                and step % args.snapshot_every == 0):
            with torch.no_grad():
                for b in range(cfg.n_layer):
                    current = flatten_trunk(model, b).cpu()
                    delta = (current - init_params[b]).numpy()
                    snapshot_deltas[b].append(delta)

        # Transition: Compute backbone vectors at freeze_step
        if step == args.freeze_step and v_backbone is None:
            n_snaps = len(snapshot_deltas[0])
            print(f"\n  >>> PHASE 2 TRANSITION (step {step})")
            print(f"  >>> Computing backbone vectors "
                  f"(SVD on {n_snaps} snapshots)...")

            v_backbone = compute_backbone_vectors(
                snapshot_deltas, cfg.n_layer)

            # Move backbone vectors and init params to device
            for b in v_backbone:
                v_backbone[b] = v_backbone[b].to(device)
            init_params_dev = {b: init_params[b].to(device)
                               for b in init_params}

            # Record a(t) at freeze_step for blocks to be frozen
            with torch.no_grad():
                for b in freeze_blocks:
                    current = flatten_trunk(model, b)
                    drift = current - init_params_dev[b]
                    frozen_a[b] = torch.dot(
                        drift, v_backbone[b]).item()
                    print(f"    Block {b}: a_frozen = {frozen_a[b]:.6f}")

            # Free snapshot memory
            del snapshot_deltas
            print(f"  >>> Backbone frozen for blocks {freeze_blocks}")
            print(f"  >>> Transverse pruning starts at step "
                  f"{args.prune_step}\n")

        # Phase 3: Transverse pruning (applied BEFORE backbone freeze
        # so the freeze correction is authoritative)
        if (v_backbone is not None
                and step >= args.prune_step
                and step % args.prune_every == 0):
            phase = "prune"
            n_zeroed = 0
            n_total = 0
            with torch.no_grad():
                for b in prune_blocks:
                    current = flatten_trunk(model, b)
                    new_weights = backbone_residual_pruning(
                        current, init_params_dev[b], v_backbone[b],
                        threshold=args.prune_threshold)
                    delta = new_weights - current
                    n_zeroed += (delta.abs() > 1e-10).sum().item()
                    n_total += current.numel()
                    apply_trunk_delta(model, b, delta)

            if step % args.eval_every == 0:
                pruning_log.append({
                    "step": step,
                    "n_modified": n_zeroed,
                    "n_total": n_total,
                    "frac_modified": n_zeroed / max(n_total, 1),
                })

        # Phase 2: Freeze backbone projection (always last so it's exact)
        if v_backbone is not None and step >= args.freeze_step:
            if phase == "free":
                phase = "freeze"
            with torch.no_grad():
                for b in freeze_blocks:
                    current = flatten_trunk(model, b)
                    drift = current - init_params_dev[b]
                    a_t = torch.dot(drift, v_backbone[b]).item()
                    correction = (frozen_a[b] - a_t) * v_backbone[b]
                    apply_trunk_delta(model, b, correction)

        # ════════════════════════════════════════════════════════════════
        # Evaluation & Checkpointing
        # ════════════════════════════════════════════════════════════════

        if step % args.eval_every == 0 or step == 1:
            avg_train = train_loss_accum / max(n_accum, 1)
            val_loss = evaluate_lm(model, val_loader, device)
            pin_acc = evaluate_probe(model, probe_in, device)
            pood_acc = evaluate_probe(model, probe_ood, device)
            nll_in, _, _ = evaluate_probe_nll(model, probe_in, device)
            nll_ood, lm_ood, r_ood = evaluate_probe_nll(
                model, probe_ood, device)
            elapsed = (time.time() - t0) / 60

            rec = {
                "step": step,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "probe_in_acc": pin_acc,
                "probe_ood_acc": pood_acc,
                "nll_in": nll_in,
                "nll_ood": nll_ood,
                "lm_ood": lm_ood,
                "r_ood": r_ood,
                "cur_lambda": cur_lambda,
                "lr": lr,
                "phase": phase,
            }
            metrics.append(rec)

            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            torch.save({"step": step,
                         "model_state_dict": model.state_dict()}, ckpt_path)

            if pood_acc > best_ood:
                best_ood = pood_acc

            print(f"{step:7d}  {avg_train:7.4f}  {val_loss:7.4f}  "
                  f"{pin_acc:6.3f}  {pood_acc:6.3f}  {nll_in:7.3f}  "
                  f"{nll_ood:7.3f}  {lm_ood:7.3f}  {cur_lambda:4.1f}  "
                  f"{phase:>6s}  {lr:9.2e}  {elapsed:5.1f}")

            train_loss_accum = 0.0
            n_accum = 0

            # Early stopping: OOD acc >= 0.8 for 5 consecutive evals
            recent = [m["probe_ood_acc"] for m in metrics[-5:]]
            if len(recent) >= 5 and all(r >= 0.8 for r in recent):
                print(f"\n  >>> GROKKED at step {step}! "
                      f"(probe_ood >= 0.8 for 5 evals)")
                break

    # ── Save results ──
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if pruning_log:
        with open(out_dir / "pruning_stats.json", "w") as f:
            json.dump(pruning_log, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  BACKBONE-ANCHOR PRUNING COMPLETE")
    print(f"  Best OOD probe acc: {best_ood:.4f}")
    print(f"  Steps: {metrics[-1]['step']}")
    print(f"  Final val loss: {metrics[-1]['val_loss']:.4f}")
    print(f"  Saved to {out_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
