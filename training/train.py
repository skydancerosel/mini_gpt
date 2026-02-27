import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import _paths  # noqa
"""
Training loop for TinyStories + probe experiment.

Handles:
- AdamW with cosine decay + warmup
- Gradient accumulation
- Periodic evaluation (val loss, probe_in accuracy, probe_ood accuracy)
- Checkpointing with weight snapshots
- Geometric monitoring (commutator defect + PCA)
- Capability control (subspace suppression)
"""

import math
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, get_device
from model import GPTModel
from geometric import (
    commutator_defect_from_dataloader, UpdatePCA, flatten_params,
    detect_emergence, detect_defect_onset,
)
from control import SubspaceSuppressor


# ═══════════════════════════════════════════════════════════════════════════
# Learning rate schedule
# ═══════════════════════════════════════════════════════════════════════════

def get_lr(step, cfg):
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.lr * max(coeff, 0.1)  # floor at 10% of peak lr


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_lm(model, dataloader, device, max_batches=50):
    """Evaluate language modeling loss on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (input_ids, targets, _) in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        mask = targets != -100
        n_tokens = mask.sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_probe(model, probe_dataset, device, batch_size=64):
    """
    Evaluate probe accuracy (exact match on codeword tokens).

    For each example, check if the model predicts the correct tokens
    at the positions marked by probe_mask.
    """
    model.eval()
    loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0
    total_examples = 0

    for input_ids, targets, probe_mask in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        probe_mask = probe_mask.to(device)

        logits, _ = model(input_ids)  # [B, T, V]
        preds = logits.argmax(dim=-1)  # [B, T]

        # For each example, check if ALL probe positions are correct
        for b in range(input_ids.shape[0]):
            mask_b = probe_mask[b]
            if mask_b.sum() == 0:
                continue
            pred_tokens = preds[b][mask_b]
            target_tokens = targets[b][mask_b]
            if (pred_tokens == target_tokens).all():
                total_correct += 1
            total_examples += 1

    model.train()
    accuracy = total_correct / max(total_examples, 1)
    return accuracy


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: Config, data: dict, run_tag: str = "default"):
    """
    Main training loop.

    Args:
        cfg: experiment configuration
        data: dict from build_datasets() with tokenizer, datasets, etc.
        run_tag: string identifier for this run
    """
    device = get_device()
    out_dir = Path(cfg.log_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)

    # ── Model ──────────────────────────────────────────────────────────
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
    print(f"Model: {n_params:,} parameters, device={device}")

    # ── Optimizer ──────────────────────────────────────────────────────
    # Separate weight decay for non-bias, non-layernorm params
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

    # ── Data loaders ───────────────────────────────────────────────────
    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        data["val_dataset"], batch_size=cfg.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )
    # Separate loader for commutator measurements
    comm_loader = DataLoader(
        data["train_dataset"], batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )

    probe_eval_in = data["probe_eval_in"]
    probe_eval_ood = data["probe_eval_ood"]

    # ── Geometric monitoring ───────────────────────────────────────────
    update_pca = UpdatePCA(n_components=cfg.n_pca_components)
    update_pca.record(model)  # initial snapshot

    # ── Capability control ─────────────────────────────────────────────
    suppressor = SubspaceSuppressor(
        model, control_k=cfg.control_k, lam=cfg.control_lambda,
    )

    # Probe-only loader for control (reuse probe_eval_in data)
    probe_loader = DataLoader(
        probe_eval_in, batch_size=cfg.control_batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    probe_iter = iter(probe_loader)
    lm_iter = iter(train_loader)

    # ── Logging ────────────────────────────────────────────────────────
    metrics_log = []
    defect_log = []
    pca_log = []
    prev_ckpt_params = None
    train_loss_accum = 0.0
    n_loss_accum = 0

    t0 = time.time()
    data_iter = iter(train_loader)
    micro_step = 0

    # ── Save config ────────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    print(f"\nTraining: {cfg.total_steps} steps, eval every {cfg.eval_every}, "
          f"ckpt every {cfg.ckpt_every}")
    print(f"Output: {out_dir}/")
    print(f"Weight decay: {cfg.weight_decay}, lr: {cfg.lr}, "
          f"control_lambda: {cfg.control_lambda}")
    print()

    for step in range(1, cfg.total_steps + 1):
        model.train()

        # Update learning rate
        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──────────────────────────────────────
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            try:
                input_ids, targets, probe_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets, probe_mask = next(data_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            _, loss = model(input_ids, targets)
            loss = loss / cfg.grad_accum_steps
            loss.backward()

            train_loss_accum += loss.item() * cfg.grad_accum_steps
            n_loss_accum += 1

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # ── Capability control ─────────────────────────────────────────
        if cfg.control_lambda > 0 and step % cfg.control_every == 0:
            try:
                lm_batch = next(lm_iter)
            except StopIteration:
                lm_iter = iter(train_loader)
                lm_batch = next(lm_iter)
            try:
                probe_batch = next(probe_iter)
            except StopIteration:
                probe_iter = iter(probe_loader)
                probe_batch = next(probe_iter)

            suppressor.update_basis(
                (lm_batch[0], lm_batch[1]),
                (probe_batch[0], probe_batch[1]),
                device,
            )

        if cfg.control_lambda > 0:
            suppressor.suppress_gradient()

        opt.step()

        # ── Evaluation ─────────────────────────────────────────────────
        if step % cfg.eval_every == 0 or step == 1:
            avg_train_loss = train_loss_accum / max(n_loss_accum, 1)

            val_loss = evaluate_lm(model, val_loader, device)
            probe_in_acc = evaluate_probe(model, probe_eval_in, device)
            probe_ood_acc = evaluate_probe(model, probe_eval_ood, device)

            elapsed = (time.time() - t0) / 60
            metrics = {
                "step": step,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "probe_in_acc": probe_in_acc,
                "probe_ood_acc": probe_ood_acc,
                "lr": lr,
                "elapsed_min": elapsed,
            }
            metrics_log.append(metrics)

            if step % (cfg.eval_every * 5) == 0 or step == 1:
                print(f"  step {step:6d} | "
                      f"train {avg_train_loss:.4f} | val {val_loss:.4f} | "
                      f"probe_in {probe_in_acc:.3f} | probe_ood {probe_ood_acc:.3f} | "
                      f"lr {lr:.2e} | {elapsed:.1f}m")

            train_loss_accum = 0.0
            n_loss_accum = 0

        # ── Checkpointing + geometric monitoring ───────────────────────
        if step % cfg.ckpt_every == 0:
            # Save model weights
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            }, ckpt_path)

            # PCA on weight updates
            update_pca.record(model)
            pca_result = update_pca.compute_pca()
            if pca_result is not None:
                pca_result["step"] = step
                pca_log.append(pca_result)

            # Commutator defect
            print(f"    Computing commutator defect at step {step}...")
            defect_result = commutator_defect_from_dataloader(
                model, comm_loader, device,
                eta=cfg.comm_eta, K=cfg.comm_k,
            )
            defect_entry = {
                "step": step,
                "defect_median": defect_result["defect_median"],
                "defect_p25": defect_result["defect_p25"],
                "defect_p75": defect_result["defect_p75"],
            }
            defect_log.append(defect_entry)
            print(f"    defect={defect_result['defect_median']:.4f} "
                  f"[{defect_result['defect_p25']:.4f}, {defect_result['defect_p75']:.4f}]")

            if pca_result:
                print(f"    PCA: k*95={pca_result['k_star_95']}, "
                      f"k*99={pca_result['k_star_99']}, "
                      f"PC1={pca_result['explained_variance_ratio'][0]:.3f}")

    # ── Post-training analysis ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  POST-TRAINING ANALYSIS")
    print(f"{'='*70}")

    # Emergence detection
    ood_trace = [(m["step"], m["probe_ood_acc"]) for m in metrics_log]
    emergence_step = detect_emergence(
        ood_trace, threshold=cfg.emergence_threshold,
        consecutive=cfg.emergence_consecutive,
    )

    # Defect onset detection
    defect_trace = [(d["step"], d["defect_median"]) for d in defect_log]
    defect_onset = detect_defect_onset(
        defect_trace,
        baseline_window=cfg.defect_baseline_window,
        sigma_mult=cfg.defect_sigma_mult,
        sustained=cfg.defect_sustained_evals,
    )

    lead_time = None
    if emergence_step is not None and defect_onset is not None:
        lead_time = emergence_step - defect_onset

    print(f"  Emergence step (probe_ood >= {cfg.emergence_threshold}): {emergence_step}")
    print(f"  Defect onset step: {defect_onset}")
    print(f"  Lead time: {lead_time}")

    # ── Save all results ───────────────────────────────────────────────
    results = {
        "cfg": cfg.to_dict(),
        "metrics": metrics_log,
        "defect_log": defect_log,
        "pca_log": pca_log,
        "emergence_step": emergence_step,
        "defect_onset": defect_onset,
        "lead_time": lead_time,
        "n_params": n_params,
    }

    torch.save(results, out_dir / "results.pt")

    # Also save metrics as JSON for easy inspection
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    return results
