#!/usr/bin/env python3
"""
Pilot run: does the model grok on the key-retrieval probe?

Lean training loop — no geometric monitoring, just LM + probe accuracy.
Evaluates more frequently to catch emergence timing.

Usage:
    python pilot.py                           # default: wd=3e-3, 50k steps
    python pilot.py --wd 1e-3 --steps 20000
"""

import argparse
import math
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets


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
    """Exact-match accuracy on codeword tokens (full eval set)."""
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


def main():
    parser = argparse.ArgumentParser(description="Pilot grokking check")
    parser.add_argument("--wd", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-probe", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--lambda-probe", type=float, default=0.0,
                        help="Probe loss weight: L = L_LM + lambda * L_probe")
    parser.add_argument("--lambda-probe2", type=float, default=None,
                        help="Second-phase lambda (after --lambda-step)")
    parser.add_argument("--lambda-step", type=int, default=4000,
                        help="Step at which to switch from lambda-probe to lambda-probe2")
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from (fresh optimizer, LR reset)")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Path to checkpoint to continue training from (same LR schedule, fresh optimizer)")
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

    out_dir = Path("runs") / f"pilot_wd{args.wd}_lr{args.lr}_lp{args.lambda_probe}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  PILOT: wd={args.wd}, lr={args.lr}, steps={args.steps}, p_probe={args.p_probe}, lambda={args.lambda_probe}")
    print(f"  Model: {cfg.n_layer}L, d={cfg.d_model}, h={cfg.n_head}, ff={cfg.d_ff}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    data = build_datasets(cfg)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)

    # Save codewords for reproducibility across analysis scripts
    cw_path = out_dir / "codewords.json"
    if not cw_path.exists():
        with open(cw_path, "w") as f:
            json.dump({"codewords": data["codewords"], "count": len(data["codewords"])}, f, indent=2)
        print(f"Saved codewords to {cw_path}")

    # Save config for cross-seed verification
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        with open(cfg_path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        print(f"Saved config to {cfg_path}")

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

    # Resume from checkpoint (fresh optimizer, LR reset)
    start_step = 0  # 0 means start from step 1
    if args.continue_from:
        ckpt = torch.load(args.continue_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt["step"]
        print(f"  Continuing from {args.continue_from} (step {start_step}), LR schedule preserved")
        del ckpt
    elif args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed from {args.resume_from} (step {ckpt['step']})")
        del ckpt

    # Separate weight decay groups
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
    lambda_phase2 = args.lambda_probe2 if args.lambda_probe2 is not None else lambda_base
    lambda_step = args.lambda_step
    ce_none = nn.CrossEntropyLoss(reduction='none')

    metrics = []
    # If continuing, load existing metrics up to start_step
    if start_step > 0:
        metrics_path = out_dir / "pilot_metrics.json"
        if metrics_path.exists():
            all_prev = json.load(open(metrics_path))
            metrics = [m for m in all_prev if m["step"] <= start_step]
            print(f"  Loaded {len(metrics)} metric entries up to step {start_step}")
    data_iter = iter(train_loader)
    train_loss_accum = 0.0
    n_accum = 0
    t0 = time.time()
    best_ood = max((m["probe_ood_acc"] for m in metrics), default=0.0)

    print(f"\n{'step':>7s}  {'train':>7s}  {'val':>7s}  "
          f"{'p_in':>6s}  {'p_ood':>6s}  {'nll_in':>7s}  {'nll_ood':>7s}  "
          f"{'lm_ood':>7s}  {'lam':>4s}  {'lr':>9s}  {'min':>5s}")
    print("-" * 90)

    for step in range(1, cfg.total_steps + 1):
        if step <= start_step:
            continue  # skip already-trained steps (--continue-from)
        model.train()

        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Lambda scheduling
        cur_lambda = lambda_phase2 if step >= lambda_step else lambda_base

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
                loss_flat = ce_none(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_all = loss_flat.view(targets.shape)
                pmask = probe_mask.bool()
                lm_mask = ~pmask & (targets != -100)
                lm_loss = loss_all[lm_mask].mean() if lm_mask.any() else torch.tensor(0.0, device=device)
                p_loss = loss_all[pmask].mean() if pmask.any() else torch.tensor(0.0, device=device)
                loss = lm_loss + cur_lambda * p_loss
            else:
                _, loss = model(input_ids, targets)

            loss = loss / cfg.grad_accum_steps
            loss.backward()
            train_loss_accum += loss.item() * cfg.grad_accum_steps
            n_accum += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            avg_train = train_loss_accum / max(n_accum, 1)
            val_loss = evaluate_lm(model, val_loader, device)
            pin_acc = evaluate_probe(model, probe_in, device)
            pood_acc = evaluate_probe(model, probe_ood, device)
            nll_in, _, _ = evaluate_probe_nll(model, probe_in, device)
            nll_ood, lm_ood, r_ood = evaluate_probe_nll(model, probe_ood, device)
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
            }
            metrics.append(rec)

            # Save checkpoint
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

            if pood_acc > best_ood:
                best_ood = pood_acc

            print(f"{step:7d}  {avg_train:7.4f}  {val_loss:7.4f}  "
                  f"{pin_acc:6.3f}  {pood_acc:6.3f}  {nll_in:7.3f}  {nll_ood:7.3f}  "
                  f"{lm_ood:7.3f}  {cur_lambda:4.1f}  {lr:9.2e}  {elapsed:5.1f}")

            train_loss_accum = 0.0
            n_accum = 0

            # Early check: if OOD acc > 0.8 for 5 consecutive evals, declare grokking
            recent = [m["probe_ood_acc"] for m in metrics[-5:]]
            if len(recent) >= 5 and all(r >= 0.8 for r in recent):
                print(f"\n  >>> GROKKED at step {step}! (probe_ood >= 0.8 for 5 evals)")
                break

    # Save results
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PILOT COMPLETE")
    print(f"  Best OOD probe acc: {best_ood:.4f}")
    print(f"  Steps: {metrics[-1]['step']}")
    print(f"  Final val loss: {metrics[-1]['val_loss']:.4f}")
    print(f"  Saved to {out_dir}/pilot_metrics.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
