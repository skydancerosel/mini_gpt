#!/usr/bin/env python3
"""
Cyclic reheating training: periodic hot windows with higher λ and constant LR.

Loads dataset ONCE, runs all β2 conditions in one process.
Compares against existing baseline runs (no re-training needed).

Usage:
    python cyclic_train.py --beta2s "0.99,0.95,0.90,0.80" --steps 4000

Default config (from cyclic_reheat.md):
    K=800 (cycle period), H=200 (hot window), lr_hot=6e-4, λ_hot=4
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe, evaluate_probe_nll, evaluate_lm


def get_lr(step, total_steps, warmup, base_lr):
    """Baseline LR schedule: linear warmup + cosine decay to 10% floor."""
    if step < warmup:
        return base_lr * step / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * max(coeff, 0.1)


def run_one_cyclic(model, train_loader, val_loader, probe_in, probe_ood,
                   device, base_lr, beta2, warmup, steps, eval_every,
                   lambda_base, cycle_K, cycle_H, lr_hot, lambda_hot,
                   out_dir, seed=42):
    """Run a single cyclic training from scratch. Returns metrics list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-init model weights
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model._init_weights()

    # Fresh optimizer
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
        {"params": decay_params, "weight_decay": 0.5},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=base_lr, betas=(0.9, beta2), eps=1e-8)

    ce_none = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(train_loader)
    metrics = []
    t0 = time.time()
    train_loss_accum = 0.0
    n_accum = 0

    print(f"    {'step':>5s}  {'train':>7s}  {'val':>7s}  "
          f"{'p_in':>6s}  {'p_ood':>6s}  {'lam':>4s}  {'lr':>9s}  {'hot':>3s}")
    print(f"    {'-'*65}")

    for step in range(1, steps + 1):
        model.train()

        # Cyclic schedule
        is_hot = (step - 1) % cycle_K < cycle_H  # step-1 so step=1 is hot

        if is_hot:
            lr = lr_hot
            cur_lambda = lambda_hot
        else:
            lr = get_lr(step, steps, warmup, base_lr)
            cur_lambda = lambda_base

        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        # Gradient accumulation (2 micro-batches like baseline)
        for _ in range(2):
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
            loss = (lm_loss + cur_lambda * p_loss) / 2
            loss.backward()
            train_loss_accum += loss.item() * 2
            n_accum += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
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
                "is_hot": is_hot,
            }
            metrics.append(rec)

            hot_str = "HOT" if is_hot else ""
            print(f"    {step:5d}  {avg_train:7.4f}  {val_loss:7.4f}  "
                  f"{pin_acc:6.3f}  {pood_acc:6.3f}  {cur_lambda:4.1f}  {lr:9.6f}  {hot_str}")

            # Save checkpoint
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

            train_loss_accum = 0.0
            n_accum = 0

    # Save metrics
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cyclic reheating training")
    parser.add_argument("--base-dir", type=str, default="runs/beta2_ablation",
                        help="Base directory with β2 runs")
    parser.add_argument("--beta2s", type=str, default="0.99,0.95,0.90,0.80",
                        help="Comma-separated β2 values")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Baseline peak learning rate")
    parser.add_argument("--lambda-base", type=float, default=2.0,
                        help="Lambda during cold phase")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Cyclic schedule
    parser.add_argument("--cycle-K", type=int, default=800,
                        help="Cycle period in steps")
    parser.add_argument("--cycle-H", type=int, default=200,
                        help="Hot window length in steps")
    parser.add_argument("--lr-hot", type=float, default=0.0006,
                        help="Constant LR during hot window")
    parser.add_argument("--lambda-hot", type=float, default=4.0,
                        help="Lambda during hot window")

    parser.add_argument("--skip-done", action="store_true", default=True)
    args = parser.parse_args()

    device = get_device()
    beta2_list = [float(x) for x in args.beta2s.split(",")]

    print(f"{'='*60}")
    print(f"  Cyclic Reheating Training")
    print(f"  Device: {device}")
    print(f"  β2: {beta2_list}")
    print(f"  Steps: {args.steps}, Warmup: {args.warmup}")
    print(f"  Cycle: K={args.cycle_K}, H={args.cycle_H}")
    print(f"  LR_hot={args.lr_hot}, λ_hot={args.lambda_hot}")
    print(f"  Baseline: LR={args.lr}, λ={args.lambda_base}")
    print(f"{'='*60}")

    # Load dataset ONCE
    print("\n  Loading dataset (one-time cost)...")
    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)

    # Find codewords from existing runs
    cw_path = None
    for b2 in beta2_list:
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        p = Path(args.base_dir) / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42" / "codewords.json"
        if p.exists():
            cw_path = str(p)
            break

    data = build_datasets(cfg, codewords_path=cw_path)
    vocab_size = len(data["tokenizer"])

    train_loader = DataLoader(
        data["train_dataset"], batch_size=64,
        shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        data["val_dataset"], batch_size=64,
        shuffle=False, drop_last=False, num_workers=0)
    probe_in = data["probe_eval_in"]
    probe_ood = data["probe_eval_ood"]

    # Create model ONCE
    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    run_idx = 0
    t_total = time.time()

    for b2 in beta2_list:
        run_idx += 1
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        run_dir = Path(args.base_dir) / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
        out_dir = run_dir / f"cyclic_K{args.cycle_K}_H{args.cycle_H}"

        # Skip if done
        if args.skip_done and (out_dir / "pilot_metrics.json").exists():
            met = json.load(open(out_dir / "pilot_metrics.json"))
            if met and met[-1]["step"] >= args.steps:
                best = max(m["probe_ood_acc"] for m in met)
                print(f"\n  [{run_idx}/{len(beta2_list)}] β2={b2}: "
                      f"SKIP (done, best_pood={best:.4f})")
                continue

        print(f"\n  [{run_idx}/{len(beta2_list)}] β2={b2} — cyclic training from scratch")
        t0 = time.time()

        metrics = run_one_cyclic(
            model, train_loader, val_loader, probe_in, probe_ood,
            device, base_lr=args.lr, beta2=b2, warmup=args.warmup,
            steps=args.steps, eval_every=args.eval_every,
            lambda_base=args.lambda_base,
            cycle_K=args.cycle_K, cycle_H=args.cycle_H,
            lr_hot=args.lr_hot, lambda_hot=args.lambda_hot,
            out_dir=out_dir, seed=args.seed)

        elapsed = time.time() - t0
        best_pood = max(m["probe_ood_acc"] for m in metrics)
        final_pood = metrics[-1]["probe_ood_acc"]
        print(f"    Done in {elapsed/60:.1f}min: best_pood={best_pood:.4f}, "
              f"final_pood={final_pood:.4f}")

        if device == "mps":
            torch.mps.empty_cache()

    total_elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  All cyclic training complete in {total_elapsed:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
