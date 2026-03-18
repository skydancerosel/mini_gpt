#!/usr/bin/env python3
"""
Fast β2 reheating: loads dataset ONCE, runs all conditions in one process.

Usage:
    python beta2_reheating.py --base-dir runs/beta2_ablation/

Runs reheating for β2=0.95 and β2=0.80, checkpoints 2000 and 4000,
LR grid {1e-3, 6e-4, 3e-4}. Skips already-completed runs.
"""

import argparse
import json
import math
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


def get_lr(step, total_steps, warmup, base_lr, constant=False):
    """LR schedule: linear warmup then either cosine decay or constant."""
    if step < warmup:
        return base_lr * step / warmup
    if constant:
        return base_lr
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * max(coeff, 0.1)


def run_one_reheat(model, base_sd, train_loader, val_loader, probe_in, probe_ood,
                   device, lr, beta2, warmup, steps, lam, eval_every, out_dir,
                   constant_lr=False):
    """Run a single reheating condition. Returns metrics list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fresh model from checkpoint
    model.load_state_dict(base_sd)

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
    ], lr=lr, betas=(0.9, beta2), eps=1e-8)

    ce_none = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(train_loader)
    metrics = []
    t0 = time.time()
    train_loss_accum = 0.0
    n_accum = 0

    for step in range(1, steps + 1):
        model.train()
        cur_lr = get_lr(step, steps, warmup, lr, constant=constant_lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        train_loss_accum += loss.item()
        n_accum += 1

        if step % eval_every == 0 or step == 1:
            avg_train = train_loss_accum / max(n_accum, 1)
            val_loss = evaluate_lm(model, val_loader, device)
            pin_acc = evaluate_probe(model, probe_in, device)
            pood_acc = evaluate_probe(model, probe_ood, device)
            nll_ood, lm_ood, r_ood = evaluate_probe_nll(model, probe_ood, device)
            elapsed = (time.time() - t0) / 60

            rec = {
                "step": step,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "probe_in_acc": pin_acc,
                "probe_ood_acc": pood_acc,
                "nll_ood": nll_ood,
                "lm_ood": lm_ood,
                "r_ood": r_ood,
                "cur_lambda": lam,
                "lr": cur_lr,
            }
            metrics.append(rec)

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
    parser = argparse.ArgumentParser(description="Fast β2 reheating")
    parser.add_argument("--base-dir", type=str, default="runs/beta2_ablation",
                        help="Base directory with β2 runs")
    parser.add_argument("--beta2s", type=str, default="0.95,0.80",
                        help="Comma-separated β2 values to reheat")
    parser.add_argument("--ckpts", type=str, default="2000,4000",
                        help="Comma-separated checkpoint steps")
    parser.add_argument("--lrs", type=str, default="0.001,0.0006,0.0003",
                        help="Comma-separated learning rates")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--skip-done", action="store_true", default=True)
    parser.add_argument("--constant-lr", action="store_true", default=False,
                        help="Use constant LR after warmup instead of cosine decay")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to reheat output dir name")
    args = parser.parse_args()

    device = get_device()
    beta2_list = [float(x) for x in args.beta2s.split(",")]
    ckpt_list = [int(x) for x in args.ckpts.split(",")]
    lr_list = [float(x) for x in args.lrs.split(",")]

    total_runs = len(beta2_list) * len(ckpt_list) * len(lr_list)
    print(f"{'='*60}")
    print(f"  Fast β2 Reheating")
    print(f"  Device: {device}")
    print(f"  β2: {beta2_list}, ckpts: {ckpt_list}, LRs: {lr_list}")
    print(f"  {total_runs} total conditions, {args.steps} steps each")
    print(f"{'='*60}")

    # Load dataset ONCE
    print("\n  Loading dataset (one-time cost)...")
    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)

    # Try to find codewords from any run
    cw_path = None
    for b2 in beta2_list:
        p = Path(args.base_dir) / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2}_s42" / "codewords.json"
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
        # Format β2 to match directory naming (e.g. 0.8 → "0.80")
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        run_dir = Path(args.base_dir) / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
        if not run_dir.exists():
            print(f"\n  [SKIP] β2={b2}: run directory not found")
            continue

        for ckpt_step in ckpt_list:
            ckpt_path = run_dir / f"ckpt_{ckpt_step:06d}.pt"
            if not ckpt_path.exists():
                print(f"\n  [SKIP] β2={b2}, ckpt={ckpt_step}: checkpoint not found")
                continue

            # Load checkpoint ONCE per (β2, ckpt) pair
            print(f"\n  Loading checkpoint β2={b2}, step {ckpt_step}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            base_sd = ckpt["model_state_dict"]
            del ckpt

            for lr in lr_list:
                run_idx += 1
                suffix = args.out_suffix
                out_dir = run_dir / f"reheat_ckpt{ckpt_step}_lr{lr}{suffix}"

                # Skip if done
                if args.skip_done and (out_dir / "pilot_metrics.json").exists():
                    met = json.load(open(out_dir / "pilot_metrics.json"))
                    if met and met[-1]["step"] >= args.steps:
                        print(f"\n  [{run_idx}/{total_runs}] β2={b2} ckpt={ckpt_step} lr={lr}: "
                              f"SKIP (already done, best_pood={max(m['probe_ood_acc'] for m in met):.4f})")
                        continue

                print(f"\n  [{run_idx}/{total_runs}] β2={b2} ckpt={ckpt_step} lr={lr}")
                t0 = time.time()

                metrics = run_one_reheat(
                    model, base_sd, train_loader, val_loader, probe_in, probe_ood,
                    device, lr=lr, beta2=b2, warmup=args.warmup,
                    steps=args.steps, lam=args.lam, eval_every=args.eval_every,
                    out_dir=out_dir, constant_lr=args.constant_lr)

                elapsed = time.time() - t0
                best_pood = max(m["probe_ood_acc"] for m in metrics)
                final_pood = metrics[-1]["probe_ood_acc"]
                print(f"    Done in {elapsed/60:.1f}min: best_pood={best_pood:.4f}, "
                      f"final_pood={final_pood:.4f}")

            del base_sd
            if device == "mps":
                torch.mps.empty_cache()

    total_elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  All reheating complete in {total_elapsed:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
