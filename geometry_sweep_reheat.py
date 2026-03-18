#!/usr/bin/env python3
"""
Geometry Sweep Reheating: extended grid with λ support.

Runs reheating for β2={0.95, 0.80}, checkpoints {1000, 4000, 10000},
LR × λ grid. Reuses existing completed runs via skip-done logic.

Usage:
    python geometry_sweep_reheat.py
    python geometry_sweep_reheat.py --beta2s 0.95,0.80 --ckpts 1000,4000,10000 \
        --lrs 0.001,0.0006,0.0003 --lams 2.0,4.0
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training'))

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe, evaluate_probe_nll, evaluate_lm


def get_lr(step, total_steps, warmup, base_lr):
    """LR schedule: linear warmup then cosine decay with floor=0.1."""
    if step < warmup:
        return base_lr * step / warmup
    decay_ratio = (step - warmup) / max(1, total_steps - warmup)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * max(coeff, 0.1)


def run_one_reheat(model, base_sd, train_loader, val_loader, probe_in, probe_ood,
                   device, lr, beta2, warmup, steps, lam, eval_every, out_dir,
                   save_ckpts=True):
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
        cur_lr = get_lr(step, steps, warmup, lr)
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

        # Early stop: val_loss > 10 at step 500
        if step == 500:
            val_loss = evaluate_lm(model, val_loader, device)
            if val_loss > 10:
                print(f"    EARLY STOP: val_loss={val_loss:.2f} > 10 at step 500")
                return metrics

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

            # Save checkpoint (optional — skip for speed)
            if save_ckpts:
                ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
                torch.save({"step": step, "model_state_dict": model.state_dict()}, ckpt_path)

            train_loss_accum = 0.0
            n_accum = 0

            # Early stop: p_ood < 0.05 after 400 steps for λ=4, LR=1e-3
            if step >= 400 and lam >= 4.0 and pood_acc < 0.05:
                print(f"    EARLY STOP: p_ood={pood_acc:.4f} < 0.05 at step {step} (λ={lam}, lr={lr})")
                break

    # Save metrics
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def find_existing_reheat(run_dir, ckpt_step, lr, lam, steps):
    """Check for existing reheat run that matches this config.

    Looks in both the standard output dir and old-style dirs.
    """
    # Standard output pattern for this script
    out_dir = run_dir / f"reheat_ckpt{ckpt_step}_lr{lr}_lam{lam}"
    if (out_dir / "pilot_metrics.json").exists():
        met = json.load(open(out_dir / "pilot_metrics.json"))
        if met and met[-1]["step"] >= steps:
            return out_dir, met

    # Old-style pattern (from beta2_reheating.py, λ=4.0 only)
    if lam == 4.0:
        old_dir = run_dir / f"reheat_ckpt{ckpt_step}_lr{lr}"
        if (old_dir / "pilot_metrics.json").exists():
            met = json.load(open(old_dir / "pilot_metrics.json"))
            if met and met[-1]["step"] >= steps:
                # Check lambda matches
                if met[0].get("cur_lambda") == lam:
                    return old_dir, met

    return None, None


def extract_reheat_summary(metrics):
    """Extract summary metrics from a reheat run."""
    if not metrics:
        return {}

    p0 = metrics[0]["probe_ood_acc"]
    pood_vals = [m["probe_ood_acc"] for m in metrics]
    peak_pood = max(pood_vals)
    step_peak = metrics[pood_vals.index(peak_pood)]["step"]
    final_pood = metrics[-1]["probe_ood_acc"]
    G = peak_pood - p0

    # D = p_after_warmup - p0 (find first eval after warmup=200)
    p_after_warmup = p0
    for m in metrics:
        if m["step"] >= 200:
            p_after_warmup = m["probe_ood_acc"]
            break
    D = p_after_warmup - p0

    return {
        "p0": p0,
        "peak_p_ood": peak_pood,
        "step_peak": step_peak,
        "final_p_ood": final_pood,
        "G": G,
        "D": D,
    }


def main():
    parser = argparse.ArgumentParser(description="Geometry Sweep Reheating")
    parser.add_argument("--base-dir", type=str, default="runs/beta2_ablation",
                        help="Base directory with β2 runs")
    parser.add_argument("--beta2s", type=str, default="0.95,0.80",
                        help="Comma-separated β2 values")
    parser.add_argument("--ckpts", type=str, default="1000,4000,10000",
                        help="Comma-separated checkpoint steps")
    parser.add_argument("--lrs", type=str, default="0.001,0.0006,0.0003",
                        help="Comma-separated learning rates")
    parser.add_argument("--lams", type=str, default="2.0,4.0",
                        help="Comma-separated lambda_probe values")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Reheating duration (steps)")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=200,
                        help="Evaluation interval")
    parser.add_argument("--no-save-ckpts", action="store_true",
                        help="Skip saving checkpoints (faster, only save metrics)")
    parser.add_argument("--out-csv", type=str, default="runs/reheat_sweep/reheat_summary.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    device = get_device()
    beta2_list = [float(x) for x in args.beta2s.split(",")]
    ckpt_list = [int(x) for x in args.ckpts.split(",")]
    lr_list = [float(x) for x in args.lrs.split(",")]
    lam_list = [float(x) for x in args.lams.split(",")]

    total_runs = len(beta2_list) * len(ckpt_list) * len(lr_list) * len(lam_list)
    print(f"{'='*60}")
    print(f"  Geometry Sweep Reheating")
    print(f"  Device: {device}")
    print(f"  β2: {beta2_list}")
    print(f"  Checkpoints: {ckpt_list}")
    print(f"  LRs: {lr_list}")
    print(f"  Lambdas: {lam_list}")
    print(f"  {total_runs} total conditions, {args.steps} steps each")
    print(f"{'='*60}")

    # Load dataset ONCE
    print("\n  Loading dataset (one-time cost)...")
    cfg = Config(seed=42, p_probe=0.20, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)

    # Find codewords
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
    all_summaries = []

    for b2 in beta2_list:
        b2_str = f"{b2:.2f}" if b2 < 1.0 else str(b2)
        run_dir = Path(args.base_dir) / f"pilot_wd0.5_lr0.001_lp2.0_b2{b2_str}_s42"
        if not run_dir.exists():
            print(f"\n  [SKIP] β2={b2}: run directory not found")
            continue

        for ckpt_step in ckpt_list:
            ckpt_path = run_dir / f"ckpt_{ckpt_step:06d}.pt"
            if not ckpt_path.exists():
                print(f"\n  [SKIP] β2={b2}, ckpt={ckpt_step}: checkpoint not found")
                # Skip all LR/lam combos for this checkpoint
                run_idx += len(lr_list) * len(lam_list)
                continue

            # Load checkpoint ONCE per (β2, ckpt) pair
            print(f"\n  Loading checkpoint β2={b2}, step {ckpt_step}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            base_sd = ckpt["model_state_dict"]
            del ckpt

            for lr in lr_list:
                for lam in lam_list:
                    run_idx += 1

                    # Check for existing run
                    existing_dir, existing_met = find_existing_reheat(
                        run_dir, ckpt_step, lr, lam, args.steps)

                    if existing_met:
                        summary = extract_reheat_summary(existing_met)
                        summary.update({
                            "beta2": b2, "ckpt_step": ckpt_step,
                            "lr": lr, "lam": lam, "source": str(existing_dir),
                        })
                        all_summaries.append(summary)
                        print(f"\n  [{run_idx}/{total_runs}] β2={b2} ckpt={ckpt_step} "
                              f"lr={lr} λ={lam}: SKIP (done, G={summary['G']:.4f})")
                        continue

                    out_dir = run_dir / f"reheat_ckpt{ckpt_step}_lr{lr}_lam{lam}"
                    print(f"\n  [{run_idx}/{total_runs}] β2={b2} ckpt={ckpt_step} lr={lr} λ={lam}")
                    t0 = time.time()

                    metrics = run_one_reheat(
                        model, base_sd, train_loader, val_loader,
                        probe_in, probe_ood, device,
                        lr=lr, beta2=b2, warmup=args.warmup,
                        steps=args.steps, lam=lam, eval_every=args.eval_every,
                        out_dir=out_dir, save_ckpts=not args.no_save_ckpts)

                    elapsed = time.time() - t0

                    summary = extract_reheat_summary(metrics)
                    summary.update({
                        "beta2": b2, "ckpt_step": ckpt_step,
                        "lr": lr, "lam": lam, "source": str(out_dir),
                    })
                    all_summaries.append(summary)

                    if metrics:
                        best_pood = max(m["probe_ood_acc"] for m in metrics)
                        print(f"    Done in {elapsed/60:.1f}min: G={summary.get('G', 0):.4f}, "
                              f"best_pood={best_pood:.4f}")

            del base_sd
            if device == "mps":
                torch.mps.empty_cache()

    # Save summary CSV
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_summaries:
        fieldnames = ["beta2", "ckpt_step", "lr", "lam", "p0", "peak_p_ood",
                       "step_peak", "final_p_ood", "G", "D", "source"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({k: s.get(k, "") for k in fieldnames})
        print(f"\n  Saved summary: {csv_path}")

    total_elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  All reheating complete in {total_elapsed:.1f} min")
    print(f"  {len(all_summaries)} conditions processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
