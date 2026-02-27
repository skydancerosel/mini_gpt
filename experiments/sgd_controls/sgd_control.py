import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
SGD control experiment for backbone oscillation study.

Three runs (all 4000 steps, seed 42, matched schedule/data/model):
  Run A: AdamW baseline    (lr=1e-3, betas=(0.9,0.95), wd=0.5 decoupled)
  Run B: SGD no momentum   (lr=1e-3, momentum=0.0, wd=0.5 L2-coupled)
  Run C: SGD + momentum    (lr=3e-3, momentum=0.9, wd=0.5 L2-coupled)

Everything else identical: 8L/d512/h16/ff2048, warmup 1500, cosine→10% floor,
grad_clip=1.0, lambda=2.0 (→4.0 at step 4000), eval_every=200, batch=64, accum=2.

Usage:
  python sgd_control.py                # all 3 runs
  python sgd_control.py --run A        # just AdamW baseline
  python sgd_control.py --run B        # just SGD no-momentum
  python sgd_control.py --run C        # just SGD + momentum
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
from pilot import evaluate_lm, evaluate_probe, evaluate_probe_nll


# ═══════════════════════════════════════════════════════════════════════════
# Constants (matched to main experiment: run_seed.sh seed=42)
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
TOTAL_STEPS = 4000
EVAL_EVERY = 200
WARMUP_STEPS = 1500
GRAD_CLIP = 1.0
BATCH_SIZE = 64
GRAD_ACCUM = 2
LAMBDA_PROBE = 2.0
LAMBDA_PROBE2 = 4.0
LAMBDA_STEP = 4000

N_LAYER = 8
D_MODEL = 512
N_HEAD = 16
D_FF = 2048


def _should_save_ckpt(step):
    """Checkpoint schedule for backbone analysis.

    Every 50 steps in [600, 2000] (backbone estimation window).
    Every 100 steps in (2000, 4000] (out-of-window evaluation).
    """
    if 600 <= step <= 2000 and step % 50 == 0:
        return True
    if 2000 < step <= 4000 and step % 100 == 0:
        return True
    return False


# Per-run optimizer configs (fixed, no probing)
RUN_CONFIGS = {
    "A": {"name": "adamw",     "opt": "adamw", "lr": 1e-3, "momentum": 0.0, "wd": 0.5},
    "B": {"name": "sgd_nomom", "opt": "sgd",   "lr": 1e-3, "momentum": 0.0, "wd": 0.5},
    "C": {"name": "sgd_mom",   "opt": "sgd",   "lr": 1e-2, "momentum": 0.9, "wd": 0.05},
}


def get_lr(step, base_lr):
    """Cosine decay with linear warmup, floor at 10% of base_lr."""
    if step < WARMUP_STEPS:
        return base_lr * step / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * max(coeff, 0.1)


def create_optimizer(model, opt_type, lr, wd, momentum=0.0):
    """Create optimizer with proper param groups (no wd on LN/bias)."""
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if opt_type == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    elif opt_type == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def train_run(run_label, rcfg, out_dir, model_factory, data, device):
    """Full training run to TOTAL_STEPS."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_type = rcfg["opt"]
    lr = rcfg["lr"]
    momentum = rcfg["momentum"]
    wd = rcfg["wd"]

    print(f"\n{'='*70}")
    print(f"  RUN {run_label}: {opt_type}  lr={lr:.0e}  momentum={momentum}  wd={wd}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = model_factory().to(device)
    print(f"  Model: {model.count_params():,} params")

    opt = create_optimizer(model, opt_type, lr, wd, momentum)

    train_loader = DataLoader(
        data["train_dataset"], batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        data["val_dataset"], batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, num_workers=0,
    )
    probe_in = data["probe_eval_in"]
    probe_ood = data["probe_eval_ood"]
    ce_none = nn.CrossEntropyLoss(reduction='none')

    metrics = []
    data_iter = iter(train_loader)
    train_loss_accum = 0.0
    n_accum = 0
    t0 = time.time()

    print(f"\n  {'step':>5s}  {'train':>7s}  {'val':>7s}  {'p_in':>5s}  "
          f"{'p_ood':>5s}  {'nll_ood':>7s}  {'lr':>9s}  {'min':>5s}")
    print(f"  {'-'*65}")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()

        cur_lr = get_lr(step, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        cur_lambda = LAMBDA_PROBE2 if step >= LAMBDA_STEP else LAMBDA_PROBE

        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            try:
                input_ids, targets, probe_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets, probe_mask = next(data_iter)

            input_ids, targets = input_ids.to(device), targets.to(device)
            probe_mask = probe_mask.to(device)

            logits, _ = model(input_ids)
            loss_flat = ce_none(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_all = loss_flat.view(targets.shape)
            pmask = probe_mask.bool()
            lm_mask = ~pmask & (targets != -100)
            lm_loss = (loss_all[lm_mask].mean()
                       if lm_mask.any()
                       else torch.tensor(0.0, device=device))
            p_loss = (loss_all[pmask].mean()
                      if pmask.any()
                      else torch.tensor(0.0, device=device))
            loss = (lm_loss + cur_lambda * p_loss) / GRAD_ACCUM
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        train_loss_accum += loss.item() * GRAD_ACCUM
        n_accum += 1

        # Save checkpoint on backbone schedule (every 50/100 steps)
        if _should_save_ckpt(step):
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            if not ckpt_path.exists():
                torch.save({"step": step,
                            "model_state_dict": model.state_dict()},
                           ckpt_path)

        # Evaluate and log at eval cadence
        if step % EVAL_EVERY == 0 or step == 1:
            avg_train = train_loss_accum / max(n_accum, 1)
            val_loss = evaluate_lm(model, val_loader, device)
            pin_acc = evaluate_probe(model, probe_in, device)
            pood_acc = evaluate_probe(model, probe_ood, device)
            nll_in, _, _ = evaluate_probe_nll(model, probe_in, device)
            nll_ood, lm_ood, r_ood = evaluate_probe_nll(model, probe_ood, device)
            elapsed = (time.time() - t0) / 60

            rec = {
                "step": step, "train_loss": avg_train, "val_loss": val_loss,
                "probe_in_acc": pin_acc, "probe_ood_acc": pood_acc,
                "nll_in": nll_in, "nll_ood": nll_ood,
                "lm_ood": lm_ood, "r_ood": r_ood,
                "cur_lambda": cur_lambda, "lr": cur_lr,
            }
            metrics.append(rec)

            # Also save checkpoint at eval steps (may overlap with above)
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            if not ckpt_path.exists():
                torch.save({"step": step,
                            "model_state_dict": model.state_dict()},
                           ckpt_path)

            print(f"  {step:5d}  {avg_train:7.4f}  {val_loss:7.4f}  "
                  f"{pin_acc:5.3f}  {pood_acc:5.3f}  {nll_ood:7.3f}  "
                  f"{cur_lr:9.2e}  {elapsed:5.1f}")

            train_loss_accum = 0.0
            n_accum = 0

    # Save metrics
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save run config
    run_config = {
        "optimizer": opt_type, "lr": lr, "momentum": momentum,
        "weight_decay": wd, "warmup_steps": WARMUP_STEPS,
        "total_steps": TOTAL_STEPS, "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM,
        "lambda_probe": LAMBDA_PROBE, "lambda_probe2": LAMBDA_PROBE2,
        "lambda_step": LAMBDA_STEP, "seed": SEED,
        "n_layer": N_LAYER, "d_model": D_MODEL,
        "n_head": N_HEAD, "d_ff": D_FF,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n  Run {run_label} done. "
          f"Final p_ood={metrics[-1]['probe_ood_acc']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SGD control experiment")
    parser.add_argument("--run", choices=["A", "B", "C"], default=None,
                        help="Run a specific experiment (default: all three)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Build shared datasets
    cfg = Config(
        seed=SEED, p_probe=0.10, batch_size=BATCH_SIZE,
        n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD, d_ff=D_FF,
    )
    ref_cw = Path("runs/pilot_wd0.5_lr0.001_lp2.0_s42/codewords.json")
    cw_path = str(ref_cw) if ref_cw.exists() else None
    print("Building datasets...")
    data = build_datasets(cfg, codewords_path=cw_path)
    vocab_size = len(data["tokenizer"])

    def model_factory():
        return GPTModel(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            d_model=D_MODEL, n_layer=N_LAYER,
            n_head=N_HEAD, d_ff=D_FF, dropout=0.0,
        )

    base_dir = Path("runs/sgd_control")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save codewords
    cw_out = base_dir / "codewords.json"
    if not cw_out.exists():
        with open(cw_out, "w") as f:
            json.dump({"codewords": data["codewords"],
                       "count": len(data["codewords"])}, f, indent=2)

    # Run training
    todo = ["A", "B", "C"] if args.run is None else [args.run]

    for label in todo:
        rcfg = RUN_CONFIGS[label]
        out = base_dir / f"{rcfg['name']}_s{SEED}"

        # Skip if already complete
        mp = out / "pilot_metrics.json"
        if mp.exists():
            existing = json.load(open(mp))
            if existing and existing[-1]["step"] >= TOTAL_STEPS:
                print(f"\n  Run {label} ({rcfg['name']}) already complete, "
                      f"skipping.")
                continue

        train_run(label, rcfg, out, model_factory, data, device)

    print(f"\n{'='*70}")
    print(f"  SGD CONTROL EXPERIMENT COMPLETE")
    print(f"  Results: {base_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
