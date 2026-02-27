import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Run C': SGD + Nesterov momentum + decoupled weight decay (SGDW).

Key differences from Run C:
  - nesterov=True
  - Decoupled WD (SGDW): weight_decay=0 in optimizer, manually apply
    theta <- (1 - eta * wd) * theta  each step for decay params
  - wd=0.5 (same as AdamW, not L2-coupled)
  - peak_lr=1e-2
  - Runs to 2000 steps (early stop if val>5.1 and p_ood<0.02 at step 2000)

SGDW implementation:
  Standard SGD step: theta <- theta - eta * (grad + wd * theta)   [L2 coupled]
  SGDW step:         theta <- (1 - eta * wd) * theta - eta * grad [decoupled]

We achieve SGDW by:
  1. Setting weight_decay=0 in torch.optim.SGD (no L2 term in gradient)
  2. After opt.step(), manually: theta *= (1 - cur_lr * wd) for decay params

Usage:
  python sgd_nesterov_run.py
"""

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
# Constants (matched to sgd_control.py)
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
TOTAL_STEPS = 2000        # short run, early-stop decision at 2000
EVAL_EVERY = 200
WARMUP_STEPS = 1500
GRAD_CLIP = 1.0
BATCH_SIZE = 64
GRAD_ACCUM = 2
LAMBDA_PROBE = 2.0
LAMBDA_PROBE2 = 4.0
LAMBDA_STEP = 4000         # never triggers (run ends at 2000)

N_LAYER = 8
D_MODEL = 512
N_HEAD = 16
D_FF = 2048

# Run C' config
PEAK_LR = 1e-2
MOMENTUM = 0.9
NESTEROV = True
WD = 0.5                   # decoupled, same as AdamW


def _should_save_ckpt(step):
    """Same checkpoint schedule as sgd_control.py."""
    if 600 <= step <= 2000 and step % 50 == 0:
        return True
    if 2000 < step <= 4000 and step % 100 == 0:
        return True
    return False


def get_lr(step, base_lr):
    """Cosine decay with linear warmup, floor at 10% of base_lr.

    NOTE: Uses TOTAL_STEPS=2000, so cosine schedule is faster than
    the original 4000-step runs. The warmup is 1500 steps so most
    of this run is in warmup.
    """
    if step < WARMUP_STEPS:
        return base_lr * step / WARMUP_STEPS
    # Use 4000 for the cosine denominator to keep schedule comparable
    decay_ratio = (step - WARMUP_STEPS) / max(1, 4000 - WARMUP_STEPS)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * max(coeff, 0.1)


def create_sgdw_optimizer(model):
    """Create SGD optimizer with nesterov momentum, wd=0 (decoupled WD done manually).

    Returns:
        optimizer: SGD with wd=0 for all groups
        decay_params: list of params that should get decoupled WD
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Both groups have weight_decay=0 — WD applied manually
    groups = [
        {"params": decay_params, "weight_decay": 0.0, "sgdw_decay": True},
        {"params": no_decay_params, "weight_decay": 0.0, "sgdw_decay": False},
    ]
    optimizer = torch.optim.SGD(
        groups, lr=PEAK_LR, momentum=MOMENTUM, nesterov=NESTEROV)

    return optimizer, decay_params


def apply_sgdw_decay(decay_params, lr, wd):
    """Apply decoupled weight decay: theta *= (1 - lr * wd).

    This is the SGDW step, applied after the SGD gradient update.
    """
    factor = 1.0 - lr * wd
    with torch.no_grad():
        for p in decay_params:
            p.mul_(factor)


def main():
    device = get_device()
    print(f"Device: {device}")

    base_dir = Path("runs/sgd_control")
    out_dir = base_dir / "sgdw_nesterov_s42"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RUN C': SGD + Nesterov + Decoupled WD (SGDW)")
    print(f"  lr={PEAK_LR:.0e}  momentum={MOMENTUM}  nesterov={NESTEROV}  wd={WD}")
    print(f"  Total steps: {TOTAL_STEPS}  (early-stop decision)")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Build shared datasets (same seed, same data as other runs)
    cfg = Config(
        seed=SEED, p_probe=0.10, batch_size=BATCH_SIZE,
        n_layer=N_LAYER, d_model=D_MODEL, n_head=N_HEAD, d_ff=D_FF,
    )
    ref_cw = Path("runs/pilot_wd0.5_lr0.001_lp2.0_s42/codewords.json")
    cw_path = str(ref_cw) if ref_cw.exists() else None
    print("Building datasets...")
    data = build_datasets(cfg, codewords_path=cw_path)
    vocab_size = len(data["tokenizer"])

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=D_MODEL, n_layer=N_LAYER,
        n_head=N_HEAD, d_ff=D_FF, dropout=0.0,
    ).to(device)
    print(f"  Model: {model.count_params():,} params")

    opt, decay_params = create_sgdw_optimizer(model)
    print(f"  Decay params: {len(decay_params)} tensors")
    print(f"  SGDW factor at peak_lr: (1 - {PEAK_LR} * {WD}) = "
          f"{1 - PEAK_LR * WD:.4f}")

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

    # Save initial checkpoint
    torch.save({"step": 1, "model_state_dict": model.state_dict()},
               out_dir / "ckpt_000001.pt")

    print(f"\n  {'step':>5s}  {'train':>7s}  {'val':>7s}  {'p_in':>5s}  "
          f"{'p_ood':>5s}  {'nll_ood':>7s}  {'lr':>9s}  {'wd_eff':>9s}  {'min':>5s}")
    print(f"  {'-'*75}")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()

        cur_lr = get_lr(step, PEAK_LR)
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

        # SGD step (gradient update only, no L2 penalty)
        opt.step()

        # SGDW: manually apply decoupled weight decay
        apply_sgdw_decay(decay_params, cur_lr, WD)

        train_loss_accum += loss.item() * GRAD_ACCUM
        n_accum += 1

        # Save checkpoint on backbone schedule
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
            nll_ood, lm_ood, r_ood = evaluate_probe_nll(
                model, probe_ood, device)
            elapsed = (time.time() - t0) / 60

            wd_eff = cur_lr * WD  # effective WD this step

            rec = {
                "step": step, "train_loss": avg_train, "val_loss": val_loss,
                "probe_in_acc": pin_acc, "probe_ood_acc": pood_acc,
                "nll_in": nll_in, "nll_ood": nll_ood,
                "lm_ood": lm_ood, "r_ood": r_ood,
                "cur_lambda": cur_lambda, "lr": cur_lr,
            }
            metrics.append(rec)

            # Also save checkpoint at eval steps
            ckpt_path = out_dir / f"ckpt_{step:06d}.pt"
            if not ckpt_path.exists():
                torch.save({"step": step,
                            "model_state_dict": model.state_dict()},
                           ckpt_path)

            print(f"  {step:5d}  {avg_train:7.4f}  {val_loss:7.4f}  "
                  f"{pin_acc:5.3f}  {pood_acc:5.3f}  {nll_ood:7.3f}  "
                  f"{cur_lr:9.2e}  {wd_eff:9.2e}  {elapsed:5.1f}")

            train_loss_accum = 0.0
            n_accum = 0

    # Save metrics
    with open(out_dir / "pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save run config
    run_config = {
        "optimizer": "sgdw_nesterov",
        "lr": PEAK_LR, "momentum": MOMENTUM, "nesterov": NESTEROV,
        "weight_decay": WD, "weight_decay_type": "decoupled (SGDW)",
        "warmup_steps": WARMUP_STEPS,
        "total_steps": TOTAL_STEPS, "grad_clip": GRAD_CLIP,
        "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM,
        "lambda_probe": LAMBDA_PROBE, "lambda_probe2": LAMBDA_PROBE2,
        "lambda_step": LAMBDA_STEP, "seed": SEED,
        "n_layer": N_LAYER, "d_model": D_MODEL,
        "n_head": N_HEAD, "d_ff": D_FF,
        "note": "SGDW = SGD + decoupled WD. theta *= (1 - lr*wd) after grad step.",
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Copy codewords
    cw_src = base_dir / "codewords.json"
    cw_dst = out_dir / "codewords.json"
    if cw_src.exists() and not cw_dst.exists():
        import shutil
        shutil.copy2(cw_src, cw_dst)

    # Decision: should we continue?
    final_val = metrics[-1]["val_loss"]
    final_pood = metrics[-1]["probe_ood_acc"]
    best_pood = max(r["probe_ood_acc"] for r in metrics)

    print(f"\n{'='*70}")
    print(f"  RUN C' COMPLETE (2000 steps)")
    print(f"  Final val_loss: {final_val:.4f}")
    print(f"  Final p_ood:    {final_pood:.4f}")
    print(f"  Best p_ood:     {best_pood:.4f}")
    print(f"{'='*70}")

    # Compare to old Run C
    old_c_metrics_path = base_dir / "sgd_mom_s42" / "pilot_metrics.json"
    if old_c_metrics_path.exists():
        old_m = json.load(open(old_c_metrics_path))
        # Find old C at step 2000
        old_at_2000 = [r for r in old_m if r["step"] == 2000]
        if old_at_2000:
            old_val = old_at_2000[0]["val_loss"]
            old_pood = old_at_2000[0]["probe_ood_acc"]
            print(f"\n  Old Run C at step 2000:")
            print(f"    val_loss: {old_val:.4f}  p_ood: {old_pood:.4f}")
            print(f"  New Run C' at step 2000:")
            print(f"    val_loss: {final_val:.4f}  p_ood: {final_pood:.4f}")
            if final_val < old_val:
                print(f"  => C' is BETTER by {old_val - final_val:.4f} val_loss")
            else:
                print(f"  => C' is WORSE by {final_val - old_val:.4f} val_loss")

    # Early stop decision
    print(f"\n  EARLY-STOP DECISION:")
    if final_val > 5.1 and best_pood < 0.02:
        print(f"  => STOP. val_loss ({final_val:.2f}) > 5.1 and "
              f"p_ood ({best_pood:.3f}) < 0.02.")
        print(f"  => Nesterov + SGDW did not help enough.")
    elif final_val > 5.1:
        print(f"  => MARGINAL. val_loss ({final_val:.2f}) > 5.1 but "
              f"p_ood ({best_pood:.3f}) shows some signal.")
        print(f"  => Consider extending to 4000 steps.")
    else:
        print(f"  => CONTINUE! val_loss ({final_val:.2f}) < 5.1 — "
              f"clearly better than old C.")
        print(f"  => Extend to 4000 steps for full comparison.")

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
