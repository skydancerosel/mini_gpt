"""OLMoE per-head attention mech-interp classification — Pythia 410M port.

Mirrors analyses/pythia_410m_mechinterp.py exactly:
  1. Load model with eager attention (so output_attentions works)
  2. Run forward over the 2000-example induction batch
  3. For each (L, H), collect attention weight from the last (second-A)
     position to all positions
  4. Mean attention to canonical target positions:
       - induction: ab+1 (position after the first A)
       - duplicate: ab (position of the first A)
       - previous-token: T-2
       - self: T-1
       - first-token: 0
       - local: mean of [T-3, T-4, T-5, T-6]
       - baseline: mean over 50 random "other" positions
  5. Selectivity[cls] = mean_attn[cls] / baseline
  6. Classify each head by highest-selectivity class ≥30×

Ranks heads by INTEGRAL from olmoe_phase1_features.json (Phase 1 output).
Reports top-K classifications + precision-at-k + class breakdown.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import OlmoForCausalLM

from mamba2_per_head import build_induction_batch


def reconstruct_ab_indices(tokens, targets):
    """For each example, find the position of B (= targets[i]).
    Return ab_index = position-of-B - 1 (so ab_index points to A, ab_index+1 points to B).

    Each row has structure [... A B ... A], so we look for the unique
    occurrence of targets[i] (= B) in positions [0, T-1)."""
    n, T = tokens.shape
    ab_indices = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        match = (tokens[i, :-1] == targets[i]).nonzero(as_tuple=True)[0]
        if len(match) == 1:
            ab_indices[i] = match[0].item() - 1
        else:
            ab_indices[i] = -1
    return ab_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--features-json", default="olmoe_phase1_features.json")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=45,
                    help="K for top-K mech-interp display (writeup heuristic = 17-19% = ~45 for 256 heads)")
    ap.add_argument("--selectivity-threshold", type=float, default=30.0)
    ap.add_argument("--out", default="olmoe_mechinterp.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    print(f"Loading {args.model}@{args.revision} (eager attention for output_attentions=True)...")
    t0 = time.time()
    model = OlmoForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16,
                                              attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s")
    print(f"  arch: {n_layer}L x {cfg.hidden_size}d x {n_head}h (head_dim={head_dim})")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(n_examples=args.n_examples,
                                                seq_len=args.seq_len, rng=rng)
    T = tokens.shape[1]
    last_pos = T - 1
    prev_pos = T - 2
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0
    print(f"  valid examples: {int(valid.sum().item())}/{tokens.shape[0]}")

    print(f"\nExtracting attention weights at last query position (batch_size={args.batch_size})...")
    n = tokens.shape[0]
    attn_at_last = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True)
            for L in range(n_layer):
                attn_at_last[start:end, L] = out.attentions[L][:, :, last_pos, :].float().cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start % (args.batch_size * 20) == 0 and start > 0:
                rate = (start + args.batch_size) / (time.time() - t0)
                eta = (n - start - args.batch_size) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  attention extraction done in {time.time()-t0:.0f}s")

    print("\nComputing per-class selectivity ...")
    rng_b = np.random.RandomState(0)
    sample = []
    avoid = {0, last_pos, prev_pos}
    avoid.update(range(prev_pos - 4, prev_pos))
    while len(sample) < 50:
        rp = rng_b.randint(1, last_pos)
        if rp not in avoid and rp not in sample:
            sample.append(rp)
    baseline = attn_at_last[:, :, :, sample].mean(dim=(0, 3)).numpy()

    cls_attn = {}
    cls_attn["previous-token"] = attn_at_last[:, :, :, prev_pos].mean(dim=0).numpy()
    cls_attn["self"] = attn_at_last[:, :, :, last_pos].mean(dim=0).numpy()
    cls_attn["first-token"] = attn_at_last[:, :, :, 0].mean(dim=0).numpy()
    local_pos = [prev_pos - k for k in range(1, 5)]
    cls_attn["local"] = attn_at_last[:, :, :, local_pos].mean(dim=(0, 3)).numpy()

    induction_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    duplicate_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_valid = 0
    for i in range(n):
        if not valid[i]:
            continue
        ab = ab_indices[i].item()
        if ab < 0 or ab + 1 >= T:
            continue
        induction_sum += attn_at_last[i, :, :, ab + 1].double()
        duplicate_sum += attn_at_last[i, :, :, ab].double()
        n_valid += 1
    cls_attn["induction"] = (induction_sum / max(n_valid, 1)).numpy()
    cls_attn["duplicate-token"] = (duplicate_sum / max(n_valid, 1)).numpy()

    classes = ["induction", "previous-token", "duplicate-token",
               "first-token", "self", "local"]
    selectivity = {cls: cls_attn[cls] / np.maximum(baseline, 1e-8) for cls in classes}

    # Load integral ranking from Phase 1 features
    feats = json.load(open(args.features_json))["features"]
    ranked = []
    for L in range(n_layer):
        for H in range(n_head):
            f = feats[f"L{L}_H{H}"]
            ranked.append((L, H, f["integral"], f["spread"]))
    ranked.sort(key=lambda x: -x[2])  # descending by integral

    print(f"\n{'='*100}")
    print(f"Top-{args.top_k} OLMoE picks by INTEGRAL, classified (threshold ≥{args.selectivity_threshold}×):")
    print(f"{'='*100}")
    print(f"  {'rank':>4} {'head':<10} {'integral':>10}  {'best class':<18} {'sel':>10}  {'2nd class':<18} {'sel':>10}")

    classifications = []
    for rank, (L, H, integ, sp) in enumerate(ranked[:args.top_k], 1):
        per_cls = [(cls, float(selectivity[cls][L, H])) for cls in classes]
        per_cls.sort(key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        sec_cls, sec_sel = per_cls[1]
        classification = best_cls if best_sel >= args.selectivity_threshold else "unclassified"
        classifications.append({
            "rank": rank, "layer": L, "head": H,
            "integral": integ, "spread": sp,
            "classification": classification,
            "best_class": best_cls, "best_selectivity": best_sel,
            "second_class": sec_cls, "second_selectivity": sec_sel,
            "all_selectivities": {cls: float(selectivity[cls][L, H]) for cls in classes},
        })
        marker = "" if best_sel >= args.selectivity_threshold else "  ← UNCLASSIFIED"
        print(f"  {rank:>4} L{L:>2}H{H:<3}  {integ:>10.0f}  {best_cls:<18} {best_sel:>10.1f}x  "
              f"{sec_cls:<18} {sec_sel:>10.1f}x{marker}")

    print(f"\nPrecision-at-k:")
    for k in [5, 10, 15, 20, 30, args.top_k]:
        if k > len(classifications):
            continue
        cls = sum(1 for c in classifications[:k] if c["classification"] != "unclassified")
        print(f"  k={k}: {cls/k:.2f}  ({cls}/{k} classified)")

    cls_counts = {}
    for c in classifications:
        cls_counts[c["classification"]] = cls_counts.get(c["classification"], 0) + 1
    print(f"\nClass breakdown across top-{args.top_k}:")
    for k, v in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {v}")

    # Also save selectivity matrix for all 256 heads (for all-head capability screen in Phase 3)
    all_head_selectivity = {f"L{L}_H{H}": {cls: float(selectivity[cls][L, H]) for cls in classes}
                            for L in range(n_layer) for H in range(n_head)}

    out = {
        "model": args.model,
        "revision": args.revision,
        "selectivity_threshold": args.selectivity_threshold,
        "ranking": "integral",
        "n_examples_valid": int(valid.sum().item()),
        "n_examples": args.n_examples,
        "top_k": args.top_k,
        "classifications": classifications,
        "class_breakdown_topK": cls_counts,
        "all_head_selectivity": all_head_selectivity,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
