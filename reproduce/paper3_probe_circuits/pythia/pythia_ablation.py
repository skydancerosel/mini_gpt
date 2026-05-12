"""Phase 3: causal ablation on OLMoE — Pythia 410M methodology.

Port of pythia_410m_ablation.py + the all-head capability screen idea
from pythia_410m_distributed_induction_test.py.

Conditions:
  1. baseline (no ablation)
  2. ablate_top45_by_integral (group ablation of all spectral picks)
  3. ablate_matched_random_same_layers (45 random heads in same layers
     as picks, no overlap — controls for "is it the layer or the heads")
  4. ablate_full_layers (all heads in spec-pick layers — upper bound)
  5. ablate_induction_screen_>=50x (ALL 256 heads scored, ablate the
     induction-selective set — writeup's prescription for low-induction-
     count case, where top-K spectral picks include too few induction
     heads to fully tank induction)
  6. Individual ablation of the 2 induction-classified heads from
     Phase 2 (verify they each matter on their own)

Metrics per condition: loss, top-1 acc, top-5 acc, mean logit-of-target-B.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM

from mamba2_per_head import build_induction_batch


def make_pre_hook(heads_in_layer, head_dim):
    """Pre-hook factory: zeros the slice for each head in heads_in_layer
    of a [B, T, hidden] input to o_proj."""
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, batch_size=4):
    """Returns dict with loss, top-1 acc, top-5 acc, mean logit_B."""
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    losses, accs1, accs5 = [], [], []
    sum_logit_B = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok).logits[:, last, :]
            B = end - start
            loss = F.cross_entropy(logits, tgt, reduction="none")
            losses.append(loss.cpu().numpy())
            top1 = logits.argmax(dim=-1)
            accs1.append((top1 == tgt).float().cpu().numpy())
            top5 = logits.topk(5, dim=-1).indices
            accs5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
            sum_logit_B += logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(accs1).mean()),
        "acc_top5": float(np.concatenate(accs5).mean()),
        "mean_logit_B": float(sum_logit_B / n),
    }


def run_condition(model, spec, tokens, targets, device, head_dim, batch_size):
    """spec: dict layer_idx -> list of head indices to ablate. Returns eval dict."""
    handles = []
    for layer_idx, heads in spec.items():
        if not heads:
            continue
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    try:
        return evaluate(model, tokens, targets, device, batch_size)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--features-json", default="olmoe_phase1_features.json")
    ap.add_argument("--mechinterp-json", default="olmoe_mechinterp.json")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=45)
    ap.add_argument("--induction-threshold", type=float, default=50.0)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out", default="olmoe_ablation.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    print(f"induction batch: {tuple(tokens.shape)}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16)
    model = model.to(device).eval()
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head} hd={head_dim}")

    # Load integral ranking → top-K spectral picks
    feats = json.load(open(args.features_json))["features"]
    ranked = sorted(
        [(L, H, feats[f"L{L}_H{H}"]["integral"]) for L in range(n_layer) for H in range(n_head)],
        key=lambda x: -x[2]
    )
    top_K = ranked[:args.top_k]
    spectral_picks = {}
    for L, H, _ in top_K:
        spectral_picks.setdefault(L, []).append(H)
    print(f"\nTop-{args.top_k} spectral picks span layers: {sorted(spectral_picks.keys())}")
    print(f"  picks per layer: {[(L, len(spectral_picks[L])) for L in sorted(spectral_picks.keys())]}")

    # Matched random: same layers, count per layer = min(picks, eligible)
    # (caps when picks fill almost all heads in a layer; otherwise matches)
    rng_c = np.random.RandomState(args.random_seed)
    matched_random = {}
    matched_random_short = False
    for L, picks in spectral_picks.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_sample = min(len(picks), len(eligible))
        if n_sample < len(picks):
            matched_random_short = True
        matched_random[L] = sorted(rng_c.choice(eligible, size=n_sample, replace=False).tolist())
    if matched_random_short:
        total_matched = sum(len(v) for v in matched_random.values())
        total_picks = sum(len(v) for v in spectral_picks.values())
        print(f"  NOTE: matched_random capped per layer (total {total_matched} vs {total_picks} picks); "
              f"some layers had more picks than non-overlap heads")

    # Upper bound: all heads in spec-pick layers
    upper_bound = {L: list(range(n_head)) for L in spectral_picks.keys()}

    # All-head induction screen ≥ threshold (from Phase 2 mech-interp)
    mech = json.load(open(args.mechinterp_json))
    all_sel = mech["all_head_selectivity"]
    induction_screen = {}
    for k, sels in all_sel.items():
        if sels.get("induction", 0.0) >= args.induction_threshold:
            L = int(k.split("_")[0][1:])
            H = int(k.split("_")[1][1:])
            induction_screen.setdefault(L, []).append(H)
    n_induction_screen = sum(len(v) for v in induction_screen.values())
    print(f"\nAll-head induction screen ≥{args.induction_threshold}×: {n_induction_screen} heads")
    if n_induction_screen > 0:
        print(f"  layers: {sorted(induction_screen.keys())}")
        print(f"  per layer: {[(L, induction_screen[L]) for L in sorted(induction_screen.keys())]}")

    # Individual induction-classified heads from Phase 2 top-K classifications
    individual_induction = [(c["layer"], c["head"]) for c in mech["classifications"]
                            if c["classification"] == "induction"]
    print(f"\nInduction-classified heads in Phase 2 top-{args.top_k}: {individual_induction}")

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_by_integral", spectral_picks),
        ("ablate_matched_random_same_layers", matched_random),
        ("ablate_full_spec_layers (upper bound)", upper_bound),
    ]
    if n_induction_screen > 0:
        conditions.append((f"ablate_induction_screen_>={args.induction_threshold:.0f}x ({n_induction_screen}h)", induction_screen))
    for L, H in individual_induction:
        conditions.append((f"ablate_L{L}H{H}_only (induction class)", {L: [H]}))

    print(f"\n{'='*100}")
    print(f"Running {len(conditions)} conditions (n={args.n_examples}, batch={args.batch_size}):")
    print(f"{'='*100}")
    print(f"  {'condition':<58} {'loss':>8} {'top1':>8} {'top5':>8} {'logitB':>9}")

    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, spec, tokens, targets, device, head_dim, args.batch_size)
        elapsed = time.time() - t0
        print(f"  {name:<58} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f} "
              f"{r['mean_logit_B']:>+9.3f}  [{elapsed:.0f}s]", flush=True)
        results.append({"name": name,
                        "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": sum(len(v) for v in spec.values()),
                        **r})
        # Incremental save in case of crash
        out = {"model": args.model, "revision": args.revision,
               "n_examples": args.n_examples, "top_k": args.top_k,
               "induction_threshold": args.induction_threshold,
               "spectral_picks_layers": list(spectral_picks.keys()),
               "n_induction_screen": n_induction_screen,
               "results": results}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
