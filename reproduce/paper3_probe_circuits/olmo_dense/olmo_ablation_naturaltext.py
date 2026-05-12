"""Tier 2: natural-text causal ablation of the 4-head induction circuit.

Validates the synthetic-batch ablation finding on natural-text induction.

Per-example query positions and induction targets are loaded from the
saved natural batch. Metric: mean logit of the target token S at the
query position. Ablation conditions:
  - baseline (no ablation)
  - ablate_4_induction_heads (L5H10, L7H0, L9H8, L12H14)
  - ablate_matched_random_same_layers (4 random heads in L5,7,9,12)
  - ablate_all_heads_those_layers (upper bound, all heads in L5/7/9/12)

If the 4-head ablation drops logit-of-S more than matched_random, the
synthetic-batch circuit identification extends to natural-text behavior.
If it doesn't, the synthetic methodology identified a synthetic-specific
signal.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoForCausalLM


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate_natural(model, tokens, query_pos, targets, device, batch_size=4):
    """Per-example query positions; targets are token IDs to score at each
    example's query position. Returns dict with mean logit-of-S + accuracy."""
    n = tokens.shape[0]
    losses, accs1, accs5, sum_logit_S = [], [], [], 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            qp = query_pos[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok).logits  # (B, T, V)
            B = end - start
            # Gather logits at per-example query positions
            qp_idx = qp.view(B, 1, 1).expand(B, 1, logits.shape[-1])
            last_logits = logits.gather(1, qp_idx).squeeze(1)  # (B, V)
            loss = F.cross_entropy(last_logits, tgt, reduction="none")
            losses.append(loss.cpu().numpy())
            top1 = last_logits.argmax(dim=-1)
            accs1.append((top1 == tgt).float().cpu().numpy())
            top5 = last_logits.topk(5, dim=-1).indices
            accs5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
            sum_logit_S += last_logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(accs1).mean()),
        "acc_top5": float(np.concatenate(accs5).mean()),
        "mean_logit_S": float(sum_logit_S / n),
    }


def run_condition(model, spec, tokens, query_pos, targets, device, head_dim, batch_size):
    handles = []
    for layer_idx, heads in spec.items():
        if not heads:
            continue
        h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    try:
        return evaluate_natural(model, tokens, query_pos, targets, device, batch_size)
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-1B-0724-hf")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-file", default="natural_induction_batch.pt")
    ap.add_argument("--mechinterp-json", required=True,
                    help="mech-interp JSON with all_head_selectivity to derive induction circuit")
    ap.add_argument("--induction-threshold", type=float, default=50.0,
                    help="selectivity threshold for the all-head induction screen")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out", default="olmo_ablation_naturaltext.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    data = torch.load(args.batch_file, weights_only=False)
    tokens = data["tokens"]
    query_pos = data["query_pos"]
    targets = data["targets"]
    n, T = tokens.shape
    print(f"natural batch: n={n}, seq_len={T}")
    print(f"  query_pos median = {int(query_pos.median().item())}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    model = OlmoForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16)
    model = model.to(device).eval()
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head} hd={head_dim}")

    # Discover the induction circuit from mech-interp output (all heads with
    # induction-selectivity >= induction_threshold on the synthetic batch).
    # This is the all-head capability screen — same logic as olmo_ablation.py.
    mech = json.load(open(args.mechinterp_json))
    threshold = args.induction_threshold
    induction_circuit = {}
    for k, sels in mech["all_head_selectivity"].items():
        if sels.get("induction", 0.0) >= threshold:
            L = int(k.split("_")[0][1:])
            H = int(k.split("_")[1][1:])
            induction_circuit.setdefault(L, []).append(H)
    n_circuit = sum(len(v) for v in induction_circuit.values())
    print(f"Auto-discovered induction circuit (induction-selectivity ≥ {threshold}×): {n_circuit} heads")
    print(f"  layers: {sorted(induction_circuit.keys())}")
    print(f"  per layer: {[(L, induction_circuit[L]) for L in sorted(induction_circuit.keys())]}")
    if n_circuit == 0:
        print("  WARNING: no induction-selective heads found at this threshold!")
        print("  Will only run baseline + matched_random (no circuit to ablate)")

    # Matched random: 4 random heads, one per layer in 5/7/9/12, no overlap with circuit
    rng_c = np.random.RandomState(args.random_seed)
    matched_random = {}
    for L, picks in induction_circuit.items():
        eligible = [h for h in range(n_head) if h not in picks]
        matched_random[L] = sorted(rng_c.choice(eligible, size=len(picks), replace=False).tolist())

    # Upper bound: all heads in L5/L7/L9/L12
    upper_bound = {L: list(range(n_head)) for L in induction_circuit.keys()}

    conditions = [("baseline", {})]
    if n_circuit > 0:
        circuit_label = ",".join(f"L{L}H{H}" for L, hs in sorted(induction_circuit.items()) for H in hs)
        conditions += [
            (f"ablate_induction_circuit ({circuit_label})", induction_circuit),
            (f"ablate_matched_random_same_layers ({matched_random})", matched_random),
            ("ablate_all_heads_in_those_layers (upper bound)", upper_bound),
        ]
        for L, H_list in induction_circuit.items():
            for H in H_list:
                conditions.append((f"ablate_L{L}H{H}_only", {L: [H]}))

    print(f"\n{'='*100}")
    print(f"Running {len(conditions)} conditions on natural-text batch (n={n}):")
    print(f"{'='*100}")
    print(f"  {'condition':<70} {'loss':>8} {'top1':>8} {'top5':>8} {'logit_S':>10}")
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, spec, tokens, query_pos, targets, device, head_dim,
                          args.batch_size)
        elapsed = time.time() - t0
        print(f"  {name:<70} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f} "
              f"{r['mean_logit_S']:>+10.3f}  [{elapsed:.0f}s]", flush=True)
        results.append({"name": name,
                        "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": sum(len(v) for v in spec.values()),
                        **r})
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "revision": args.revision,
                       "batch_file": args.batch_file, "n_examples": n,
                       "induction_circuit": induction_circuit,
                       "matched_random": matched_random,
                       "results": results}, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
