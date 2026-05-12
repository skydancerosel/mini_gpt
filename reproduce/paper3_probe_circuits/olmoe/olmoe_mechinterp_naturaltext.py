"""OLMoE mech-interp classification on natural-text induction batch.

Same methodology as `olmoe_mechinterp.py` but with PER-EXAMPLE query
positions and PER-EXAMPLE canonical target positions (instead of the
fixed last-position used for synthetic batches).

Hypothesis being tested: the 68% BOS dominance we observed on the
synthetic batch is a synthetic-batch artifact (random tokens with no
real BOS structure) rather than a real OLMoE attention property. If
natural-text BOS fraction drops, the synthetic finding was an artifact;
if it stays high, OLMoE has an architectural BOS attractor.

Per-example positions used per the natural batch:
  query_pos[i]:    where to measure attention FROM
  first_T_pos[i]:  position of first T (duplicate-token target)
  first_T_pos[i] + 1: position of S (induction target)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import OlmoeForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--batch-file", default="natural_induction_batch.pt")
    ap.add_argument("--features-json", default="olmoe_phase1_features.json")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=45)
    ap.add_argument("--selectivity-threshold", type=float, default=30.0)
    ap.add_argument("--n-baseline-positions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="olmoe_mechinterp_naturaltext.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    print(f"Loading batch: {args.batch_file}")
    data = torch.load(args.batch_file, weights_only=False)
    tokens = data["tokens"]
    query_pos = data["query_pos"]
    first_T_pos = data["first_T_pos"]
    n, T = tokens.shape
    print(f"  n_examples={n}, seq_len={T}")
    print(f"  query_pos: min={query_pos.min().item()}, "
          f"median={int(query_pos.median().item())}, max={query_pos.max().item()}")
    print(f"  first_T_pos: min={first_T_pos.min().item()}, "
          f"median={int(first_T_pos.median().item())}, max={first_T_pos.max().item()}")

    print(f"\nLoading {args.model}@{args.revision} (eager attention) ...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(args.model, revision=args.revision,
                                              dtype=torch.float16,
                                              attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head}")

    # Pre-compute per-example baseline-sample positions (consistent across all heads)
    # Per example: pick `n_baseline_positions` random positions avoiding all canonical targets
    rng = np.random.RandomState(args.seed)
    baseline_pos = torch.zeros(n, args.n_baseline_positions, dtype=torch.long)
    for i in range(n):
        qp = query_pos[i].item()
        fT = first_T_pos[i].item()
        # canonical: 0, qp, qp-1, fT, fT+1, qp-2..qp-5 (local)
        avoid = {0, qp, qp - 1, fT, fT + 1}
        avoid.update(range(qp - 5, qp - 1))  # local
        # sample
        picks = []
        for _ in range(200):  # try up to 200 random positions to find 50 valid
            rp = rng.randint(1, qp)  # positions strictly inside (1, qp)
            if rp not in avoid and rp not in picks:
                picks.append(rp)
            if len(picks) >= args.n_baseline_positions:
                break
        # Pad if needed (small examples)
        while len(picks) < args.n_baseline_positions:
            picks.append(picks[-1] if picks else 1)
        baseline_pos[i] = torch.tensor(picks[:args.n_baseline_positions])

    # Per-class attention sums (across examples), per (L, H)
    cls_sum = {cls: torch.zeros(n_layer, n_head, dtype=torch.float64)
               for cls in ["induction", "previous-token", "duplicate-token",
                            "first-token", "self", "local", "baseline"]}

    print(f"\nExtracting attention (batch={args.batch_size}) ...")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            tok = tokens[start:end].to(device)
            B = end - start
            qp_b = query_pos[start:end]  # (B,)
            fT_b = first_T_pos[start:end]  # (B,)
            bl_b = baseline_pos[start:end]  # (B, n_baseline_positions)
            out = model(tok, output_attentions=True)
            # out.attentions is a tuple of (B, n_head, T, T) per layer
            for L in range(n_layer):
                # attn at query_pos[i]: shape (B, n_head, T)
                attn = out.attentions[L]  # (B, n_head, T, T)
                # gather attn[b, :, qp[b], :] → (B, n_head, T)
                attn_at_q = attn.gather(2, qp_b.view(B, 1, 1, 1).expand(B, n_head, 1, T).to(device)).squeeze(2)
                # Now attn_at_q[b, h, k] = attention from qp[b] to k, for example b head h
                attn_at_q = attn_at_q.float().cpu()  # (B, n_head, T)

                # Per-example canonical targets:
                #   induction = fT + 1, duplicate = fT,
                #   prev = qp - 1, self = qp, first = 0,
                #   local = mean over qp-5..qp-2 (4 positions)
                ind_pos = fT_b + 1
                dup_pos = fT_b
                prev_pos = qp_b - 1
                self_pos = qp_b
                first_pos = torch.zeros(B, dtype=torch.long)
                local_idx = torch.stack([qp_b - 5, qp_b - 4, qp_b - 3, qp_b - 2], dim=1)  # (B, 4)

                def gather_at(positions):
                    # positions: (B,) → returns (B, n_head)
                    return attn_at_q.gather(2, positions.view(B, 1, 1).expand(B, n_head, 1)).squeeze(2)

                cls_sum["induction"][L] += gather_at(ind_pos).double().sum(0)
                cls_sum["duplicate-token"][L] += gather_at(dup_pos).double().sum(0)
                cls_sum["previous-token"][L] += gather_at(prev_pos).double().sum(0)
                cls_sum["self"][L] += gather_at(self_pos).double().sum(0)
                cls_sum["first-token"][L] += gather_at(first_pos).double().sum(0)
                # local: mean over 4 positions → (B, n_head)
                local_vals = torch.stack([gather_at(local_idx[:, j]) for j in range(4)], dim=0).mean(0)
                cls_sum["local"][L] += local_vals.double().sum(0)
                # baseline: mean over n_baseline_positions per example
                # bl_b shape (B, NP), gather attn_at_q at bl_b → (B, n_head, NP), mean over NP
                bl_attn = attn_at_q.gather(2, bl_b.view(B, 1, args.n_baseline_positions).expand(B, n_head, args.n_baseline_positions))
                cls_sum["baseline"][L] += bl_attn.mean(2).double().sum(0)
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start % (args.batch_size * 20) == 0 and start > 0:
                rate = (start + B) / (time.time() - t0)
                eta = (n - start - B) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  attention extraction done in {time.time()-t0:.0f}s")

    # Mean attention per class per (L, H), and selectivity (vs baseline)
    cls_mean = {cls: (s / n).numpy() for cls, s in cls_sum.items()}
    baseline = cls_mean["baseline"]
    classes = ["induction", "previous-token", "duplicate-token",
               "first-token", "self", "local"]
    selectivity = {cls: cls_mean[cls] / np.maximum(baseline, 1e-8) for cls in classes}

    # Load integral ranking
    feats = json.load(open(args.features_json))["features"]
    ranked = sorted(
        [(L, H, feats[f"L{L}_H{H}"]["integral"]) for L in range(n_layer) for H in range(n_head)],
        key=lambda x: -x[2]
    )

    print(f"\n{'='*100}")
    print(f"Top-{args.top_k} by INTEGRAL, classified on NATURAL-TEXT batch "
          f"(threshold ≥{args.selectivity_threshold}×):")
    print(f"{'='*100}")
    print(f"  {'rank':>4} {'head':<10} {'integral':>10}  {'best class':<18} {'sel':>10}  "
          f"{'2nd class':<18} {'sel':>10}")

    classifications = []
    for rank, (L, H, integ) in enumerate(ranked[:args.top_k], 1):
        per_cls = [(cls, float(selectivity[cls][L, H])) for cls in classes]
        per_cls.sort(key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        sec_cls, sec_sel = per_cls[1]
        classification = best_cls if best_sel >= args.selectivity_threshold else "unclassified"
        classifications.append({
            "rank": rank, "layer": L, "head": H,
            "integral": integ,
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

    # Whole-model class breakdown (all 256 heads ranked by integral)
    all_classified = []
    for (L, H, integ) in ranked:
        per_cls = sorted([(cls, float(selectivity[cls][L, H])) for cls in classes],
                         key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        all_classified.append({"L": L, "H": H,
                                "class": best_cls if best_sel >= args.selectivity_threshold else "unclassified",
                                "best_sel": best_sel,
                                "induction_sel": float(selectivity["induction"][L, H])})
    whole_cls_count = {}
    for c in all_classified:
        whole_cls_count[c["class"]] = whole_cls_count.get(c["class"], 0) + 1
    print(f"\n=== WHOLE-MODEL class breakdown (all {n_layer*n_head} heads) ===")
    for k, v in sorted(whole_cls_count.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {v:>3}  ({v*100/(n_layer*n_head):.1f}%)")

    # Induction-selectivity threshold table
    print(f"\n=== Induction-selectivity distribution (all {n_layer*n_head} heads) ===")
    for thresh in [10, 30, 50, 100, 200]:
        n_at = sum(1 for c in all_classified if c["induction_sel"] >= thresh)
        print(f"  selectivity ≥ {thresh:>4}x:  {n_at} heads")

    all_head_selectivity = {f"L{L}_H{H}": {cls: float(selectivity[cls][L, H]) for cls in classes}
                            for L in range(n_layer) for H in range(n_head)}

    out_data = {
        "model": args.model,
        "revision": args.revision,
        "batch_file": args.batch_file,
        "batch_type": "natural-text (OpenWebText)",
        "selectivity_threshold": args.selectivity_threshold,
        "ranking": "integral",
        "n_examples": int(n),
        "top_k": args.top_k,
        "classifications": classifications,
        "class_breakdown_topK": cls_counts,
        "class_breakdown_whole_model": whole_cls_count,
        "all_head_selectivity": all_head_selectivity,
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
