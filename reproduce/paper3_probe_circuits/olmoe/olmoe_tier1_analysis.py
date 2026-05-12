"""Tier 1 analyses on OLMoE — uses existing JSONs, no GPU.

A. Non-BOS precision-at-K: strip first-token classified heads, recompute
B. BOS-head per-layer distribution
C. PR-trajectory pattern of BOS-class heads vs other classes
D. How the 4 synthetic induction heads classify on natural text

Loads:
  - olmoe_phase1_features.json (integral ranking + trajectories)
  - olmoe_phase1_trajectory.json (PR per (L,H) per revision)
  - olmoe_mechinterp.json (synthetic classifications)
  - olmoe_mechinterp_naturaltext.json (unfiltered natural-text classifications)
  - olmoe_mechinterp_naturaltext_midseq.json (first_T >= 20 control)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np


def classify_all(all_sel, threshold=30.0):
    """Returns dict: 'L{L}_H{H}' -> class name (first-token, prev-token, etc., or 'unclassified')."""
    classes = ["induction", "previous-token", "duplicate-token",
               "first-token", "self", "local"]
    out = {}
    for key, sels in all_sel.items():
        best = max(classes, key=lambda c: sels[c])
        out[key] = best if sels[best] >= threshold else "unclassified"
    return out


def main():
    feats = json.load(open("olmoe_phase1_features.json"))["features"]
    traj = json.load(open("olmoe_phase1_trajectory.json"))
    mech_synth = json.load(open("olmoe_mechinterp.json"))
    mech_nat = json.load(open("olmoe_mechinterp_naturaltext.json"))
    mech_nat_mid = json.load(open("olmoe_mechinterp_naturaltext_midseq.json"))

    n_layer = traj["n_layer"]
    n_head = traj["num_heads"]

    # Per-head integral ranking
    ranked = sorted(
        [(L, H, feats[f"L{L}_H{H}"]["integral"]) for L in range(n_layer) for H in range(n_head)],
        key=lambda x: -x[2]
    )

    # Per-head class labels (synthetic and natural)
    cls_synth = classify_all(mech_synth["all_head_selectivity"])
    cls_nat = classify_all(mech_nat["all_head_selectivity"])
    cls_nat_mid = classify_all(mech_nat_mid["all_head_selectivity"])

    # ───────── Analysis A: non-BOS precision-at-K ─────────
    print("=" * 80)
    print("A. Non-BOS precision-at-K (strip first-token classified heads from ranking)")
    print("=" * 80)
    # Filter out BOS-classified heads (synthetic), then rank by integral on the rest
    non_bos_ranked = [(L, H, integ) for (L, H, integ) in ranked
                      if cls_synth[f"L{L}_H{H}"] != "first-token"]
    print(f"  Total heads: {n_layer*n_head}  BOS-classified: {sum(1 for c in cls_synth.values() if c=='first-token')}")
    print(f"  Non-BOS heads remaining: {len(non_bos_ranked)}")
    print(f"\n  Original precision-at-K (with BOS heads included for comparison):")
    print(f"  {'K':>4}  {'orig_prec':>10}  {'%non-BOS-heads':>16}  {'non-BOS prec':>14}")
    for K in [5, 10, 15, 20, 30, 45, 60, 80, 100, 130, 160, len(non_bos_ranked)]:
        if K > len(ranked):
            continue
        orig_prec = sum(1 for L, H, _ in ranked[:K] if cls_synth[f"L{L}_H{H}"] != "unclassified") / K
        if K > len(non_bos_ranked):
            continue
        non_bos_K = non_bos_ranked[:K]
        non_bos_prec = sum(1 for L, H, _ in non_bos_K if cls_synth[f"L{L}_H{H}"] != "unclassified") / K
        pct_nb = K * 100 / len(non_bos_ranked)
        print(f"  {K:>4}  {orig_prec:>9.2f}   {pct_nb:>14.1f}%  {non_bos_prec:>14.2f}")

    print(f"\n  Non-BOS classified-fraction (no K): "
          f"{sum(1 for c in [cls_synth[f'L{L}_H{H}'] for L, H, _ in non_bos_ranked] if c != 'unclassified')}/{len(non_bos_ranked)}"
          f" = {sum(1 for c in [cls_synth[f'L{L}_H{H}'] for L, H, _ in non_bos_ranked] if c != 'unclassified') / len(non_bos_ranked) * 100:.1f}%")
    print(f"  Pythia writeup conserved fraction:    17-19%")

    # ───────── Analysis B: BOS heads per-layer distribution ─────────
    print("\n" + "=" * 80)
    print("B. BOS-head per-layer distribution (synthetic, natural, mid-seq)")
    print("=" * 80)
    per_layer_synth = np.zeros(n_layer, dtype=int)
    per_layer_nat = np.zeros(n_layer, dtype=int)
    per_layer_nat_mid = np.zeros(n_layer, dtype=int)
    for L in range(n_layer):
        for H in range(n_head):
            if cls_synth[f"L{L}_H{H}"] == "first-token":
                per_layer_synth[L] += 1
            if cls_nat[f"L{L}_H{H}"] == "first-token":
                per_layer_nat[L] += 1
            if cls_nat_mid[f"L{L}_H{H}"] == "first-token":
                per_layer_nat_mid[L] += 1

    print(f"  {'L':>3}  {'BOS/16 synth':>14}  {'BOS/16 natural':>16}  {'BOS/16 mid-seq':>16}")
    for L in range(n_layer):
        print(f"  L{L:>2}  {per_layer_synth[L]:>9}/16     {per_layer_nat[L]:>9}/16        {per_layer_nat_mid[L]:>9}/16")
    print(f"  TOTAL {per_layer_synth.sum():>5}/256   {per_layer_nat.sum():>9}/256       {per_layer_nat_mid.sum():>9}/256")

    # ───────── Analysis C: PR-trajectory by class ─────────
    print("\n" + "=" * 80)
    print("C. PR trajectory by class (synthetic classification)")
    print("=" * 80)
    classes = ["first-token", "previous-token", "self", "induction", "duplicate-token", "unclassified"]
    steps = np.array(traj["ckpt_step"])
    tokens_B = np.array(traj["ckpt_tokens_B"], dtype=float)
    pr_arr = np.zeros((n_layer * n_head, len(steps)))
    for L in range(n_layer):
        for H in range(n_head):
            pr_arr[L * n_head + H] = traj["pr"][f"L{L}_H{H}"]

    head_classes = np.array([cls_synth[f"L{L}_H{H}"] for L in range(n_layer) for H in range(n_head)])

    print(f"  {'class':<18} {'n':>5}  mean PR at each revision (10 points):")
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        mask = head_classes == cls
        if mask.sum() == 0:
            continue
        cls_pr = pr_arr[mask].mean(axis=0)
        std_pr = pr_arr[mask].std(axis=0)
        print(f"  {cls:<18} {mask.sum():>5}  {[f'{v:.1f}' for v in cls_pr]}")
        ax.plot(tokens_B, cls_pr, marker="o", lw=1.5, ms=4, label=f"{cls} (n={mask.sum()})")
        ax.fill_between(tokens_B, cls_pr - std_pr, cls_pr + std_pr, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (B, log scale)")
    ax.set_ylabel("mean PR (± 1 std)")
    ax.set_title("PR trajectory by mech-interp class (synthetic-batch classification)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig("olmoe_tier1_pr_trajectory_by_class.png", dpi=130, bbox_inches="tight")
    print(f"\n  saved olmoe_tier1_pr_trajectory_by_class.png")

    # ───────── Analysis D: 4 synthetic induction heads on natural text ─────────
    print("\n" + "=" * 80)
    print("D. The 4 synthetic induction-screen heads classified on natural text")
    print("=" * 80)
    induction_heads = [(5, 10), (7, 0), (9, 8), (12, 14)]
    print(f"  {'head':<8}  {'synth class (sel)':<30}  {'natural class (sel)':<30}  {'natural-mid class (sel)':<30}")
    for L, H in induction_heads:
        key = f"L{L}_H{H}"
        s_sels = mech_synth["all_head_selectivity"][key]
        n_sels = mech_nat["all_head_selectivity"][key]
        nm_sels = mech_nat_mid["all_head_selectivity"][key]
        s_best = max(s_sels, key=s_sels.get)
        n_best = max(n_sels, key=n_sels.get)
        nm_best = max(nm_sels, key=nm_sels.get)
        s_ind = s_sels["induction"]
        n_ind = n_sels["induction"]
        nm_ind = nm_sels["induction"]
        print(f"  L{L:>2}H{H:<3}  {s_best:<15} ({s_sels[s_best]:.1f}x; ind={s_ind:.1f}x)  "
              f"  {n_best:<15} ({n_sels[n_best]:.1f}x; ind={n_ind:.1f}x)  "
              f"  {nm_best:<15} ({nm_sels[nm_best]:.1f}x; ind={nm_ind:.1f}x)")

    # Summary save
    out = {
        "non_bos_classified_fraction": float(
            sum(1 for L, H, _ in non_bos_ranked if cls_synth[f"L{L}_H{H}"] != "unclassified") / len(non_bos_ranked)),
        "non_bos_total": int(len(non_bos_ranked)),
        "bos_per_layer_synth": per_layer_synth.tolist(),
        "bos_per_layer_natural": per_layer_nat.tolist(),
        "bos_per_layer_natural_midseq": per_layer_nat_mid.tolist(),
        "induction_heads_cross_classification": [
            {"L": L, "H": H,
             "synth": {"best_class": max(mech_synth["all_head_selectivity"][f"L{L}_H{H}"], key=mech_synth["all_head_selectivity"][f"L{L}_H{H}"].get),
                       "induction_sel": mech_synth["all_head_selectivity"][f"L{L}_H{H}"]["induction"]},
             "natural": {"best_class": max(mech_nat["all_head_selectivity"][f"L{L}_H{H}"], key=mech_nat["all_head_selectivity"][f"L{L}_H{H}"].get),
                         "induction_sel": mech_nat["all_head_selectivity"][f"L{L}_H{H}"]["induction"]},
             "natural_midseq": {"best_class": max(mech_nat_mid["all_head_selectivity"][f"L{L}_H{H}"], key=mech_nat_mid["all_head_selectivity"][f"L{L}_H{H}"].get),
                                 "induction_sel": mech_nat_mid["all_head_selectivity"][f"L{L}_H{H}"]["induction"]},
             }
            for (L, H) in induction_heads
        ],
    }
    with open("olmoe_tier1_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved olmoe_tier1_summary.json")


if __name__ == "__main__":
    main()
