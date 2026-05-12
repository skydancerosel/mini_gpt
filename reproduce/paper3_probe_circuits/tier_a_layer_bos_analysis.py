"""Tier A #2: per-layer BOS-class distribution across all 5 models.

Tests whether OLMoE's "L0-L1 zero BOS, L3+ BOS-dominant" pattern is
universal across architectures and data, or OLMoE-specific.

Loads existing mech-interp JSONs and counts BOS-classified heads per
layer for each model.
"""
from __future__ import annotations

import json
import math

CLASSES = ["induction", "previous-token", "duplicate-token", "first-token",
           "self", "local"]


def classify_per_layer(mech_json_path, threshold=30.0):
    """Returns dict layer -> count of BOS-classified heads.

    Handles NaN/inf selectivity safely."""
    m = json.load(open(mech_json_path))
    all_sel = m["all_head_selectivity"]
    per_layer = {}  # layer -> BOS count
    layer_totals = {}
    for k, sels in all_sel.items():
        L = int(k.split("_")[0][1:])
        layer_totals[L] = layer_totals.get(L, 0) + 1
        valid = {c: sels[c] for c in CLASSES
                 if isinstance(sels[c], (int, float)) and sels[c] == sels[c] and not math.isinf(sels[c])}
        if not valid:
            continue
        best = max(valid, key=valid.get)
        if best == "first-token" and valid[best] >= threshold:
            per_layer[L] = per_layer.get(L, 0) + 1
    return per_layer, layer_totals


def main():
    files = [
        ("Pythia 160M",       "pythia_160m_mechinterp_fp32.json"),
        ("Pythia 410M",       "pythia_410m_mechinterp_fp32.json"),
        ("Pythia 1B",         "pythia_mechinterp.json"),
        ("OLMoE 1B-7B",       "olmoe_mechinterp.json"),
        ("OLMo 1B-0724-hf",   "olmo_mechinterp.json"),
    ]

    layers_per_model = {}
    bos_per_layer_per_model = {}

    for name, path in files:
        per_layer, totals = classify_per_layer(path)
        layers_per_model[name] = totals
        bos_per_layer_per_model[name] = per_layer

    # Build per-layer fraction table
    print("=" * 90)
    print("Per-layer BOS-class fraction across 5 models (synthetic batch, ≥30× threshold)")
    print("=" * 90)
    # Print model header line
    print(f"  {'Layer':<8}", end="")
    for name, _ in files:
        print(f" {name:>18}", end="")
    print()
    print("  " + "-" * 88)
    max_layers = max(max(layers_per_model[n].keys()) for n, _ in files) + 1
    for L in range(max_layers):
        row = f"  L{L:<6} "
        for name, _ in files:
            tot = layers_per_model[name].get(L, 0)
            bos = bos_per_layer_per_model[name].get(L, 0)
            if tot == 0:
                row += f" {'-':>18}"
            else:
                row += f" {bos:>3}/{tot:<3} ({bos*100/tot:>3.0f}%)"
                row = row[:-1] + " " * (18 - 11) + ""
                row = row.rstrip() + " " * (max(0, len(row) - len(row.rstrip())))
        # Simpler version: print bos/tot per cell
        row2 = f"  L{L:<6} "
        for name, _ in files:
            tot = layers_per_model[name].get(L, 0)
            bos = bos_per_layer_per_model[name].get(L, 0)
            if tot == 0:
                row2 += f"{'-':>18}"
            else:
                row2 += f"{bos:>5}/{tot:<3} ({bos*100/tot:>3.0f}%)"
        print(row2)

    # Layer-quartile aggregation for each model
    print("\n" + "=" * 90)
    print("Layer-quartile aggregation: BOS-class fraction by depth band")
    print("=" * 90)
    print(f"  {'Model':<22} {'Q1 (early)':>12} {'Q2':>10} {'Q3':>10} {'Q4 (late)':>12}")
    for name, _ in files:
        totals = layers_per_model[name]
        bos = bos_per_layer_per_model[name]
        L = max(totals.keys()) + 1
        q = L // 4
        quartile_bos = [0] * 4
        quartile_tot = [0] * 4
        for layer in range(L):
            if layer < q:
                qi = 0
            elif layer < 2 * q:
                qi = 1
            elif layer < 3 * q:
                qi = 2
            else:
                qi = 3
            quartile_bos[qi] += bos.get(layer, 0)
            quartile_tot[qi] += totals.get(layer, 0)
        cells = []
        for qi in range(4):
            if quartile_tot[qi] > 0:
                cells.append(f"{quartile_bos[qi]*100/quartile_tot[qi]:.0f}%")
            else:
                cells.append("-")
        print(f"  {name:<22} {cells[0]:>12} {cells[1]:>10} {cells[2]:>10} {cells[3]:>12}")

    # Save summary JSON
    out = {
        "per_layer_BOS_count": bos_per_layer_per_model,
        "per_layer_total": layers_per_model,
    }
    with open("tier_a_layer_bos_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote tier_a_layer_bos_summary.json")


if __name__ == "__main__":
    main()
