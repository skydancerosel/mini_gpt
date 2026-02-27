import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
β2 ablation analysis suite.

Per-run analysis:
    python beta2_analysis.py --run-dir runs/beta2_ablation/pilot_..._b20.95_s42/

Cross-run comparison (after all per-run analyses):
    python beta2_analysis.py --base-dir runs/beta2_ablation/ --compare

Reheating summary:
    python beta2_analysis.py --base-dir runs/beta2_ablation/ --reheat-summary

All-in-one (per-run + compare + reheat):
    python beta2_analysis.py --base-dir runs/beta2_ablation/ --all
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from attractor_analysis import (
    TRUNK_PATTERN,
    flatten_state_dict_filtered,
    load_checkpoint,
    load_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4.1 Backbone Geometry
# ═══════════════════════════════════════════════════════════════════════════

def backbone_geometry(run_dir, window=(600, 2000), rotation_window=8):
    """Row-normalized uncentered PCA on trunk params in window.

    Returns (metrics_dict, v_backbone_numpy_array).
    """
    run_dir = Path(run_dir)

    # Find checkpoints in window
    ckpt_files = sorted(run_dir.glob("ckpt_*.pt"))
    all_steps = sorted(int(f.stem.split("_")[1]) for f in ckpt_files)
    steps = [s for s in all_steps if window[0] <= s <= window[1]]

    if len(steps) < 3:
        raise ValueError(f"Need >=3 checkpoints in [{window[0]},{window[1]}], got {len(steps)}")

    print(f"  Loading {len(steps)} checkpoints in [{window[0]}, {window[1]}]")

    # Load θ_0 (first in window)
    theta_0 = flatten_state_dict_filtered(
        load_checkpoint(run_dir, steps[0])["model_state_dict"])

    # Build drift matrix
    drifts = []
    drift_steps = []
    for s in steps[1:]:
        theta = flatten_state_dict_filtered(
            load_checkpoint(run_dir, s)["model_state_dict"])
        drifts.append(theta - theta_0)
        drift_steps.append(s)

    X = torch.stack(drifts).numpy()  # (T-1, D)

    # Row-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_rn = X / norms

    # Uncentered SVD
    _, S, Vt = np.linalg.svd(X_rn, full_matrices=False)
    var = S ** 2
    total = var.sum()
    ratio = var / max(total, 1e-12)
    cumul = np.cumsum(ratio)

    pc1_pct = float(ratio[0] * 100)
    k95 = int((cumul < 0.95).sum()) + 1
    k99 = int((cumul < 0.99).sum()) + 1

    # Drift magnitude
    theta_end = flatten_state_dict_filtered(
        load_checkpoint(run_dir, steps[-1])["model_state_dict"])
    drift_mag = float((theta_end - theta_0).norm())

    # PC1 rotation stability (rolling window at trunk level)
    n_drifts = len(drifts)
    cos_vals = []
    cos_steps = []

    if n_drifts >= rotation_window + 1:
        pc1_vectors = []
        for w in range(n_drifts - rotation_window + 1):
            X_win = torch.stack(drifts[w:w + rotation_window]).numpy()
            norms_win = np.linalg.norm(X_win, axis=1, keepdims=True)
            X_win_rn = X_win / np.maximum(norms_win, 1e-12)
            _, _, Vt_win = np.linalg.svd(X_win_rn, full_matrices=False)
            pc1_vectors.append(Vt_win[0])

        for i in range(len(pc1_vectors) - 1):
            cos_val = abs(float(np.dot(pc1_vectors[i], pc1_vectors[i + 1])))
            cos_vals.append(cos_val)
            cos_steps.append(drift_steps[i + rotation_window])

    v_backbone = Vt[0]  # numpy, for update alignment

    result = {
        "pc1_pct": pc1_pct,
        "k95": k95,
        "k99": k99,
        "drift_magnitude": drift_mag,
        "window": list(window),
        "n_checkpoints": len(steps),
        "explained_ratio": ratio[:min(10, len(ratio))].tolist(),
        "pc1_rotation": {
            "steps": cos_steps,
            "cos_pc1": cos_vals,
            "mean_cos": float(np.mean(cos_vals)) if cos_vals else None,
            "min_cos": float(np.min(cos_vals)) if cos_vals else None,
        },
    }

    print(f"    PC1={pc1_pct:.1f}%, k95={k95}, k99={k99}, "
          f"drift={drift_mag:.2f}")
    if cos_vals:
        print(f"    PC1 rotation: mean |cos|={np.mean(cos_vals):.4f}, "
              f"min |cos|={np.min(cos_vals):.4f}")

    return result, v_backbone


# ═══════════════════════════════════════════════════════════════════════════
# 4.2 Update-direction Alignment
# ═══════════════════════════════════════════════════════════════════════════

def update_alignment(run_dir, v_backbone,
                     window_starts=(800, 1200, 1600, 2200, 3000, 3600),
                     delta=200):
    """Compute 200-step update direction and alignment with backbone."""
    run_dir = Path(run_dir)
    v_b = torch.from_numpy(v_backbone).float()

    results = []
    for t in window_starts:
        try:
            theta_t = flatten_state_dict_filtered(
                load_checkpoint(run_dir, t)["model_state_dict"])
            theta_td = flatten_state_dict_filtered(
                load_checkpoint(run_dir, t + delta)["model_state_dict"])
        except FileNotFoundError:
            print(f"    Skipping window {t}: checkpoint not found")
            continue

        u = theta_td - theta_t
        u_norm = float(u.norm())
        if u_norm < 1e-12:
            cos_val = 0.0
        else:
            cos_val = float(torch.dot(u, v_b)) / u_norm

        results.append({
            "window_start": t,
            "window_end": t + delta,
            "abs_cos": abs(cos_val),
            "signed_cos": cos_val,
            "update_norm": u_norm,
        })
        print(f"    t={t}: |cos|={abs(cos_val):.4f}, signed={cos_val:.4f}, "
              f"||u||={u_norm:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.3 Interference Metric (optional)
# ═══════════════════════════════════════════════════════════════════════════

def interference_metric(run_dir, steps_to_check=(1600, 3000),
                        device="cpu", n_batches=8):
    """Compute cos(g_LM, g_probe) at given checkpoints (trunk-level)."""
    run_dir = Path(run_dir)

    # Build datasets (codeword seed always 42)
    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)
    cw_path = run_dir / "codewords.json"
    data = build_datasets(
        cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    vocab_size = len(data["tokenizer"])
    train_loader = DataLoader(
        data["train_dataset"], batch_size=64,
        shuffle=True, drop_last=True, num_workers=0)

    ce = nn.CrossEntropyLoss(reduction='none')

    results = []
    for step in steps_to_check:
        try:
            ckpt = load_checkpoint(run_dir, step, device=device)
        except FileNotFoundError:
            print(f"    Skipping step {step}: checkpoint not found")
            continue

        model = GPTModel(
            vocab_size=vocab_size, seq_len=cfg.seq_len,
            d_model=cfg.d_model, n_layer=cfg.n_layer,
            n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt

        # Build param name → param mapping for trunk params
        trunk_params = []
        for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
            if TRUNK_PATTERN.match(name):
                trunk_params.append((name, param))

        cos_values = []
        data_iter = iter(train_loader)

        for i in range(n_batches):
            try:
                input_ids, targets, probe_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, targets, probe_mask = next(data_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            probe_mask = probe_mask.to(device)
            pmask = probe_mask.bool()
            lm_mask = ~pmask & (targets != -100)

            # LM gradient
            model.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)
            loss_flat = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_all = loss_flat.view(targets.shape)
            lm_loss = (loss_all[lm_mask].mean()
                       if lm_mask.any()
                       else torch.tensor(0.0, device=device))
            lm_loss.backward()

            g_lm_parts = []
            for name, param in trunk_params:
                if param.grad is not None:
                    g_lm_parts.append(param.grad.cpu().reshape(-1).float())
                else:
                    g_lm_parts.append(torch.zeros(param.numel()))
            g_lm = torch.cat(g_lm_parts)

            # Probe gradient
            model.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)
            loss_flat = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_all = loss_flat.view(targets.shape)
            p_loss = (loss_all[pmask].mean()
                      if pmask.any()
                      else torch.tensor(0.0, device=device))
            p_loss.backward()

            g_probe_parts = []
            for name, param in trunk_params:
                if param.grad is not None:
                    g_probe_parts.append(param.grad.cpu().reshape(-1).float())
                else:
                    g_probe_parts.append(torch.zeros(param.numel()))
            g_probe = torch.cat(g_probe_parts)

            # Cosine
            g_lm_norm = g_lm.norm()
            g_probe_norm = g_probe.norm()
            if g_lm_norm > 1e-12 and g_probe_norm > 1e-12:
                cos = float(torch.dot(g_lm, g_probe) / (g_lm_norm * g_probe_norm))
            else:
                cos = 0.0
            cos_values.append(cos)

        mean_cos = float(np.mean(cos_values))
        std_cos = float(np.std(cos_values))

        results.append({
            "step": step,
            "mean_cos": mean_cos,
            "std_cos": std_cos,
            "cos_values": cos_values,
        })

        print(f"    Step {step}: cos(g_LM, g_probe) = {mean_cos:.4f} "
              f"+/- {std_cos:.4f}")

        del model
        if device == "mps":
            torch.mps.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Per-run analysis
# ═══════════════════════════════════════════════════════════════════════════

def per_run_analysis(run_dir, device="cpu", skip_interference=False):
    """Run all per-run analyses and save JSONs."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    # 4.1 Backbone geometry
    print("\n  === 4.1 Backbone geometry ===")
    backbone, v_backbone = backbone_geometry(run_dir)
    with open(out_dir / "backbone_metrics.json", "w") as f:
        json.dump(backbone, f, indent=2)
    print(f"    Saved {out_dir / 'backbone_metrics.json'}")

    # 4.2 Update alignment
    print("\n  === 4.2 Update-direction alignment ===")
    alignment = update_alignment(run_dir, v_backbone)
    with open(out_dir / "update_alignment.json", "w") as f:
        json.dump(alignment, f, indent=2)
    print(f"    Saved {out_dir / 'update_alignment.json'}")

    # 4.3 Interference (optional)
    if not skip_interference:
        print("\n  === 4.3 Interference metric ===")
        interference = interference_metric(run_dir, device=device)
        with open(out_dir / "interference_metrics.json", "w") as f:
            json.dump(interference, f, indent=2)
        print(f"    Saved {out_dir / 'interference_metrics.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-run comparison
# ═══════════════════════════════════════════════════════════════════════════

def discover_runs(base_dir):
    """Find all β2 runs and return {beta2: run_dir} mapping."""
    base_dir = Path(base_dir)
    runs = {}
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or d.name == "analysis" or d.name == "summary":
            continue
        m = re.search(r"b2([\d.]+)", d.name)
        if m:
            b2 = float(m.group(1))
            runs[b2] = d
    return runs


def load_run_data(runs):
    """Load per-run JSONs for all discovered runs."""
    data = {}
    for b2, d in sorted(runs.items()):
        entry = {"beta2": b2, "run_dir": str(d)}

        bb_path = d / "analysis" / "backbone_metrics.json"
        ua_path = d / "analysis" / "update_alignment.json"
        if_path = d / "analysis" / "interference_metrics.json"
        met_path = d / "pilot_metrics.json"

        if bb_path.exists():
            with open(bb_path) as f:
                entry["backbone"] = json.load(f)
        if ua_path.exists():
            with open(ua_path) as f:
                entry["update_alignment"] = json.load(f)
        if if_path.exists():
            with open(if_path) as f:
                entry["interference"] = json.load(f)
        if met_path.exists():
            with open(met_path) as f:
                entry["metrics"] = json.load(f)

        data[b2] = entry
    return data


def plot_backbone_summary(data, out_dir):
    """4-panel: PC1%, k95, k99, drift vs β2."""
    beta2s = sorted(b for b in data if "backbone" in data[b])
    if not beta2s:
        return

    pc1s = [data[b]["backbone"]["pc1_pct"] for b in beta2s]
    k95s = [data[b]["backbone"]["k95"] for b in beta2s]
    k99s = [data[b]["backbone"]["k99"] for b in beta2s]
    drifts = [data[b]["backbone"]["drift_magnitude"] for b in beta2s]
    labels = [str(b) for b in beta2s]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(range(len(beta2s)), pc1s, tick_label=labels,
                    color="steelblue", alpha=0.8)
    axes[0, 0].set_title("PC1% (row-normalized)")
    axes[0, 0].set_xlabel(r"$\beta_2$")
    axes[0, 0].set_ylabel("PC1 %")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    axes[0, 1].bar(range(len(beta2s)), k95s, tick_label=labels,
                    color="indianred", alpha=0.8)
    axes[0, 1].set_title(r"$k^*$ (95%)")
    axes[0, 1].set_xlabel(r"$\beta_2$")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 0].bar(range(len(beta2s)), k99s, tick_label=labels,
                    color="seagreen", alpha=0.8)
    axes[1, 0].set_title(r"$k^*$ (99%)")
    axes[1, 0].set_xlabel(r"$\beta_2$")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(range(len(beta2s)), drifts, tick_label=labels,
                    color="darkorange", alpha=0.8)
    axes[1, 1].set_title("Drift magnitude")
    axes[1, 1].set_xlabel(r"$\beta_2$")
    axes[1, 1].set_ylabel(r"$\|\theta(2000) - \theta(600)\|$")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(r"Backbone Geometry vs $\beta_2$", fontsize=14)
    fig.tight_layout()
    path = Path(out_dir) / "fig_backbone_summary.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_update_alignment_overlay(data, out_dir):
    """Overlay |cos(u_t, v_b)| vs t for each β2."""
    beta2s = sorted(b for b in data if "update_alignment" in data[b])
    if not beta2s:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(beta2s)))

    for b2, c in zip(beta2s, colors):
        ua = data[b2]["update_alignment"]
        steps = [r["window_start"] for r in ua]
        abs_cos = [r["abs_cos"] for r in ua]
        signed_cos = [r["signed_cos"] for r in ua]

        axes[0].plot(steps, abs_cos, "-o", color=c, markersize=5,
                     linewidth=1.5, label=rf"$\beta_2$={b2}")
        axes[1].plot(steps, signed_cos, "-o", color=c, markersize=5,
                     linewidth=1.5, label=rf"$\beta_2$={b2}")

    axes[0].set_title(r"$|\cos(u_t, v_{backbone})|$")
    axes[0].set_xlabel("Window start step")
    axes[0].set_ylabel("|cos|")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(r"$\cos(u_t, v_{backbone})$ (signed)")
    axes[1].set_xlabel("Window start step")
    axes[1].set_ylabel("cos (signed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="gray", ls="-", alpha=0.3)

    fig.suptitle(r"Update-Backbone Alignment vs $\beta_2$", fontsize=14)
    fig.tight_layout()
    path = Path(out_dir) / "fig_update_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_interference_summary(data, out_dir):
    """Bar chart of cos(g_LM, g_probe) vs β2 at each checkpoint."""
    beta2s = sorted(b for b in data if "interference" in data[b])
    if not beta2s:
        return

    # Collect all step values
    steps_set = set()
    for b2 in beta2s:
        for r in data[b2]["interference"]:
            steps_set.add(r["step"])
    steps = sorted(steps_set)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(steps), 1)
    colors = plt.cm.tab10(range(len(steps)))

    for j, step in enumerate(steps):
        means = []
        stds = []
        for b2 in beta2s:
            found = False
            for r in data[b2]["interference"]:
                if r["step"] == step:
                    means.append(r["mean_cos"])
                    stds.append(r["std_cos"])
                    found = True
                    break
            if not found:
                means.append(0)
                stds.append(0)

        x = np.arange(len(beta2s)) + j * width - 0.4 + width / 2
        ax.bar(x, means, width, yerr=stds, capsize=3, color=colors[j],
               label=f"Step {step}")

    ax.set_xticks(range(len(beta2s)))
    ax.set_xticklabels([str(b) for b in beta2s])
    ax.set_xlabel(r"$\beta_2$")
    ax.set_ylabel(r"$\cos(g_{LM}, g_{probe})$")
    ax.set_title(r"Gradient Interference vs $\beta_2$")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = Path(out_dir) / "fig_interference_vs_beta2.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_probe_behavior(data, out_dir):
    """Optional: p_ood time series for each β2."""
    beta2s = sorted(b for b in data if "metrics" in data[b])
    if not beta2s:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(beta2s)))

    for b2, c in zip(beta2s, colors):
        met = data[b2]["metrics"]
        steps = [m["step"] for m in met]
        p_ood = [m["probe_ood_acc"] for m in met]
        ax.plot(steps, p_ood, "-", color=c, linewidth=1.5,
                label=rf"$\beta_2$={b2}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("$p_{ood}$ (probe accuracy)")
    ax.set_title(r"Probe Behavior vs $\beta_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "fig_probe_behavior.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def write_summary_csv(data, out_dir):
    """Write summary/table_beta2.csv with one row per run."""
    beta2s = sorted(data.keys())
    rows = []
    for b2 in beta2s:
        entry = data[b2]
        row = {"beta2": b2}

        if "metrics" in entry:
            met = entry["metrics"]
            row["final_val"] = met[-1]["val_loss"] if met else None
            row["best_p_ood"] = max(m["probe_ood_acc"] for m in met) if met else None

        if "backbone" in entry:
            bb = entry["backbone"]
            row["PC1"] = bb["pc1_pct"]
            row["k95"] = bb["k95"]
            row["k99"] = bb["k99"]
            row["drift_600_2000"] = bb["drift_magnitude"]

        if "update_alignment" in entry:
            ua = entry["update_alignment"]
            if ua:
                row["mean_update_align"] = float(np.mean([r["abs_cos"] for r in ua]))

        rows.append(row)

    if not rows:
        return

    fieldnames = ["beta2", "final_val", "best_p_ood", "PC1", "k95", "k99",
                  "drift_600_2000", "mean_update_align"]
    path = Path(out_dir) / "table_beta2.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Saved {path}")


def compare_runs(base_dir):
    """Read all per-run analyses and produce comparison figures + CSV."""
    base_dir = Path(base_dir)
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    runs = discover_runs(base_dir)
    if not runs:
        print("  No beta2 runs found!")
        return

    print(f"\n  Found {len(runs)} runs: beta2 = {sorted(runs.keys())}")
    data = load_run_data(runs)

    plot_backbone_summary(data, summary_dir)
    plot_update_alignment_overlay(data, summary_dir)
    plot_interference_summary(data, summary_dir)
    plot_probe_behavior(data, summary_dir)
    write_summary_csv(data, summary_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Reheating summary
# ═══════════════════════════════════════════════════════════════════════════

def reheat_summary(base_dir):
    """Collect reheating metrics and produce comparison figures."""
    base_dir = Path(base_dir)
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    runs = discover_runs(base_dir)
    reheat_data = {}  # (beta2, ckpt_step) → [{"lr": ..., "metrics": ...}]

    for b2, run_dir in sorted(runs.items()):
        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("reheat_"):
                continue
            metrics_path = sub / "pilot_metrics.json"
            if not metrics_path.exists():
                continue

            m = re.search(r"ckpt(\d+)_lr([\d.eE+-]+)", sub.name)
            if not m:
                continue
            ckpt_step = int(m.group(1))
            reheat_lr = float(m.group(2))

            with open(metrics_path) as f:
                metrics = json.load(f)

            key = (b2, ckpt_step)
            if key not in reheat_data:
                reheat_data[key] = []
            reheat_data[key].append({
                "lr": reheat_lr,
                "metrics": metrics,
            })

    if not reheat_data:
        print("  No reheating data found!")
        return

    # Save consolidated JSON
    consolidated = {}
    for (b2, ckpt), lr_runs in reheat_data.items():
        key = f"b2_{b2}_ckpt_{ckpt}"
        consolidated[key] = {
            "beta2": b2,
            "ckpt_step": ckpt,
            "lr_runs": [{
                "lr": r["lr"],
                "final_p_ood": r["metrics"][-1]["probe_ood_acc"] if r["metrics"] else None,
                "final_val_loss": r["metrics"][-1]["val_loss"] if r["metrics"] else None,
                "best_p_ood": max(m["probe_ood_acc"] for m in r["metrics"]) if r["metrics"] else None,
                "n_steps": r["metrics"][-1]["step"] if r["metrics"] else 0,
            } for r in lr_runs],
        }

    with open(summary_dir / "reheat_summary.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"  Saved {summary_dir / 'reheat_summary.json'}")

    # Plot per (β2, ckpt_step): p_ood time series for each LR
    for (b2, ckpt), lr_runs in sorted(reheat_data.items()):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        colors = plt.cm.tab10(range(len(lr_runs)))

        for i, run in enumerate(sorted(lr_runs, key=lambda r: r["lr"])):
            steps = [m["step"] for m in run["metrics"]]
            p_ood = [m["probe_ood_acc"] for m in run["metrics"]]
            val_loss = [m["val_loss"] for m in run["metrics"]]

            axes[0].plot(steps, p_ood, "-o", color=colors[i], markersize=3,
                         linewidth=1.5, label=f"LR={run['lr']:.1e}")
            axes[1].plot(steps, val_loss, "-o", color=colors[i], markersize=3,
                         linewidth=1.5, label=f"LR={run['lr']:.1e}")

        axes[0].set_ylabel("$p_{ood}$ (probe accuracy)")
        axes[0].set_title(rf"Reheating $\beta_2$={b2}, from step {ckpt}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("val_loss")
        axes[1].set_xlabel("Reheating step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = summary_dir / f"fig_reheat_b2{b2}_ckpt{ckpt}.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    # Overlay figure: baseline vs low-β2 at each checkpoint
    ckpt_steps = sorted(set(ckpt for _, ckpt in reheat_data.keys()))
    for ckpt in ckpt_steps:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.tab10(range(10))
        ci = 0
        for (b2, c_step), lr_runs in sorted(reheat_data.items()):
            if c_step != ckpt:
                continue
            # Plot best LR (highest final p_ood)
            best_run = max(lr_runs,
                           key=lambda r: r["metrics"][-1]["probe_ood_acc"]
                           if r["metrics"] else 0)
            steps = [m["step"] for m in best_run["metrics"]]
            p_ood = [m["probe_ood_acc"] for m in best_run["metrics"]]
            ax.plot(steps, p_ood, "-o", color=colors[ci], markersize=3,
                    linewidth=1.5,
                    label=rf"$\beta_2$={b2} (LR={best_run['lr']:.1e})")
            ci += 1

        ax.set_xlabel("Reheating step")
        ax.set_ylabel("$p_{ood}$")
        ax.set_title(f"Reheating comparison from step {ckpt} (best LR each)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = summary_dir / f"fig_reheat_overlay_ckpt{ckpt}.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Beta2 ablation analysis")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Single run directory for per-run analysis")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing all beta2 runs")
    parser.add_argument("--compare", action="store_true",
                        help="Generate cross-run comparison figures")
    parser.add_argument("--reheat-summary", action="store_true",
                        help="Generate reheating summary figures")
    parser.add_argument("--all", action="store_true",
                        help="Run per-run analysis on all runs + compare + reheat")
    parser.add_argument("--skip-interference", action="store_true",
                        help="Skip interference metric (saves time)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Per-run analysis
    if args.run_dir:
        print(f"\n{'='*60}")
        print(f"Per-run analysis: {args.run_dir}")
        print(f"{'='*60}")
        per_run_analysis(args.run_dir, device=device,
                         skip_interference=args.skip_interference)

    # All-in-one mode: per-run on all discovered runs
    if args.all and args.base_dir:
        runs = discover_runs(args.base_dir)
        for b2, run_dir in sorted(runs.items()):
            print(f"\n{'='*60}")
            print(f"Per-run analysis: beta2={b2} ({run_dir})")
            print(f"{'='*60}")
            per_run_analysis(run_dir, device=device,
                             skip_interference=args.skip_interference)
        args.compare = True
        args.reheat_summary = True

    # Cross-run comparison
    if args.compare and args.base_dir:
        print(f"\n{'='*60}")
        print(f"Cross-run comparison: {args.base_dir}")
        print(f"{'='*60}")
        compare_runs(args.base_dir)

    # Reheating summary
    if args.reheat_summary and args.base_dir:
        print(f"\n{'='*60}")
        print(f"Reheating summary: {args.base_dir}")
        print(f"{'='*60}")
        reheat_summary(args.base_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
