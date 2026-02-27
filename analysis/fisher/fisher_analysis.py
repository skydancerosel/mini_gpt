import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
#!/usr/bin/env python3
"""
Empirical Fisher Information Matrix analysis at oscillation peak/trough checkpoints.

Computes top-k eigenvalues of the empirical Fisher F = (1/M) G G^T using the
Gram matrix trick: eigenvalues of F come from the smaller (1/M) G^T G matrix,
where G is (M, D) with M = number of gradient samples and D = trunk param count.

Trunk parameters (~25M) are defined by TRUNK_PATTERN from attractor_analysis.py:
  blocks.*.attn.qkv.weight, attn.out_proj.weight, mlp.w_up.weight, mlp.w_down.weight

Gradients use the combined training loss:  L = L_LM + lambda_probe * L_probe
Each gradient sample is one mini-batch gradient (batch_size=64 by default).
Gradients are stored on CPU to save device memory.

Outputs:
  - fisher_eigenvalues.json: per-checkpoint eigenvalues, trace, timing
  - fig_fisher_spectrum.png: eigenvalue spectra + lambda_1 bar chart

Usage:
  python fisher_analysis.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/
  python fisher_analysis.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ --checkpoints 2800,5000,6400,2000,5400,6800
  python fisher_analysis.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42/ --top-k 50 --n-batches 64
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from attractor_analysis import load_checkpoint, load_metrics, TRUNK_PATTERN


# ═══════════════════════════════════════════════════════════════════════════
# Gradient collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_trunk_gradient(model):
    """Extract trunk-only gradients as a flat float32 vector on CPU.

    Trunk params are those matching TRUNK_PATTERN (attention + MLP weights
    in transformer blocks). Order is deterministic via sorted parameter names.

    Returns:
        1-D float32 tensor on CPU with ~25M elements.
    """
    parts = []
    for name, param in model.named_parameters():
        if TRUNK_PATTERN.match(name) and param.grad is not None:
            parts.append(param.grad.cpu().reshape(-1).float())
    if not parts:
        raise ValueError("No trunk gradients found — check model and TRUNK_PATTERN")
    return torch.cat(parts)


def count_trunk_params(model):
    """Count the number of trunk-only parameters (for logging)."""
    total = 0
    for name, param in model.named_parameters():
        if TRUNK_PATTERN.match(name):
            total += param.numel()
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Fisher eigenvalue computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_fisher_eigenvalues(model, dataloader, device, n_batches, lambda_probe, top_k):
    """Compute top-k eigenvalues of the empirical Fisher at a checkpoint.

    Uses the Gram matrix trick for efficiency:
      F = (1/M) G G^T  has eigenvalues from  (1/M) G^T G
    where G is (M, D) — M gradient samples, D trunk param count.

    Since M << D (~32 vs ~25M), the (M, M) Gram matrix is tiny.

    Args:
        model: GPTModel loaded with checkpoint weights, on device.
        dataloader: DataLoader yielding (input_ids, targets, probe_mask).
        device: torch device for forward/backward passes.
        n_batches: number of mini-batch gradient samples M.
        lambda_probe: weight for probe loss in combined objective.
        top_k: number of top eigenvalues to return.

    Returns:
        dict with keys: eigenvalues (list), trace (float), n_batches (int),
        n_params (int), top_k (int).
    """
    model.eval()  # BatchNorm/Dropout in eval mode, but we still compute gradients
    ce = nn.CrossEntropyLoss(reduction='none')
    data_iter = iter(dataloader)
    gradients = []

    for i in range(n_batches):
        # Get next batch, cycling if dataloader is exhausted
        try:
            input_ids, targets, probe_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, targets, probe_mask = next(data_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)
        probe_mask = probe_mask.to(device)

        # Forward pass
        model.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)

        # Combined loss: L = L_LM + lambda_probe * L_probe
        loss_flat = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss_all = loss_flat.view(targets.shape)

        pmask = probe_mask.bool()
        lm_mask = ~pmask & (targets != -100)

        lm_loss = loss_all[lm_mask].mean() if lm_mask.any() else torch.tensor(0.0, device=device)
        p_loss = loss_all[pmask].mean() if pmask.any() else torch.tensor(0.0, device=device)
        loss = lm_loss + lambda_probe * p_loss

        # Backward pass
        loss.backward()

        # Collect trunk gradient on CPU
        g = collect_trunk_gradient(model)
        gradients.append(g)

        if (i + 1) % 8 == 0 or (i + 1) == n_batches:
            print(f"      Batch {i+1}/{n_batches}: loss={loss.item():.4f} "
                  f"(lm={lm_loss.item():.4f}, probe={p_loss.item():.4f})")

    # Stack gradients: G is (M, D) on CPU
    M = len(gradients)
    D = gradients[0].numel()
    G = torch.stack(gradients)  # (M, D)
    assert G.shape == (M, D), f"Expected ({M}, {D}), got {G.shape}"

    # Gram matrix trick: eigenvalues of (1/M) G G^T from (1/M) G^T G
    # G^T G is (M, M) — much smaller than (D, D)
    gram = (G @ G.T) / M  # (M, M)

    # Eigenvalue decomposition (real symmetric → eigvalsh)
    eigvals = torch.linalg.eigvalsh(gram).flip(0)  # descending order
    eigvals_list = eigvals[:top_k].tolist()
    trace = eigvals.sum().item()

    return {
        "eigenvalues": eigvals_list,
        "trace": trace,
        "n_batches": M,
        "n_params": D,
        "top_k": min(top_k, M),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_fisher_spectrum(results, out_dir):
    """Plot Fisher eigenvalue spectra and lambda_1 comparison.

    Panel 1: All eigenvalues on log scale (semilogy), peaks in red, troughs in green.
    Panel 2: Bar chart comparing top eigenvalue (lambda_1) at peaks vs troughs.

    Args:
        results: list of dicts with keys: step, label, eigenvalues, trace.
        out_dir: Path to output directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: Full eigenvalue spectra ──────────────────────────────────
    ax1 = axes[0]
    for r in results:
        color = "red" if r["label"] == "peak" else "green"
        linestyle = "-" if r["label"] == "peak" else "--"
        marker = "o" if r["label"] == "peak" else "s"
        ax1.semilogy(
            range(1, len(r["eigenvalues"]) + 1),
            r["eigenvalues"],
            color=color, linestyle=linestyle, marker=marker,
            markersize=4, linewidth=1.2, alpha=0.8,
            label=f"step {r['step']} ({r['label']})"
        )
    ax1.set_xlabel("Eigenvalue rank")
    ax1.set_ylabel("Eigenvalue (log scale)")
    ax1.set_title("Fisher Eigenvalue Spectrum")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: lambda_1 bar chart ───────────────────────────────────────
    ax2 = axes[1]
    peak_results = [r for r in results if r["label"] == "peak"]
    trough_results = [r for r in results if r["label"] == "trough"]

    labels = []
    lambda1_vals = []
    colors = []

    for r in results:
        labels.append(f"{r['step']}\n({r['label'][0].upper()})")
        lambda1_vals.append(r["eigenvalues"][0])
        colors.append("red" if r["label"] == "peak" else "green")

    x_pos = range(len(labels))
    bars = ax2.bar(x_pos, lambda1_vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel(r"$\lambda_1$ (top eigenvalue)")
    ax2.set_title(r"Top Fisher Eigenvalue $\lambda_1$ at Peaks vs Troughs")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, lambda1_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2e}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = Path(out_dir) / "fig_fisher_spectrum.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Empirical Fisher eigenvalue analysis at oscillation peak/trough checkpoints"
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to pilot run directory with checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (for dataset construction; codeword seed is always 42)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to oscillation_manifest.json (default: <run-dir>/oscillation_manifest.json)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint steps (overrides manifest)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top eigenvalues to compute (default: 20)")
    parser.add_argument("--n-batches", type=int, default=32,
                        help="Number of gradient samples M (default: 32)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size for gradient samples (default: 64)")
    parser.add_argument("--lambda-probe", type=float, default=2.0,
                        help="Probe loss weight in combined objective (default: 2.0)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    device = get_device()
    print(f"Fisher Analysis")
    print(f"  run_dir:      {run_dir}")
    print(f"  device:       {device}")
    print(f"  top_k:        {args.top_k}")
    print(f"  n_batches:    {args.n_batches} (M)")
    print(f"  batch_size:   {args.batch_size}")
    print(f"  lambda_probe: {args.lambda_probe}")

    # ── Resolve checkpoint steps ──────────────────────────────────────────
    if args.checkpoints is not None:
        # Explicit checkpoint list from CLI
        all_steps = [int(s.strip()) for s in args.checkpoints.split(",")]
        # We don't know peak/trough labels from CLI alone; try manifest
        manifest_path = Path(args.manifest) if args.manifest else run_dir / "oscillation_manifest.json"
        peak_steps = set()
        trough_steps = set()
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            peak_steps = set(manifest["peaks"])
            trough_steps = set(manifest["troughs"])
        # Label each step
        checkpoint_info = []
        for step in sorted(all_steps):
            if step in peak_steps:
                label = "peak"
            elif step in trough_steps:
                label = "trough"
            else:
                label = "unknown"
            checkpoint_info.append({"step": step, "label": label})
    else:
        # Load from manifest
        manifest_path = Path(args.manifest) if args.manifest else run_dir / "oscillation_manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found at {manifest_path}", file=__import__('sys').stderr)
            print("Run detect_oscillations.py first, or pass --checkpoints explicitly.",
                  file=__import__('sys').stderr)
            __import__('sys').exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        peak_steps = set(manifest["peaks"])
        trough_steps = set(manifest["troughs"])

        # Use representative steps if available (noise-calibrated subset)
        rep = manifest.get("representative", {})
        if rep.get("basin_depth_steps"):
            use_steps = set(rep["basin_depth_steps"])
            # Also include paired troughs from priority_pairs
            for pp in rep.get("priority_pairs", []):
                use_steps.add(pp["peak"])
                use_steps.add(pp["trough"])
            print(f"  Using representative subset: {sorted(use_steps)}")
        else:
            use_steps = peak_steps | trough_steps
            print(f"  Using all peaks+troughs ({len(use_steps)} checkpoints)")

        # Combine and sort chronologically
        checkpoint_info = []
        for step in sorted(use_steps):
            if step in peak_steps:
                label = "peak"
            elif step in trough_steps:
                label = "trough"
            else:
                label = "late_lm"
            checkpoint_info.append({"step": step, "label": label})
        checkpoint_info.sort(key=lambda x: x["step"])

    print(f"\n  Checkpoints ({len(checkpoint_info)}):")
    for ci in checkpoint_info:
        print(f"    step {ci['step']:>6d}  ({ci['label']})")

    # ── Build datasets ────────────────────────────────────────────────────
    # Model architecture: 8L, d=512, h=16, ff=2048 (as per project spec)
    # Codeword seed is ALWAYS 42 regardless of training seed
    cfg = Config(
        seed=42,  # codeword seed is always 42
        p_probe=0.10,
        batch_size=args.batch_size,
        n_layer=8, d_model=512, n_head=16, d_ff=2048,
    )

    # Use saved codewords if available
    codewords_path = run_dir / "codewords.json"
    cw_path_str = str(codewords_path) if codewords_path.exists() else None

    print(f"\nBuilding datasets...")
    data = build_datasets(cfg, codewords_path=cw_path_str)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)

    train_loader = DataLoader(
        data["train_dataset"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    print(f"  vocab_size: {vocab_size}")

    # ── Process each checkpoint ───────────────────────────────────────────
    results = []
    total_t0 = time.time()

    for ci in checkpoint_info:
        step = ci["step"]
        label = ci["label"]
        print(f"\n{'='*60}")
        print(f"  Step {step} ({label})")
        print(f"{'='*60}")

        t0 = time.time()

        # Load checkpoint
        print(f"    Loading checkpoint ckpt_{step:06d}.pt ...")
        ckpt = load_checkpoint(run_dir, step, device=device)

        # Build model
        model = GPTModel(
            vocab_size=vocab_size,
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            d_ff=cfg.d_ff,
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt

        # Log trunk param count on first checkpoint
        if not results:
            n_trunk = count_trunk_params(model)
            print(f"    Trunk params (D): {n_trunk:,}")

        # Compute Fisher eigenvalues
        print(f"    Collecting {args.n_batches} gradient samples ...")
        fisher_result = compute_fisher_eigenvalues(
            model=model,
            dataloader=train_loader,
            device=device,
            n_batches=args.n_batches,
            lambda_probe=args.lambda_probe,
            top_k=args.top_k,
        )

        elapsed = time.time() - t0

        # Store result
        result_entry = {
            "step": step,
            "label": label,
            "eigenvalues": fisher_result["eigenvalues"],
            "trace": fisher_result["trace"],
            "n_batches": fisher_result["n_batches"],
            "n_params": fisher_result["n_params"],
            "top_k": fisher_result["top_k"],
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(result_entry)

        # Print summary for this checkpoint
        print(f"    lambda_1 = {fisher_result['eigenvalues'][0]:.6e}")
        print(f"    trace    = {fisher_result['trace']:.6e}")
        print(f"    time     = {elapsed:.1f}s")

        # Free model memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    total_elapsed = time.time() - total_t0

    # ── Summary statistics ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")

    peak_results = [r for r in results if r["label"] == "peak"]
    trough_results = [r for r in results if r["label"] == "trough"]

    if peak_results and trough_results:
        mean_lambda1_peak = sum(r["eigenvalues"][0] for r in peak_results) / len(peak_results)
        mean_lambda1_trough = sum(r["eigenvalues"][0] for r in trough_results) / len(trough_results)
        mean_trace_peak = sum(r["trace"] for r in peak_results) / len(peak_results)
        mean_trace_trough = sum(r["trace"] for r in trough_results) / len(trough_results)

        ratio_lambda1 = mean_lambda1_peak / mean_lambda1_trough if mean_lambda1_trough > 0 else float('inf')
        ratio_trace = mean_trace_peak / mean_trace_trough if mean_trace_trough > 0 else float('inf')

        print(f"\n  Peaks ({len(peak_results)} checkpoints):")
        print(f"    mean lambda_1 = {mean_lambda1_peak:.6e}")
        print(f"    mean trace    = {mean_trace_peak:.6e}")
        for r in peak_results:
            print(f"      step {r['step']}: lambda_1={r['eigenvalues'][0]:.6e}, trace={r['trace']:.6e}")

        print(f"\n  Troughs ({len(trough_results)} checkpoints):")
        print(f"    mean lambda_1 = {mean_lambda1_trough:.6e}")
        print(f"    mean trace    = {mean_trace_trough:.6e}")
        for r in trough_results:
            print(f"      step {r['step']}: lambda_1={r['eigenvalues'][0]:.6e}, trace={r['trace']:.6e}")

        print(f"\n  Ratios (peak / trough):")
        print(f"    lambda_1 ratio = {ratio_lambda1:.3f}")
        print(f"    trace ratio    = {ratio_trace:.3f}")
    else:
        print("  (Insufficient peak/trough data for comparison)")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "config": {
            "run_dir": str(run_dir),
            "seed": args.seed,
            "top_k": args.top_k,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "lambda_probe": args.lambda_probe,
            "device": device,
        },
        "results": results,
        "summary": {},
        "total_elapsed_sec": round(total_elapsed, 1),
    }

    if peak_results and trough_results:
        output["summary"] = {
            "n_peaks": len(peak_results),
            "n_troughs": len(trough_results),
            "mean_lambda1_peak": mean_lambda1_peak,
            "mean_lambda1_trough": mean_lambda1_trough,
            "mean_trace_peak": mean_trace_peak,
            "mean_trace_trough": mean_trace_trough,
            "lambda1_ratio_peak_over_trough": ratio_lambda1,
            "trace_ratio_peak_over_trough": ratio_trace,
        }

    json_path = out_dir / "fisher_eigenvalues.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nPlotting Fisher spectrum...")
    plot_fisher_spectrum(results, out_dir)

    print(f"\nDone! All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
