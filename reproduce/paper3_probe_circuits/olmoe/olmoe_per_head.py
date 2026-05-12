"""Per-head attention PR probe on OLMoE across training revisions.

Mirrors analyses/pythia_410m_per_head.py methodology exactly:
  - Same induction batch (RNG seed 42) for apples-to-apples comparison
  - PR formula: exp(H(p)) where p = s^2 / sum(s^2) over SVD singular values
  - Hook: forward_pre_hook on self_attn.o_proj, capture input, reshape
    to (B, T, n_head, head_dim), select last position

OLMoE-1B-7B-0924 is 16L × 2048d × 16H × hd=128, 64 experts top-8.
Attention is vanilla MHA (num_key_value_heads == num_attention_heads),
so the per-head probe ports identically from Pythia.

This script handles the trajectory axis (training-time Integral analog
to Mamba's missing trajectory data). Per-expert FFN PR is a separate
script (deferred).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import OlmoeForCausalLM

# Verbatim copies from analyses/induction_heads_per_head_124m.py
# (kept inline so the script is self-contained — matches mamba2_per_head.py)
from mamba2_per_head import build_induction_batch, compute_pr


# Default log-spaced revision set for OLMoE-1B-7B-0924
# (range step5000→step1220000, ~20B→5117B tokens, every 5K steps)
DEFAULT_REVISIONS = [
    "step5000-tokens20B",
    "step10000-tokens41B",
    "step25000-tokens104B",
    "step50000-tokens209B",
    "step100000-tokens418B",
    "step200000-tokens836B",
    "step400000-tokens1672B",
    "step600000-tokens2508B",
    "step1000000-tokens4181B",
    "step1220000-tokens5117B",  # final
]


def revision_step(rev: str) -> int:
    """step5000-tokens20B → 5000."""
    return int(rev.split("step")[1].split("-")[0])


def revision_tokens_B(rev: str) -> int:
    """step5000-tokens20B → 20."""
    import re
    m = re.search(r"tokens(\d+)B", rev)
    return int(m.group(1)) if m else -1


def per_head_pr_at_last(model, tokens, n_layer, n_head, head_dim, device,
                         batch_size=4):
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    captured = {}
    out = torch.zeros(n, n_layer, n_head, head_dim, dtype=torch.float32)

    def make_hook(L):
        def hook(module, args):
            captured[L] = args[0].detach()
        return hook

    handles = [
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L))
        for L in range(n_layer)
    ]

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    x = captured[L]                              # (B, T, hidden)
                    xr = x.view(B, x.shape[1], n_head, head_dim)
                    out[start:end, L] = xr[:, last, :, :].cpu().float()
                if start % (batch_size * 10) == 0:
                    print(f"    {start+B}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()

    pr = np.zeros((n_layer, n_head))
    for L in range(n_layer):
        for H in range(n_head):
            pr[L, H] = compute_pr(out[:, L, H, :])
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--revisions", nargs="+", default=DEFAULT_REVISIONS)
    ap.add_argument("--n-examples", type=int, default=500,
                    help="smaller than Pythia's 2000 because OLMoE is 7B params")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--out", default="olmoe_per_head.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    print(f"device={device}  dtype={args.dtype}  model={args.model}")

    print("Building induction batch ...")
    rng = np.random.RandomState(args.seed)
    tokens, _, _ = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    print(f"  tokens={tuple(tokens.shape)}, query at position {args.seq_len - 1}")

    # Initialize output skeleton (head_dim/n_layer/n_head set from first load)
    out = {
        "model": args.model,
        "revisions": args.revisions,
        "n_examples": args.n_examples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "ckpt_step": [],
        "ckpt_tokens_B": [],
        "n_layer": None,
        "num_heads": None,
        "head_dim": None,
        "pr": {},
    }

    for ck_idx, rev in enumerate(args.revisions):
        step = revision_step(rev)
        toks_B = revision_tokens_B(rev)
        print(f"\n[{ck_idx+1}/{len(args.revisions)}] revision={rev} (step={step}, {toks_B}B tokens)")
        t0 = time.time()
        try:
            model = OlmoeForCausalLM.from_pretrained(args.model, revision=rev, dtype=dtype)
            model = model.to(device).eval()
        except Exception as e:
            print(f"  SKIP: load failed: {e}")
            continue

        cfg = model.config
        n_layer = cfg.num_hidden_layers
        n_head = cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_head
        if out["n_layer"] is None:
            out["n_layer"] = n_layer
            out["num_heads"] = n_head
            out["head_dim"] = head_dim
            out["pr"] = {f"L{L}_H{H}": [] for L in range(n_layer) for H in range(n_head)}
        print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head} hd={head_dim}")

        pr = per_head_pr_at_last(model, tokens, n_layer, n_head, head_dim, device,
                                  args.batch_size)
        out["ckpt_step"].append(step)
        out["ckpt_tokens_B"].append(toks_B)
        for L in range(n_layer):
            for H in range(n_head):
                out["pr"][f"L{L}_H{H}"].append(float(pr[L, H]))
        print(f"  PR  min={pr.min():.2f}  max={pr.max():.2f}  mean={pr.mean():.2f}")

        del model
        if device == "mps":
            torch.mps.empty_cache()

        # Incremental save in case of crash
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved (incremental) → {args.out}")

    print(f"\ndone — wrote {args.out}")


if __name__ == "__main__":
    main()
