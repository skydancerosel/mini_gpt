"""Per-SSD-head PR probe on Mamba-2, mirroring the transformer methodology in
analyses/pythia_410m_per_head.py and analyses/induction_heads_per_head_124m.py.

Mamba-2 structural analog of "per-head attention output":
  In attention: forward_pre_hook on out_proj/dense captures the per-head
    outputs concatenated along the head dim, reshaped to (B,T,n_head,head_dim).
  In Mamba-2: forward_pre_hook on mixer.out_proj captures the post-gate
    post-norm SSD output reshaped (B,T,num_heads,head_dim). Same structural
    role: this is the input that gets mixed by W_O.

build_induction_batch and compute_pr are copied verbatim from
analyses/induction_heads_per_head_124m.py to keep the methodology identical
across the transformer and Mamba scripts (same RNG seed 42, same batch).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import Mamba2Config, Mamba2ForCausalLM


CONFIG_PATCHES = {
    "state-spaces/mamba2-130m": dict(
        hidden_size=768, num_hidden_layers=24, num_heads=24, head_dim=64,
        n_groups=1, state_size=128, expand=2, vocab_size=50288,
    ),
    "state-spaces/mamba2-370m": dict(
        hidden_size=1024, num_hidden_layers=48, num_heads=32, head_dim=64,
        n_groups=1, state_size=128, expand=2, vocab_size=50288,
    ),
    "state-spaces/mamba2-780m": dict(
        hidden_size=1536, num_hidden_layers=48, num_heads=48, head_dim=64,
        n_groups=1, state_size=128, expand=2, vocab_size=50288,
    ),
}


def load_mamba2(model_id: str, device: str, dtype=torch.float32):
    """Load with state-dict key rename. state-spaces ckpts use
    backbone.embedding.weight (singular) but HF expects plural."""
    cfg = Mamba2Config.from_pretrained(model_id)
    for k, v in CONFIG_PATCHES[model_id].items():
        setattr(cfg, k, v)
    cfg.tie_word_embeddings = False  # avoid HF meta-tensor tying trap
    model = Mamba2ForCausalLM(cfg).to(dtype)
    sd = torch.load(hf_hub_download(model_id, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    sd["backbone.embeddings.weight"] = sd.pop("backbone.embedding.weight")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval(), cfg


# ───────────── Methodology helpers (copied from master's analyses/) ─────────────


def build_induction_batch(n_examples=2000, seq_len=256, vocab_lo=100, vocab_hi=10000, rng=None):
    """Verbatim copy of analyses/induction_heads_per_head_124m.py:build_induction_batch.

    Synthetic induction eval: each sequence has structure
        [filler] ... A, B ... [filler] ... A
    The test position is the second A. An induction head should attend back
    to the position after the first A (which contains B), making the V output
    content-dependent on B.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    tokens = np.zeros((n_examples, seq_len), dtype=np.int64)
    test_pos = np.zeros(n_examples, dtype=np.int64)
    targets = np.zeros(n_examples, dtype=np.int64)

    for i in range(n_examples):
        seq = rng.randint(vocab_lo, vocab_hi, size=seq_len).astype(np.int64)
        a, b = rng.choice(np.arange(vocab_lo, vocab_hi), size=2, replace=False)
        ab_idx = rng.randint(20, seq_len // 2)
        seq[ab_idx] = a
        seq[ab_idx + 1] = b
        for k in range(seq_len):
            if seq[k] == a and k != ab_idx:
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        for k in range(seq_len):
            if seq[k] == b and k != ab_idx + 1:
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        seq[-1] = a
        tokens[i] = seq
        test_pos[i] = seq_len - 1
        targets[i] = int(b)

    return (torch.from_numpy(tokens), torch.from_numpy(test_pos), torch.from_numpy(targets))


def compute_pr(matrix):
    """Verbatim copy of analyses/induction_heads_per_head_124m.py:compute_pr.

    PR = exp(H(p)) where p = s²/Σs², s = singular values of `matrix` (n, d).
    """
    if matrix.dim() != 2 or matrix.shape[0] == 0:
        return float("nan")
    matrix = matrix.float()
    try:
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        U, S, V = torch.linalg.svd(matrix.cpu(), full_matrices=False)
    s2 = S * S
    s2_sum = s2.sum().clamp(min=1e-12)
    p = s2 / s2_sum
    H = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(H).item())


# ───────────── Mamba-2 per-SSD-head extraction ─────────────


def per_head_pr_at_last(model, tokens, n_layer, num_heads, head_dim, device,
                         batch_size=8):
    """For each (layer, head), extract the per-SSD-head output at the last
    position, then compute PR across the batch.

    Hook: forward_pre_hook on mixer.out_proj. Captured tensor has shape
    (B, T, d_inner) where d_inner = num_heads * head_dim. Reshape and select
    last position.
    """
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    captured = {}
    out = torch.zeros(n, n_layer, num_heads, head_dim, dtype=torch.float32)

    def make_hook(L):
        def hook(module, args):
            captured[L] = args[0].detach()
        return hook

    handles = [
        model.backbone.layers[L].mixer.out_proj.register_forward_pre_hook(make_hook(L))
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
                    x = captured[L]               # (B, T, d_inner)
                    xr = x.view(B, x.shape[1], num_heads, head_dim)
                    out[start:end, L] = xr[:, last, :, :].cpu().float()
                if start % (batch_size * 5) == 0:
                    print(f"    {start+B}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()

    pr = np.zeros((n_layer, num_heads))
    for L in range(n_layer):
        for H in range(num_heads):
            pr[L, H] = compute_pr(out[:, L, H, :])
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba2-130m")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: derived from model name)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  model={args.model}")

    print("Building induction batch ...")
    rng = np.random.RandomState(args.seed)
    tokens, _, _ = build_induction_batch(n_examples=args.n_examples,
                                          seq_len=args.seq_len, rng=rng)
    print(f"  tokens={tuple(tokens.shape)}, query at position {args.seq_len - 1}")

    print(f"Loading {args.model} ...")
    t0 = time.time()
    model, cfg = load_mamba2(args.model, device)
    print(f"  loaded in {time.time()-t0:.1f}s  layers={cfg.num_hidden_layers}  "
          f"num_heads={cfg.num_heads}  head_dim={cfg.head_dim}")

    print("Running per-head PR ...")
    t0 = time.time()
    pr = per_head_pr_at_last(model, tokens,
                              n_layer=cfg.num_hidden_layers,
                              num_heads=cfg.num_heads,
                              head_dim=cfg.head_dim,
                              device=device,
                              batch_size=args.batch_size)
    print(f"  done in {time.time()-t0:.0f}s")
    print(f"  PR  min={pr.min():.2f}  max={pr.max():.2f}  mean={pr.mean():.2f}")

    result = {
        "model": args.model,
        "n_layer": cfg.num_hidden_layers,
        "num_heads": cfg.num_heads,
        "head_dim": cfg.head_dim,
        "n_examples": args.n_examples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "pr": {f"L{L}_H{H}": float(pr[L, H])
               for L in range(cfg.num_hidden_layers)
               for H in range(cfg.num_heads)},
    }
    out_path = args.out or f"mamba2_per_head_{args.model.split('/')[-1]}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
