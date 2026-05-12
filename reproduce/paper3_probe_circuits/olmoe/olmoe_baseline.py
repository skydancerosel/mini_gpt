"""Step 0: baseline OLMoE induction-acc on synthetic batch.

Sanity check before any new downloads or multi-revision compute.

Does OLMoE-1B-7B-0924 main actually do this task?
- If acc ≈ 0 (Mamba's 2.4%): documented methodology won't apply cleanly.
- If acc in Pythia-410M range (3-16%): methodology can proceed.

Same synthetic induction batch used by induction_heads_writeup.md:
2000-example version is the Pythia standard; we run 500 here for speed
since this is just sanity. RNG seed 42 for reproducibility.

No new downloads — uses cached `main` revision only.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeForCausalLM

from mamba2_per_head import build_induction_batch


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(500, 256, rng=rng)
    print(f"induction batch: {tuple(tokens.shape)}")

    print("Loading OLMoE main (from cache)...")
    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(
        "allenai/OLMoE-1B-7B-0924", revision="main", dtype=torch.float16
    ).to(device).eval()
    print(f"loaded in {time.time()-t0:.0f}s")

    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    losses, accs1, accs5, sum_logit_B = [], [], [], 0.0
    print("running baseline forward passes...")
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, 4):
            end = min(start + 4, n)
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
            if start % 80 == 0 and start > 0:
                print(f"  {end}/{n}", flush=True)

    loss = float(np.concatenate(losses).mean())
    acc1 = float(np.concatenate(accs1).mean())
    acc5 = float(np.concatenate(accs5).mean())
    logitB = sum_logit_B / n
    elapsed = time.time() - t0

    print(f"\n=== OLMoE main baseline (n={n}, seq=256, seed=42) ===")
    print(f"  loss:           {loss:.4f}")
    print(f"  top-1 acc:      {acc1:.4f}  ({acc1*100:.2f}%)")
    print(f"  top-5 acc:      {acc5:.4f}  ({acc5*100:.2f}%)")
    print(f"  mean logit_B:   {logitB:+.3f}")
    print(f"  elapsed:        {elapsed:.0f}s")

    print(f"\n=== Comparison (from induction_heads_writeup.md) ===")
    print(f"  Karpathy 124M baseline:  16.1%")
    print(f"  Pythia 410M baseline:     3.7%")
    print(f"  OLMoE 1B-7B baseline:     {acc1*100:.2f}%   ← this run")

    print(f"\n=== Verdict ===")
    if acc1 < 0.005:
        print(f"  acc < 0.5% → methodology likely won't apply cleanly. Consider task switch.")
    elif acc1 < 0.05:
        print(f"  acc in Pythia-410M range. Methodology applicable; signal will be subtle.")
    else:
        print(f"  acc clearly above noise floor. Methodology should work cleanly.")


if __name__ == "__main__":
    main()
