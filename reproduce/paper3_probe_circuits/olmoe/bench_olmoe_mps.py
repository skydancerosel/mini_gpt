"""Benchmark OLMoE forward pass on MPS. Same role as bench_mamba2_mps.py.

OLMoE-1B-7B has 16 layers, 2048 hidden, 16 heads (head_dim=128), 64 experts,
top-8 routing. Stored as bf16 (~14 GB). MoE inference on MPS has not been
heavily exercised — this benchmark just confirms forward pass works and
gives a per-forward-pass time so we can plan the probe budget.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import OlmoeForCausalLM


def benchmark(model_id: str, revision: str, seq_len: int,
              n_warmup: int, n_runs: int, dtype: torch.dtype):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== {model_id}@{revision} | seq_len={seq_len} | dtype={dtype} | device={device} ===")

    t0 = time.time()
    model = OlmoeForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model = model.to(device).eval()
    load_s = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {load_s:.1f}s | params={n_params/1e9:.2f}B")

    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq_len), device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
    sync()

    times = []
    with torch.no_grad():
        for i in range(n_runs):
            sync()
            t0 = time.time()
            _ = model(input_ids)
            sync()
            dt = time.time() - t0
            times.append(dt)
            print(f"  run {i+1}: {dt*1000:.0f} ms")

    median_ms = statistics.median(times) * 1000
    print(f"  -> median {median_ms:.0f} ms | {seq_len/median_ms*1000:.0f} tok/s")
    return median_ms


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", default="main")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    args = p.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    benchmark(args.model, args.revision, args.seq_len, args.warmup, args.runs,
              dtype_map[args.dtype])
