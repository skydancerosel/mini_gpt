"""Benchmark Mamba-2 forward pass on MPS without mamba_ssm (slow fallback path).

Goal: confirm Mamba-2 130M / 370M / 780M are realistic for probe-circuit work.
Probe work is inference-only on val tokens; we just need a workable forward pass.
"""
import argparse
import statistics
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import Mamba2Config, Mamba2ForCausalLM


# Per-checkpoint config corrections — state-spaces uploaded HF configs with
# wrong hidden_size/num_hidden_layers/num_heads/n_groups/state_size. Values
# here are reverse-engineered from the actual state_dict shapes.
CONFIG_PATCHES = {
    # All verified against actual state_dict shapes.
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


def benchmark(model_id: str, seq_len: int, n_warmup: int, n_runs: int, dtype: torch.dtype):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== {model_id} | seq_len={seq_len} | dtype={dtype} | device={device} ===")

    cfg = Mamba2Config.from_pretrained(model_id)
    if model_id in CONFIG_PATCHES:
        for k, v in CONFIG_PATCHES[model_id].items():
            setattr(cfg, k, v)

    t0 = time.time()
    # Manual load: state-spaces ckpt uses `backbone.embedding.weight` (singular)
    # but HF expects `backbone.embeddings.weight`. from_pretrained leaves both
    # embedding and lm_head as meta tensors. Load state_dict ourselves with the
    # key rename, then construct the model fresh and copy weights in.
    cfg.tie_word_embeddings = False  # avoid HF auto-tie meta-tensor trap
    model = Mamba2ForCausalLM(cfg).to(dtype)
    ckpt_path = hf_hub_download(model_id, "pytorch_model.bin")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd["backbone.embeddings.weight"] = sd.pop("backbone.embedding.weight")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    load_s = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {load_s:.1f}s | params={n_params/1e6:.1f}M")

    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq_len), device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
    sync()

    # Timed
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
    p.add_argument("--model", default="state-spaces/mamba2-130m")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = p.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    benchmark(args.model, args.seq_len, args.warmup, args.runs, dtype_map[args.dtype])
