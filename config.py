"""
Configuration for TinyStories + Long-Range Key Retrieval Probe experiment.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class Config:
    # ── Model (GPT-small) ──────────────────────────────────────────────
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 8
    d_ff: int = 1024
    seq_len: int = 256
    dropout: float = 0.0

    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer_name: str = "roneneldan/TinyStories"

    # ── Dataset ────────────────────────────────────────────────────────
    p_probe: float = 0.10
    n_codewords: int = 512
    probe_gap_train: tuple = (5, 30)
    probe_gap_ood: tuple = (80, 200)
    n_probe_eval_in: int = 2000
    n_probe_eval_ood: int = 2000
    max_train_examples: int = 200_000  # limit training texts (2M is overkill)
    max_val_examples: int = 5_000

    # ── Optimization ───────────────────────────────────────────────────
    lr: float = 3e-4
    batch_size: int = 64         # per-device; accumulate to effective 128-256
    grad_accum_steps: int = 2    # effective batch = batch_size * grad_accum_steps
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 1e-3
    warmup_steps: int = 1500
    total_steps: int = 50_000

    # ── Logging & checkpoints ──────────────────────────────────────────
    eval_every: int = 200
    ckpt_every: int = 1000
    log_dir: str = "runs"

    # ── Geometric monitoring ───────────────────────────────────────────
    comm_k: int = 5              # median over K commutator samples
    comm_eta: float = 1e-3       # perturbation scale for commutator
    n_pca_components: int = 10   # track top-10 PCs of update trajectory

    # ── Capability control ─────────────────────────────────────────────
    control_every: int = 200
    control_batch_size: int = 64
    control_lambda: float = 0.0  # 0.0 = off, 0.3 or 1.0 = suppression
    control_k: int = 3           # PCA rank of differential subspace

    # ── Grokking metric ───────────────────────────────────────────────
    emergence_threshold: float = 0.8
    emergence_consecutive: int = 3  # must stay above threshold for N evals
    defect_baseline_window: int = 10
    defect_sigma_mult: float = 3.0
    defect_sustained_evals: int = 2

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
