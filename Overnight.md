# Claude Code Overnight Project: β2 Ablation + Minimal Reheating/Interference Tests

Owner: Yongzhong Xu  
Goal: Produce a clean, mechanistic follow-up to the “backbone” paper by testing whether **AdamW’s second-moment normalization strength (β2)** controls (i) backbone non-degeneracy and (ii) transverse dynamics / reheating accessibility.

This is designed to run overnight with minimal code changes and crisp outputs.

---

## 0) Summary of What We’ll Run

### A. Main experiment (β2 ablation; 4k steps each)
Run the same pilot training with identical settings except **AdamW betas**:

- β2 ∈ {0.99, 0.95 (baseline), 0.90, 0.80, 0.0}
- Keep β1 fixed at 0.9 (unless code requires otherwise).
- Keep everything else identical.

Expected: as β2 decreases (weaker variance normalization), trajectory may become more colinear (PC1 ↑), and transverse structure may weaken.

### B. Minimal reheating test (only on 1–2 settings)
Pick 1–2 checkpoints (late and mid) and run reheating with the same protocol as in the paper for:
- baseline β2=0.95
- one low-β2 condition (e.g., β2=0.80 or β2=0.0)

Goal: does reheating still re-excite transverse modes when second-moment normalization is weakened?

### C. Minimal “task interference” / competition test
Only if cheap to implement: evaluate “competition strength” between LM and probe gradients via cosine or correlation (already have scripts). Compare baseline vs low-β2.

Goal: does reduced normalization change effective interference (and thus switching / oscillations)?

---

## 1) Hard Constraints (Do Not Change These)

Use the exact same baseline hyperparameters as the main paper:

- Optimizer: AdamW
- peak lr: 1e-3
- betas: (0.9, β2)  ← only β2 changes
- eps: 1e-8
- weight_decay: 0.5 (decoupled, exclude bias/LN as in current code)
- warmup: 1500 steps (linear)
- schedule: cosine decay with min_lr_ratio = 0.1
- grad_clip: 1.0
- total_steps: 4000  (not 10k; we only need early/mid geometry)
- seed: 42 (single seed, keep scope tight)
- dataset/probe generation: same as current pilot pipeline (do not regenerate with different codeword seeds)

If any run diverges (NaNs), stop that run and record failure.

---

## 2) Implementation Tasks

### 2.1 Add β2 as a CLI arg (pilot.py)
Add flag:
- `--beta2` (float, default 0.95)

Wire into AdamW:
- `torch.optim.AdamW(..., betas=(0.9, beta2), eps=1e-8, ...)`

Do NOT change β1.

### 2.2 Run directory naming
Use a consistent naming convention:

`runs/beta2_ablation/pilot_wd0.5_lr0.001_lp{LAMBDA}_b2{BETA2}_s42/`

If lambda-probe is already scheduled in code, keep it identical to baseline.
If you have a two-phase λ schedule (e.g., 2 -> 4), keep it identical.

### 2.3 Checkpoint cadence
To support PCA windows without “too little data”, save checkpoints:
- every 50 steps from 600 → 2000
- every 100 steps from 2000 → 4000

If disk is a concern, save trunk-only params (same subset used in backbone scripts).

Also save pilot_metrics.json as usual.

---

## 3) Runs to Execute (Overnight)

### 3.1 Train commands (example)
For each β2 in [0.99, 0.95, 0.90, 0.80, 0.0]:

```bash
python pilot.py \
  --seed 42 \
  --wd 0.5 \
  --lr 0.001 \
  --beta2 <BETA2> \
  --steps 4000 \
  --warmup 1500 \
  --min_lr_ratio 0.1 \
  --grad_clip 1.0 \
  --eval_every 200 \
  --save_ckpt_every 50 \
  --save_ckpt_dense_from 600 \
  --save_ckpt_dense_to 2000 \
  --save_ckpt_sparse_every 100 \
  --save_ckpt_sparse_from 2000