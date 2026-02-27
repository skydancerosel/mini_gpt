# Experiment Spec: Grokking-like Capability Emergence + Geometric Monitoring (TinyStories)

## Goals
1) Demonstrate *non-toy* grokking-like delayed capability emergence.
2) Test whether a weight-space geometric signal (commutator defect + low-d update subspace) predicts emergence.
3) Demonstrate capability control by suppressing task/probe-differential update subspaces.

We will run experiment:
A) TinyStories language modeling with an embedded long-range retrieval probe (primary).

Reference:
\dyck 
most of the probes has been done for dyck_1 dataset there, you can look up the code there and build on it. 

---

## A) TinyStories (Language) — "Long-Range Key Retrieval" Probe

### Dataset
- Base: TinyStories (standard train split).
- Create a small synthetic "probe-injected" subset by modifying a fraction of training examples:
  - For p_probe in {0.02, 0.05} (start with 0.05 if compute allows).
  - Insert a KEY sentence early and a QUERY late.
- Format (example):
  - Early: "The secret code is ZORPLE."
  - Later: "What is the secret code? ZORPLE"
- Code token should be a single token if possible:
  - Use a list of 512–2048 codewords (uppercase strings) chosen to be single-token under the tokenizer.
  - If not possible, allow 2 tokens but evaluate exact-string match.

### Train vs OOD Test
- Training probe gaps: distance(query_pos - key_pos) in [5, 30] tokens.
- OOD probe gaps: distance in [80, 200] tokens.
- Create fixed probe evaluation sets:
  - probe_eval_in  = 2000 examples, gaps in [5,30]
  - probe_eval_ood = 2000 examples, gaps in [80,200]
- Also evaluate standard LM validation set perplexity/loss.

### Model
- GPT-small:
  - n_layer = 4
  - d_model = 256
  - n_head  = 8
  - d_ff    = 1024 (or 4*d_model)
  - seq_len = 256
  - vocab/tokenizer = same as TinyStories baseline used by repo
- Use tied embeddings if existing, otherwise standard.

### Optimization
- AdamW, betas=(0.9,0.95), eps=1e-8
- lr = 3e-4 (cosine decay with warmup 500 steps)
- batch_size = as large as stable on MPS (target effective 128–256 sequences)
- grad_clip = 1.0
- Weight decay sweep (run all):
  - wd ∈ {0.0, 1e-4, 3e-4, 1e-3, 3e-3}
- Total steps:
  - Start with 50k steps; if no delayed transition appears, extend best-looking WD to 150k.

### Logging cadence
- Evaluate every eval_every = 200 steps:
  - train loss (moving avg)
  - val loss
  - probe_in accuracy (exact match)
  - probe_ood accuracy (exact match)
- Checkpoint every ckpt_every = 1000 steps:
  - save model weights
  - save optimizer state (optional)
- Save run metadata: seed, wd, lr schedule, probe fraction, tokenizer info.

### Grokking-like metric
- Define "capability emergence time" for OOD probe:
  - Let A_ood(t) be probe_ood accuracy at eval checkpoints.
  - emergence_time = first checkpoint where A_ood(t) >= 0.8 AND remains >=0.8 for 3 consecutive evals.
- Also record:
  - t_trainflat = first checkpoint where val loss slope (EMA) < small threshold for N evals.

### Geometric monitoring signals
At each checkpoint (ckpt_every):
1) Save update vectors:
  - store Δθ between successive checkpoints (flattened or sketched).
2) Compute commutator defect on update sequence:
  - Use your existing commutator defect definition for weight-space updates.
  - Output scalar defect(t).
3) Low-d trajectory structure:
  - Maintain PCA over recent Δθ (or over all Δθ so far) and log explained variance of top PCs (PC1..PC10) and k* for 95%/99%.
4) Early warning test:
  - Measure lead_time = emergence_time - defect_onset_time
  - defect_onset_time defined as first ckpt where defect exceeds baseline + 3σ and stays elevated for 2 ckpts.

### Optional: Capability control (subspace suppression)
Periodically (every 200 steps), compute two probe gradients on small batches (size 64):
- g_lm    = gradient on normal LM batch
- g_probe = gradient on probe-injected batch (only probe examples)
Construct differential direction:
- v = normalize(g_probe - g_lm)
Maintain a small basis S (top-k PCA over recent v; k=1..5).
Intervention on trunk update:
- Δθ ← Δθ - λ * P_S(Δθ), applied only to transformer blocks (exclude embeddings/head if desired)
Test λ ∈ {0.0, 0.3, 1.0}.
Goal: suppress/delay probe_ood emergence with minimal change in val loss/perplexity.


## Success Criteria
TinyStories:
- For some wd, probe_ood accuracy remains low for long time while LM loss stabilizes, then jumps sharply.
- Defect onset precedes probe_ood jump with positive lead time.
- Subspace suppression delays probe_ood emergence with small impact on val loss.

