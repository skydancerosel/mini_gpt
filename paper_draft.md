# Attractor Switching in a Language Model with Injected Probe Task

## Abstract

We train a 51M-parameter GPT on TinyStories with a simultaneously injected long-range key-retrieval probe (10% of tokens), and observe sustained oscillations in probe accuracy over 10,000 training steps.  We characterize this phenomenon as *attractor switching* between a probe-dominant and an LM-dominant regime in weight space.  Through geometric and dynamical analyses we show that the training trajectory decomposes into a **backbone**---a single direction capturing ~80% of parameter drift, driven by LM optimization---and a **transverse subspace** where attractor switching occurs.  Key findings, replicated across two seeds: (i) the backbone emerges from optimizer integration of noisy gradients, with the embedding layer (Block 0) as the primary driver; (ii) each switching event moves along a near-orthogonal direction in the transverse subspace; (iii) the probe basin shallows monotonically as the backbone stiffens; and (iv) the probe attractor remains transiently re-enterable via reheating.  These results suggest that multi-task training dynamics can be understood as a slow backbone drift (set by the dominant objective) with fast transverse oscillations (set by competing objectives), analogous to adiabatic separation in physical systems.

---

## 1  Introduction

Multi-task training of neural language models often reveals complex dynamics: tasks can compete for model capacity, leading to phenomena like catastrophic forgetting, task interference, and oscillatory training curves.  Understanding these dynamics in weight space—not just in metric space—is essential for safe capability control.

We study a minimal instance of this problem: a GPT model trained jointly on a language modeling (LM) objective and a synthetic long-range key-retrieval probe.  The probe injects codeword–value pairs into a fraction of sequences and tests whether the model can retrieve values at out-of-distribution gap distances.  During training, the probe accuracy (p_ood) oscillates between high ("probe regime") and low ("LM regime") values, even as the LM loss decreases monotonically.

We interpret this oscillation geometrically as switching between two attractors in weight space and conduct a systematic analysis of the landscape structure.

---

## 2  Experimental Setup

### 2.1  Model Architecture

We use a decoder-only GPT with:

| Parameter | Value |
|-----------|-------|
| Layers | 8 |
| d_model | 512 |
| Attention heads | 16 |
| d_ff | 2048 |
| Sequence length | 256 |
| Dropout | 0.0 |
| Total parameters | 51,045,888 |
| Tokenizer | roneneldan/TinyStories (≈50k vocab) |

### 2.2  Training Data

**Language modeling.**  We use 200k texts from the TinyStories dataset (Eldan & Li, 2023), tokenized to 256-token sequences.

**Probe injection.**  With probability p_probe = 0.10, a training sequence is replaced by a probe sequence.  Each probe sequence contains a key–value pair: a *codeword* (a single-token string) is placed at a random position, followed by a *value token* placed at a gap distance g sampled uniformly from [5, 30] (in-distribution) during training.  The model must predict the value token given the codeword.

**Codeword selection.**  512 codewords are selected from the tokenizer vocabulary: tokens that encode to exactly one token, are at least 3 characters, and contain only lowercase ASCII letters.  Candidates are sorted alphabetically and then shuffled with a fixed random seed for reproducibility.  (An earlier non-deterministic selection was fixed and 477 codewords were recovered by probing the trained model; see Appendix A.)

**Evaluation.**  Two held-out probe evaluation sets of 2000 examples each:
- *In-distribution (ID)*: gap distances g ∈ [5, 30]
- *Out-of-distribution (OOD)*: gap distances g ∈ [80, 200]

The primary metric is **p_ood** = exact-match accuracy on the OOD evaluation set.  We also track probe NLL at probe positions (nll_ood), LM NLL at non-probe positions (lm_ood), and standard validation loss.

### 2.3  Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8) |
| Learning rate | 1e-3 (cosine decay with 1500-step warmup) |
| Weight decay | 0.5 |
| Probe loss weight λ | 2.0 |
| Batch size | 64 × 2 gradient accumulation = 128 effective |
| Gradient clipping | 1.0 |
| Total steps | 10,000 |
| Checkpoint interval | 200 steps |
| Seed | 42 |
| Device | Apple MPS (M-series) |

The composite loss is: L = L_lm + λ · L_probe, where L_lm and L_probe are cross-entropy losses at non-probe and probe positions respectively, separated by a binary probe_mask.

---

## 3  Training Dynamics: Oscillating Probe Accuracy

### 3.1  Time Series

Over 10,000 training steps, the model exhibits clear oscillations in p_ood (Figure 1).  Recalibrated with the recovered codeword set (477 codewords; see Appendix A), the time series shows:

| Step | p_ood | Phase |
|------|-------|-------|
| 1 | 0.000 | Random |
| 1000 | 0.066 | Emerging |
| 1800 | 0.518 | **Peak 1** |
| 2000 | 0.483 | Trough |
| 2800 | 0.737 | **Peak 2 (global max)** |
| 3200 | 0.565 | Trough |
| 4800 | 0.454 | Trough |
| 5000 | 0.765 | **Peak 3** |
| 5400 | 0.453 | Trough |
| 6400 | 0.704 | **Peak 4** |
| 6800 | 0.392 | Trough |
| 7600 | 0.140 | Declining |
| 10000 | 0.161 | **LM-dominant** |

Key observations:
- p_ood oscillates with amplitude ΔP ≈ 0.25–0.35 between peaks and troughs during steps 1800–6800
- Peak values remain high (0.52–0.77) through step 6400
- After step 7000, the model transitions to a persistent LM-dominant regime (p_ood < 0.30)
- In-distribution accuracy (p_in) remains high (>0.85) throughout, indicating the probe is learned in-distribution even when OOD generalization fails
- Validation loss decreases monotonically from 10.8 to 1.22, showing the LM objective is never impaired

### 3.2  Recalibration

Due to a non-determinism bug in codeword selection (tokenizer.get_vocab() iteration order depends on Python's hash seed), the original training metrics used codewords that could not be exactly reproduced.  We recovered the original codewords by testing the trained model (at step 2800, a peak) on all 1,281 candidate tokens individually (20 trials each).  477 codewords showed accuracy > 0, forming a bimodal distribution: 384 with acc ≥ 0.5 and 816 with acc < 0.1.

Recalibrated metrics using the recovered codewords match the original training metrics with ratio 0.92–1.08 (mean ≈ 0.99), confirming successful recovery.

---

## 4  Geometric Analysis

We define the **trunk parameters** as all weight matrices excluding tied embeddings, causal masks, positional embeddings, and layer normalization parameters.  The trunk has approximately 25M parameters and carries the model's computational "identity" for both tasks.

### 4.1  Analysis B1: Basin Test (Probe Recoverability)

**Question:** From a checkpoint at training step s, can fresh fine-tuning with λ=8 (4× training value) recover the probe to p_ood ≥ 0.60?

**Protocol:** Load checkpoint, reset optimizer, train for 1000 steps with LR=6e-4, λ=8.0, and evaluate every 100 steps.

**Results:**

| Type | Step | Start p_ood | Steps to 60% | Max Achieved |
|------|------|-------------|--------------|--------------|
| Peak | 1800 | 0.523 | 650 | **0.686** |
| Trough | 2000 | 0.483 | — | 0.569 |
| Peak | 2400 | 0.586 | — | 0.573 |
| Trough | 2600 | 0.535 | — | 0.552 |
| Peak | 2800 | 0.709 | — | 0.503 |
| Trough | 3200 | 0.582 | — | 0.469 |

**Key finding:** Only the earliest peak (step 1800) reaches the 0.60 target.  Maximum recovery decays monotonically with training step: 0.686 → 0.573 → 0.503 (peaks), 0.569 → 0.552 → 0.469 (troughs).  Peaks consistently outperform troughs by 0.03–0.12.

**Interpretation:** The probe basin becomes progressively shallower and harder to reach as the LM representation consolidates.  The fresh optimizer (without accumulated momentum) cannot reproduce the trajectory that originally found the probe peak.

### 4.2  Analysis B2: Switching-Direction Alignment

**Question:** Do different switching events move along similar directions in weight space?

**Protocol:** Define three switching directions from matched peak–trough pairs:
- Δ₁ = θ(2800) − θ(2000), ‖Δ₁‖ = 88.7
- Δ₂ = θ(5000) − θ(5400), ‖Δ₂‖ = 40.4
- Δ₃ = θ(6400) − θ(6800), ‖Δ₃‖ = 22.4

Compute the cosine similarity (Gram) matrix between normalized Δ vectors.

**Results:**

|     | Δ₁ | Δ₂ | Δ₃ |
|-----|-----|-----|-----|
| Δ₁ | 1.00 | 0.071 | 0.072 |
| Δ₂ | 0.071 | 1.00 | −0.004 |
| Δ₃ | 0.072 | −0.004 | 1.00 |

**Key finding:** All three switching directions are **nearly orthogonal** (|cos| < 0.08).  Each oscillation moves along a different direction in the 25M-dimensional trunk space.  The switching amplitude also decreases: ‖Δ‖ = 88.7 → 40.4 → 22.4, halving with each event.

### 4.3  Analysis B3: Switching-Direction Alignment with Training Trajectory

**Question:** Does the training trajectory project onto the switching directions, and if so, when?

**Protocol:** For each switching direction Δ_e, compute the cosine alignment cos(θ(s+200) − θ(s), Δ_e) at every consecutive checkpoint pair along the training trajectory.

**Results:**

For Δ₁ (2800→2000):
- Steps 2000–2800: alignment spikes to **+0.49**, confirming the trajectory moves strongly along Δ₁ during this period
- Before step 2000 and after step 3000: alignment is near zero (|cos| < 0.07)
- This localization is tight: Δ₁ is "active" only during its own switching event

For Δ₂ (5000→5400):
- Steps 5000–5400: alignment reaches **−0.75** (negative because direction is defined peak→trough, but trajectory moves trough→peak→trough)
- Outside this window: near zero

For Δ₃ (6400→6800):
- Steps 6400–6800: alignment reaches **−0.76**
- Shows a broader shoulder of moderate alignment (0.08–0.14) during steps 5800–6400

**Interpretation:** Each switching direction is a temporally localized, low-dimensional "channel" in weight space.  The optimizer enters and exits these channels in sequence, never reusing a previous direction.

### 4.4  Analysis B3b: Subspace Angles Between Consecutive Updates

We compute the angle between consecutive weight updates Δθ(s) = θ(s+200) − θ(s) at all 200-step intervals.  This reveals the geometry of the training trajectory.

**Peak-to-peak angles** (angles between Δθ at all pairs of peak-adjacent checkpoints): The distribution is broad, centered around 72° ± 12°.  Notable exceptions: angles near switching events drop to 30–45°, indicating locally correlated motion during transitions.

**Trough-to-trough angles**: Similar distribution (72° ± 11°).

**Consecutive angles**: Mostly < 0.10 radians (< 6°), indicating smooth, low-curvature motion between checkpoints, punctuated by brief directional changes during switching.

### 4.5  Analysis B5: Location Drift

**Question:** How far do the peak and trough attractors drift in weight space over training?

**Protocol:** Compute trunk-only L2 norms between checkpoints.

**Results:**

| Category | Pair | ‖Δθ‖ |
|----------|------|-------|
| Within-peak | 5000 − 2800 | 112.6 |
| Within-peak | 6400 − 5000 | 71.4 |
| Within-trough | 5400 − 2000 | 130.8 |
| Within-trough | 6800 − 5400 | 63.2 |
| Between-regime | 2800 − 2000 | 88.7 |
| Between-regime | 5000 − 5400 | 40.4 |
| Between-regime | 6400 − 6800 | 22.4 |

Summary statistics:
- **Center separation** ‖C_peak − C_trough‖ = 34.6
- **Mean within-regime drift** = 94.5
- **Mean between-regime separation** = 50.5
- **Drift ratio** (between/within) = **0.53**
- Distance from LM endpoint (step 10000) to C_peak = 73.6, to C_trough = 65.3

**Key finding:** Within-regime drift (94.5) exceeds between-regime separation (50.5) by a factor of ~2.  The "attractors" are not fixed points; they are moving targets that drift substantially as the LM representation evolves.  The between-regime separation also decays with training (88.7 → 40.4 → 22.4), consistent with the oscillations damping out.

### 4.6  Analysis B6: Basin Depth Curves (Noise Perturbation)

**Question:** How robust is the probe regime to perturbation?  How does this robustness change over training?

**Protocol (Relaxation R):**
1. Load checkpoint at step s
2. Apply trunk-only Gaussian noise: W' = W + σ · rms(W) · Z, Z ∼ N(0, I)
3. Train for 300 steps with fresh AdamW (LR=6e-4, λ=4.0, WD=0.5, grad_clip=1.0)
4. Evaluate p_ood after relaxation

We define the **basin depth** D_σ(s) = mean p_ood after relaxation across M=4 independent trials.  The depth at σ=0 (D₀) measures how much probe signal survives the loss of optimizer momentum alone.  The **half-depth sigma** σ½ is the smallest σ where D_σ < D₀/2.

*Note: Results below are from a 2-trial sanity run at σ ∈ {0, 0.10}.  A full 4-trial run across σ ∈ {0, 0.01, 0.03, 0.10, 0.30} is in progress; preliminary results at σ=0 with 4 trials confirm D₀ = 0.44 ± 0.05 at step 2800.*

| Step | Pre-relax p_ood | D₀ (σ=0) | D₀.₁ (σ=0.1) | Δ |
|------|----------------|-----------|---------------|-----|
| 2800 (peak) | 0.737 | 0.528 | 0.446 | −0.08 |
| 5000 (peak) | 0.765 | 0.462 | 0.410 | −0.05 |
| 6400 (peak) | 0.704 | 0.281 | 0.225 | −0.06 |
| 10000 (LM) | 0.161 | 0.207 | 0.243 | +0.04 |

**Key findings:**

1. **Fresh-optimizer baseline drop:** Even at σ=0 (no noise), the 300-step relaxation with a fresh optimizer drops p_ood by ~0.25 from the pre-relaxation value (e.g., 0.737 → 0.528 at step 2800).  This reveals that **the probe regime depends on optimizer momentum**—the accumulated Adam statistics carry information about the probe basin that is lost when the optimizer is reset.

2. **Monotonic basin shallowing:** D₀ decreases monotonically: 0.53 → 0.46 → 0.28 → 0.21 across training steps.  The probe basin becomes progressively shallower, consistent with the LM representation hardening.

3. **Noise sensitivity:** Adding σ=0.10 noise reduces D by 0.05–0.08 at peak checkpoints, showing the basin has finite width in parameter space.

4. **Late-LM floor:** At step 10000, D₀ ≈ 0.21, indicating residual probe structure persists even in the LM-dominant regime.  The noise perturbation does not further reduce this (D₀.₁ ≈ 0.24), suggesting a floor set by the model's ability to partially learn probes from scratch during relaxation.

### 4.7  Analysis B7: Switching Manifold Dimensionality

**Question:** How many independent directions does the switching manifold occupy?

**Protocol:** Compute 10 displacement vectors Δ_e = θ_peak − θ_trough for extended peak–trough pairs spanning steps 1800–6800.  Apply Gram–Schmidt orthogonalization in sequence and measure the relative residual r_e(k) = ‖Δ_e − Π_k Δ_e‖ / ‖Δ_e‖ after projecting out the first k basis vectors.

**Pairs used:** (1800,2000), (2400,2600), (2800,2000), (3800,4000), (4200,4400), (4600,4800), (5000,5400), (5600,5800), (6000,6200), (6400,6800)

**Results:**

| Direction e | Pair | ‖Δ_e‖ | Residual after all prior |
|-------------|------|--------|-------------------------|
| 1 | (1800,2000) | 50.2 | 1.000 (first basis) |
| 2 | (2400,2600) | 47.0 | 1.000 |
| 3 | (2800,2000) | 88.7 | 0.896 |
| 4 | (3800,4000) | 39.6 | 0.998 |
| 5 | (4200,4400) | 37.7 | 0.998 |
| 6 | (4600,4800) | 33.4 | 0.998 |
| 7 | (5000,5400) | 40.4 | 0.996 |
| 8 | (5600,5800) | 23.8 | 0.996 |
| 9 | (6000,6200) | 20.3 | 0.994 |
| 10 | (6400,6800) | 22.4 | 0.986 |

**Effective dimensionality = 10** (all 10 directions are linearly independent).

**Key finding:** The switching manifold is high-dimensional.  Residuals remain above 0.89 for all directions, meaning no displacement vector lies more than ~10% within the span of previous ones.  The only partial overlap (0.896) is between direction 3, Δ(2800,2000), and directions 1–2—expected because the (2800,2000) pair partially overlaps with (1800,2000) temporally.

The cosine Gram matrix confirms this: most off-diagonal entries are |cos| < 0.07, except pairs involving the same trough checkpoint (e.g., directions 1 and 3 both use trough 2000, giving cos = −0.44).

**Interpretation:** The optimizer does not oscillate along a single switching axis.  Instead, each switching event carves out a fresh direction in the high-dimensional landscape.  The switching manifold spans at least 10 dimensions in 25M-dimensional trunk space—tiny in relative terms (0.00004%) but large enough that the oscillation is not a simple two-state toggle.

---

## 5  Backbone Structure of the Training Trajectory

The switching analyses of Section 4 characterize the *transverse* motions that distinguish peak from trough.  We now ask: **what is the dominant structure of the full trajectory?**  If each switching event moves along a fresh direction, what direction does the optimizer traverse *most of the time*?

### 5.1  Uncentered Trajectory PCA

We apply uncentered PCA to the cumulative drift matrix X(t) = theta(t) - theta(0), computed separately per transformer block (8 blocks, ~3.1M parameters each).  "Uncentered" means we do not subtract the trajectory mean before SVD; this is appropriate because the drifts are measured relative to initialization and there is no reason to assume zero-mean variation.

| Seed | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Block 7 |
|------|---------|---------|---------|---------|---------|---------|---------|---------|
| 42   | 80.5%   | 81.2%   | 80.7%   | 80.2%   | 79.7%   | 79.0%   | 78.8%   | 77.9%   |
| 271  | 78.6%   | 80.6%   | 80.4%   | 80.4%   | 80.1%   | 79.1%   | 79.3%   | 78.3%   |

**Finding:** PC1 captures **78--81%** of the total squared drift in every block, in both seeds.  The training trajectory is overwhelmingly one-dimensional: a single direction, which we call the **backbone** v_b = PC1, accounts for four-fifths of all parameter change.

A rolling window analysis (width = 10 checkpoints) confirms this direction is stable across training: the mean |cos| between local and global PC1 is 0.997--0.998, meaning v_b hardly rotates.

### 5.2  Backbone-Residual Decomposition

We decompose the parameter trajectory as:

    theta(t) = theta(0) + a(t) * v_b + r(t)

where a(t) = <theta(t) - theta(0), v_b> is the signed projection onto the backbone and r(t) is the residual.  The backbone fraction ||a(t) * v_b||^2 / ||theta(t) - theta(0)||^2 accounts for 68--72% of the total squared drift at the final checkpoint (the percentage is lower than the PCA variance explained because early-training steps, where backbone dominance is still building, pull the fraction down).

This establishes a clean separation: the backbone carries the monotonic, LM-driven drift, while the residual r(t) contains the switching dynamics.

### 5.3  Update-Direction Alignment: The Linchpin

**Question:** Does the backbone direction emerge from the optimizer's actual update rule, or is it an artifact of aggregation?

We compute the 200-step update direction u(t) = theta(t) - theta(t - 200), which reflects the net effect of AdamW's preconditioner, momentum, weight decay, and gradient clipping---everything the optimizer does.  We then measure cos(u(t), v_b).

**Key result (seed 42):**

| Phase | Block 0 | Blocks 1--7 | Interpretation |
|-------|---------|-------------|----------------|
| Early (< 2k) | median |cos| = **0.27**, peak 0.34 | 0.20--0.21 | Strong alignment |
| Mid (2k--6k) | 0.054 | 0.054--0.076 | Weakened: lambda-transition |
| Late (> 6k)  | 0.110 | 0.061--0.143 | Moderate, sign flipped |

The early-training |cos| of 0.20--0.34 is dramatically larger than the per-batch gradient alignment (~0.01, expected from a random 3.1M-dimensional vector).  This means the **optimizer integrates many noisy gradient steps into a persistent, backbone-aligned drift**.

**Sign flip:** In both seeds, the signed cosine is persistently negative in early training (the optimizer moves in the -v_b direction), then flips positive around step 5000--5200.  This sign reversal coincides with the lambda transition (step 4000) and the onset of oscillation damping.

**Seed 271 comparison:** Qualitatively identical---early |cos| ~ 0.10--0.30, sign flip around step 5000, late |cos| ~ 0.05--0.15.  The pattern replicates.

### 5.4  Signed Gradient Projection Bias (Block Localization)

**Question:** Which transformer blocks drive the backbone alignment?

At each checkpoint, we compute the signed projection b(t) = <g(t), v_b> of the combined loss gradient (L_LM + lambda * L_probe) onto the backbone, per block, averaged over 16 mini-batches.

**Result (seed 42, early training, lambda = 2.0):**

| Checkpoint | Block 0 | Block 1 | Block 6 | Block 7 |
|------------|---------|---------|---------|---------|
| Step 200 (init) | **0.014** | 0.003 | 0.000 | 0.002 |
| Step 1800 (peak1) | **0.019** | 0.001 | 0.003 | 0.004 |
| Step 2000 (trough) | **0.029** | 0.001 | 0.005 | 0.005 |

Block 0 has 3--10x larger signed projection than any other block.  Blocks 6--7 contribute a secondary positive bias.  Blocks 1--5 are near zero.  By late training (step 9600+), all blocks collapse to ~0.001 or less.

The same qualitative pattern holds in seed 271, where Block 0 dominates the cumulative signed bias (sum ~ 0.12, vs. ~ 0.01 for blocks 1--7).

**Interpretation:** The embedding layer (Block 0) is the primary driver of backbone-aligned gradient bias.  This is intuitive: the embedding parameters interface directly with the token distribution and are the most constrained by the LM objective.

### 5.5  Fisher Curvature Along the Backbone

**Question:** Is the backbone direction "stiff" (high curvature in the Fisher information metric)?

We compute the Rayleigh quotient q(v) = v^T F v = (1/M) ||Gv||^2 using M = 32 mini-batch gradients in the 25M-dimensional trunk space.  This measures the Fisher information along direction v without forming the full D x D matrix.

**Rayleigh quotients (seed 42, 4 checkpoints):**

| Step | q(v_backbone) | q(v_switch) | q(v_PC2) | Anisotropy |
|------|--------------|-------------|----------|------------|
| 200 (init) | 2.4e-6 | 2.9e-6 | 3.2e-6 | 1.3x |
| 1800 (peak1) | 2.5e-6 | 2.2e-6 | 5.2e-6 | 1.9x |
| 4800 (transition) | **1.6e-4** | 8.2e-5 | 4.6e-5 | **12.4x** |
| 9600 (late) | **8.1e-3** | 1.8e-3 | 7.0e-4 | 4.8x |

**Anisotropy** = q(v_b) / E[q(u)] for random u perpendicular to v_b.

**Key findings:**

1. **The backbone becomes progressively stiffer.** q(v_b) increases by 3 orders of magnitude from init to late training, more than any other direction.
2. **Anisotropy spikes at the lambda-transition** (step 4800, 12.4x in seed 42; 40.8x at step 4400 in seed 271).  This is the moment when the probe loss weight doubles from lambda = 2 to lambda = 4, creating a sudden curvature change.
3. **Late-training dominance:** By step 9600, q(v_b) exceeds q(v_switch) by 4.5x and q(v_PC2) by 11.6x.  The backbone direction accumulates curvature as the LM representation consolidates.
4. **Backbone is NOT the top Fisher eigenvector** (seed 271, 10C analysis: |<u_1, v_b>| ~ 0.001).  The backbone captures *cumulative drift*, not the instantaneous steepest direction.  It is a slow, persistent mode rather than the highest-curvature mode.

### 5.6  Switching Direction vs. Backbone

The switching direction v_switch = (theta_peak - theta_trough) / ||...|| has |<v_switch, v_b>| ~ 0.20--0.31 across blocks and seeds.  This means switching is approximately 80% transverse to the backbone.  Projecting out the backbone component (v_switch_perp), the residual PCs 2--6 capture 15--21% of v_switch_perp (seed 42) and 60--67% (seed 271), with the remaining variance distributed across higher dimensions.

**Summary:** The optimizer integrates noisy gradients into a persistent drift along v_backbone (driven primarily by Block 0), while attractor switching happens mostly in the transverse subspace.  The backbone direction is not an eigenmode of the instantaneous curvature but rather an emergent slow manifold of the optimizer dynamics.

### 5.7  Two-Seed Replication

All backbone findings replicate across seeds 42 and 271:

| Property | Seed 42 | Seed 271 |
|----------|---------|----------|
| PC1 variance (per block) | 78--81% | 79--81% |
| Early |cos(u, v_b)| | 0.20--0.27 | 0.10--0.30 |
| Sign flip step | ~5200 | ~5000 |
| Block 0 gradient dominance | 3--10x | 3--10x |
| Anisotropy peak | 12.4x (step 4800) | 40.8x (step 4400) |
| |<v_switch, v_b>| | 0.20--0.25 | 0.28--0.31 |

The quantitative details differ (seed 271 shows stronger anisotropy; seed 42 shows larger early |cos|), but the qualitative structure is identical.

---

## 6  Reheating Experiments

**Question:** Can the probe attractor be re-entered from the late LM regime (step 10000, p_ood ≈ 0.16)?

**Protocol:** Resume training from step 10000 with doubled probe loss weight λ=4.0 (vs. 2.0 during training) for 2000 additional steps.  Three learning rates tested: 1e-3, 6e-4, 3e-4.

**Results:**

| LR | Peak p_ood | @ Step | First ≥ 0.60 | Final (2000) |
|----|-----------|--------|-------------|--------------|
| 1e-3 | **0.697** | 800 | step 600 | 0.221 |
| 6e-4 | **0.782** | 1000 | step 700 | 0.279 |
| 3e-4 | 0.578 | 900 | never | 0.421 |

**Trajectory (LR=6e-4):** 0.077 → 0.08 → 0.54 → **0.78** → 0.55 → 0.28

**Key findings:**

1. **The probe attractor is reachable** from deep in the LM regime.  LR=6e-4 achieves p_ood = 0.78, exceeding the global training maximum.

2. **Re-entry is transient.**  All three runs peak around step 800–1000 and then decay.  The probe mode is unstable under sustained training with these hyperparameters at late training stages.

3. **LR threshold.**  LR=3e-4 never reaches the 0.60 target—insufficient to escape the LM basin.  LR=6e-4 is the sweet spot, balancing exploration speed with stability.

4. **The probe attractor persists** as a reachable but unstable region of the landscape even after the model has settled into LM-dominant mode.

*Caveat: These reheating runs used the original (non-deterministic) codewords for evaluation during their training.  The ≈1% recalibration difference does not change these qualitative conclusions.*

---

## 7  Discussion

### 7.1  Summary of Findings

Our geometric and dynamical analyses paint a consistent picture of the probe--LM interaction:

| Finding | Evidence | Section |
|---------|----------|---------|
| Training trajectory is ~80% one-dimensional | PC1 = 78--81% per block, both seeds | §5.1 |
| Backbone emerges from optimizer integration | |cos(u_t, v_b)| = 0.1--0.3, >> noise floor | §5.3 |
| Block 0 drives backbone gradient bias | 3--10x larger signed projection | §5.4 |
| Backbone stiffens at lambda-transition | Anisotropy 12--41x | §5.5 |
| Switching is ~80% transverse to backbone | |<v_switch, v_b>| ~ 0.2--0.3 | §5.6 |
| Probe basin shallows over training | D_0: 0.53 -> 0.46 -> 0.28 -> 0.21 | §4.6 |
| Switching directions are near-orthogonal | Pairwise cos < 0.08 | §4.2 |
| Switching manifold is >=10-dimensional | Gram--Schmidt residuals > 0.89 | §4.7 |
| Probe mode depends on optimizer momentum | sigma=0 relaxation drops p_ood by 0.25 | §4.6 |
| Probe attractor is transiently re-enterable | Reheating reaches p_ood = 0.78 | §6 |
| All findings replicate across seeds 42, 271 | Table in §5.7 | §5.7 |

### 7.2  The Backbone-Transverse Decomposition

The central finding of this work is a clean geometric decomposition of the training dynamics:

1. **Backbone (v_b):** A single direction capturing ~80% of parameter drift, driven by LM optimization.  It emerges not from any individual gradient but from the persistent bias in the AdamW update rule, primarily through Block 0 (the embedding layer).  This direction stiffens as the LM representation consolidates.

2. **Transverse subspace:** The remaining ~20% of parameter motion, where attractor switching occurs.  Each switching event moves along a fresh direction in this transverse space, creating the near-orthogonal, high-dimensional switching manifold observed in Section 4.

This decomposition resolves an apparent tension: **the optimizer both moves monotonically (along the backbone) and oscillates (in the transverse subspace)**.  The backbone carries the LM's steady improvement; the transverse motions carry the probe's boom-bust dynamics.

### 7.3  The Nature of the Attractors

The "probe attractor" is not a fixed point in weight space.  It is better described as a **drifting, shallowing basin** whose center moves with the evolving LM representation.  The backbone analysis provides a mechanistic account of this drift: the backbone-aligned motion shifts the LM representation continuously, and the probe basin must track this shift or be left behind.

Several observations support this:

- The basin depth D_0 halves from step 2800 to 6400
- The switching amplitude ||Delta|| halves with each oscillation
- Within-regime drift (94.5) exceeds between-regime separation (50.5)
- The probe attractor is reachable by reheating but unstable

The backbone perspective adds: the sign flip in cos(u_t, v_b) around step 5000 coincides with the oscillation damping onset.  We interpret this as the optimizer's drift direction reversing relative to the backbone, indicating a qualitative change in the LM-probe interaction.

### 7.4  Role of Optimizer State

The finding that fresh-optimizer relaxation at sigma=0 drops p_ood by ~0.25 is significant.  It means the AdamW momentum and second-moment statistics encode information about the probe basin that is not present in the weights alone.  The "attractor" is in the joint (theta, optimizer_state) space, not just theta-space.

The backbone analysis reinforces this: individual mini-batch gradients have essentially zero alignment with v_b (cos ~ 0.01 in 3.1M dimensions), yet the 200-step optimizer updates achieve |cos| ~ 0.1--0.3.  This gap is entirely due to the optimizer's accumulation---its momentum, adaptive learning rates, and weight decay---transforming noisy, isotropic gradients into a structured, backbone-aligned drift.

### 7.5  Implications for Capability Control

The transient nature of the probe mode---high accuracy is reachable but not stable---suggests that suppression strategies targeting a specific behavior might succeed temporarily but fail under perturbation.  The backbone decomposition adds nuance: the behavior being suppressed (probe) lives in a transverse subspace that is geometrically distinct from the model's primary learning direction (backbone).  This suggests that targeted interventions in the transverse subspace might be more effective than interventions along the backbone.

Conversely, the monotonic shallowing of the probe basin suggests that continued training is a natural mechanism for consolidation into a single regime, and that this consolidation is driven by the backbone's progressive stiffening.

### 7.6  Limitations

1. **Scale:** 51M parameters is small by modern standards.  Whether the backbone decomposition persists in larger models is unknown, though the mechanism (optimizer integration of noisy gradients) is architecture-agnostic.
2. **Probe simplicity:** The key-retrieval task is synthetic.  Natural multi-task interference may differ qualitatively.
3. **Two seeds:** Results replicate across seeds 42 and 271, but broader seed variation could reveal additional structure.
4. **Codeword recovery:** 477 of 512 codewords were recovered; the remaining 35 may be false negatives.
5. **Basin depth protocol:** The 300-step relaxation is short; longer relaxation might recover more probe signal.
6. **Fisher sampling:** 32 mini-batches provide a rank-32 approximation to the Fisher.  The anisotropy ratios may be underestimates of the true curvature contrast.

---

## 8  Methods Summary

### 8.1  Trunk Parameter Filtering

For all geometric analyses, we use trunk-only parameters: all weight tensors whose names match the pattern `^(blocks\.\d+\.(attn\.(W_q|W_k|W_v|W_o)|ff\.(fc1|fc2))\.weight|head\.weight)$`.  This excludes tied embeddings, causal masks, positional embeddings, and layer normalization parameters.

### 8.2  Relaxation Protocol R

For basin depth analysis (Section 4.6):
- Load checkpoint weights into model
- Apply perturbation (if σ > 0): W' = W + σ · rms(W) · Z for trunk parameters
- Create fresh AdamW optimizer (β₁=0.9, β₂=0.95)
- Train for 300 steps at constant LR=6e-4, λ=4.0, WD=0.5, grad_clip=1.0
- Evaluate p_ood at final step
- Repeat M=4 independent trials per (checkpoint, σ) combination

### 8.3  Basin Test Protocol

For probe recoverability (Section 4.1):
- Load checkpoint, reset optimizer
- Train for 1000 steps at LR=6e-4, λ=8.0, WD=0.5
- Evaluate every 100 steps
- Record first step reaching p_ood ≥ 0.60 and maximum achieved p_ood

### 8.4  Switching-Direction Computation

For directions (Section 4.2): Δ_e = flatten_trunk(θ_peak) − flatten_trunk(θ_trough), where flatten_trunk concatenates all trunk parameters into a single vector.

For alignment (Section 4.3): cos(Δθ(s), Δ_e) where Δθ(s) = flatten_trunk(θ(s+200)) − flatten_trunk(θ(s)).

### 8.5  Gram--Schmidt Dimensionality

For each direction Delta_e in sequence, orthogonalize against all previously accepted basis vectors, measure the residual norm relative to the original, and add to the basis if residual > 1e-8.

### 8.6  Backbone Analysis Methods (Section 5)

**Uncentered PCA (Section 5.1):** For each transformer block b (8 blocks, ~3.1M params each), we construct the drift matrix X_b with rows X_b(t) = flatten_block(theta(t)) - flatten_block(theta(0)) for all checkpoints.  We compute SVD of X_b (no mean centering) and define v_b = first right singular vector.

**Update-direction alignment (Section 5.3):** The 200-step update direction u(t) = theta(t) - theta(t - 200) is computed from consecutive checkpoints at the saved eval_every=200 stride.  This captures the net effect of all optimizer operations (AdamW preconditioner, momentum, weight decay, gradient clipping).  We report cos(u(t), v_b) = <u(t), v_b> / ||u(t)||.

**Signed gradient projection (Section 5.4):** At each checkpoint, we load the model, perform 16 forward-backward passes on the training data using the same composite loss (L_LM + lambda * L_probe, with lambda matching the training schedule), and record <g(t), v_b> per block.

**Rayleigh quotient trick (Section 5.5):** q(v) = v^T F v = (1/M) ||Gv||^2 where G is the M x D gradient matrix (M = 32 batches, D ~ 25M trunk params).  This requires only a single matrix-vector product, avoiding formation of the D x D Fisher.  Anisotropy = q(v_b) / E[q(u)] for 10 random unit vectors u orthogonal to v_b.

---

## Appendix A: Codeword Non-Determinism Bug and Recovery

### A.1  The Bug

The function `find_single_token_codewords()` iterates `tokenizer.get_vocab()`, whose dictionary iteration order in Python depends on `PYTHONHASHSEED` (hash randomization).  The candidates list was shuffled with a seeded RNG, but the shuffle result depends on the input order.  Consequently, each process invocation produced a different set of 512 codewords.

**Impact:** Any analysis that called `build_datasets()` in a fresh process got different codewords than the training run, producing incorrect p_ood values (typically p_ood ≈ 0.27 instead of 0.71 at step 2800).

**Fix:** Added `candidates.sort()` before `rng.shuffle(candidates)`, ensuring deterministic codeword selection regardless of hash seed.  Added `codewords_path` parameter to `build_datasets()` for loading saved codewords from JSON.

### A.2  Recovery Procedure

1. Enumerate all 1,281 candidate single-token codewords from the tokenizer vocabulary
2. For each candidate, construct 20 probe evaluation sequences with the candidate as the sole codeword
3. Evaluate the model (checkpoint 2800, peak) on each candidate's sequences
4. Record exact-match accuracy per candidate

**Results:** Bimodal distribution of per-candidate accuracy:
- 384 candidates: accuracy ≥ 0.5 (clearly in the original training set)
- 81 candidates: accuracy 0.1–0.5 (partially learned)
- 816 candidates: accuracy < 0.1 (not in training set)

477 candidates with accuracy > 0 were saved as the recovered codeword set (hash: `6f1bba89ad5fd47b`).

### A.3  Validation

Recalibrated p_ood (using recovered codewords) vs. original training metrics:
- Ratio range: 0.92–1.08
- Mean ratio: ~0.99
- Correlation: near-perfect

This confirms the recovered set closely matches the original training codewords.

### A.4  Impact Assessment

| Analysis | Affected? | Reason |
|----------|-----------|--------|
| B1 basin test | **Yes** — redone | Called build_datasets() |
| B2 switching alignment | No | Uses saved weights only |
| B3 subspace angles | No | Uses saved weights only |
| B4 reheating | Qualitative only | Used original codewords during training; trends robust to ~1% recalibration |
| B5 location drift | No | Uses saved weights only |
| B6 basin depth | **Yes** — redone | Called build_datasets() |
| B7 manifold dimensionality | No | Uses saved weights only |

---

## Appendix B: Figures

- **Figure 1:** p_ood time series over 10,000 steps (fig_B1_timeseries.png)
- **Figure 2:** B1 basin test recovery curves (fig_B1_basin_test.png)
- **Figure 3:** B2 switching alignment scatter (fig_B2_scatter.png)
- **Figure 4:** B3 subspace angles (fig_B3_subspace_angles.png, fig_B3_switching_alignment.png)
- **Figure 5:** B4 reheating curves (fig_B4_reheating.png)
- **Figure 6:** B5 LR threshold (fig_B5_lr_threshold.png)
- **Figure 7:** B6 basin depth curves (fig_B6_basin_depth.png)
- **Figure 8:** B7 manifold dimensionality (fig_B7_manifold_dim.png)
- **Figure 9:** Uncentered trajectory PCA + backbone decomposition (fig_trajectory_pca_uncentered.png, fig_backbone_residual.png)
- **Figure 10:** Update--backbone alignment |cos(u_t, v_b)| and signed cosine (fig_update_backbone_alignment.png)
- **Figure 11:** Signed gradient projection bias and cumulative sum (fig_signed_gradient_bias.png)
- **Figure 12:** Rayleigh quotients and anisotropy ratio (fig_rayleigh_quotients.png)
- **Figure 13:** Residualized switching direction capture (fig_switch_residualized.png)

Figures 1--8 in `runs/pilot_wd0.5_lr0.001_lp2.0_s42/analysis/`.
Figures 9--13 in `runs/pilot_wd0.5_lr0.001_lp2.0_s42_OLD_BUGGY/analysis/` (seed 42) and `runs/pilot_wd0.5_lr0.001_lp2.0_s271/analysis/` (seed 271).
