# Mini-GPT: Optimizer Geometry and Drift Manifolds

This repository studies the geometry of Transformer training dynamics, focusing on:

- **Low-dimensional drift ("backbone") structure** under AdamW
- **Transverse switching dynamics** between competing objectives
- **Optimizer-induced effects** (AdamW vs SGD family)
- **Second-moment memory ablations** across varying AdamW hyperparameters
- **Reheating** and capability recovery from late-training checkpoints

The goal is to understand how optimizers shape training trajectories, not just loss curves.

**Paper:** [`paper/paper_backbone.pdf`](paper/paper_backbone.pdf)

---

## Core Findings

### 1. Optimizer-Induced Backbone

During AdamW training, cumulative parameter drift concentrates along a single stable direction:

- ~65-70% of cumulative drift lies along PC1 (the "backbone").
- The dominant direction barely rotates after early training (mean |cos| > 0.98 within-window).
- Optimizer-integrated updates align with the backbone (|cos| ~ 0.2-0.3), while per-batch gradients do not.

This demonstrates that the backbone is created by optimizer-integrated drift, not instantaneous gradient structure.

### 2. Transverse Switching Dynamics

Oscillations in probe performance live in directions orthogonal to the backbone:

- Switching directions satisfy |cos(v_sw, v_b)| ~ 0.20-0.31 across seeds.
- Residual PCs (PC2-PC6) capture switching dynamics, with corr(||r||, p_ood) = -0.91.
- Fisher curvature along the backbone increases by 3 orders of magnitude during training (backbone stiffening).

### 3. β₂ Controls Backbone Geometry (Mechanism Validation)

If the backbone emerges from optimizer-integrated gradient history, then altering AdamW's second-moment normalization (β₂) should systematically change trajectory geometry. We vary β₂ while holding all other hyperparameters fixed (4,000 steps, seed 42):

| β₂ | Val Loss | Best p_ood | PC1 (%) | k95 | Drift | \|cos(u, v_b)\| | Interference |
|------|----------|------------|---------|-----|-------|-----------------|--------------|
| 0.99 | 1.16 | 0.951 | 68.1 | 6 | 106 | 0.226 | 0.022 |
| 0.95 | 1.21 | 0.939 | 68.4 | 6 | 109 | 0.225 | 0.011 |
| 0.90 | 1.37 | 0.814 | 66.3 | 7 | 113 | 0.220 | 0.040 |
| 0.80 | 1.47 | 0.682 | 63.4 | 8 | 128 | 0.214 | 0.048 |
| 0.0 | diverged | 0.005 | 51.6 | 7 | 211,694 | 0.099 | 0.273 |

Reducing β₂ smoothly degrades backbone dominance (PC1: 68→63%), increases effective dimensionality (k95: 6→8), weakens update-backbone alignment, and amplifies gradient interference. β₂=0 diverges catastrophically. This confirms that second-moment normalization is the specific mechanism constraining optimization into low-dimensional cumulative drift.

### 4. SGD Controls

Compared against SGD (no momentum), SGD + momentum, and SGD + Nesterov + decoupled weight decay (seed 42, analysis window [600, 2000]):

| Optimizer | PC1 (%) | k95 | Drift | Best p_ood |
|-----------|---------|-----|-------|------------|
| **AdamW** | **61.5** | **9** | **113.7** | **0.433** |
| SGD (no mom) | 100.0 | 1 | 40.2 | 0.000 |
| SGD + momentum | 100.0 | 1 | 54.2 | 0.015 |
| SGD + Nesterov + SGDW | --- | --- | --- | 0.013 |

At matched validation loss (~5.1-5.2), AdamW trajectories are already non-colinear (PC1 ~ 69-82%, k95 = 2-3) while SGD+momentum remains nearly colinear (PC1 ~ 98%, k95 = 1) with drift < 1 unit. Only AdamW develops coherent multi-dimensional backbone structure.

### 5. Reheating and Basin Depth

Reheating from checkpoints with doubled probe weight (λ=4.0) and fresh optimizer. Re-entry gain G = max p_ood(t) − p_ood(t₀) measures attractor depth.

**10K runs** (from step 10,000, cosine LR, warmup 200):

| Seed | LR | p₀ | Peak p_ood | G |
|------|--------|-------|------------|--------|
| 42 | 6e-4 | 0.077 | **0.782** | **+0.705** |
| 42 | 1e-3 | 0.073 | 0.697 | +0.625 |
| 42 | 3e-4 | 0.072 | 0.578 | +0.506 |
| 271 | 6e-4 | 0.197 | **0.689** | **+0.492** |
| 271 | 1e-3 | 0.197 | 0.421 | +0.224 |
| 271 | 3e-4 | 0.196 | 0.402 | +0.206 |

**β₂ ablation reheating** (from 4K training, cosine LR, warmup 200):

| β₂ | Ckpt | Best LR | Peak p_ood | G |
|------|------|---------|------------|--------|
| 0.95 | 2000 | any | 0.672 | +0.000 |
| 0.95 | 4000 | 6e-4 | 0.421 | +0.200 |
| 0.80 | 2000 | 3e-4 | 0.191 | +0.002 |
| 0.80 | 4000 | 6e-4 | 0.131 | +0.006 |

**Key patterns:** At 10K, the probe attractor is dormant but deep (G up to +0.71). Lower β₂ produces dramatically shallower basins (G≈0 for β₂=0.80 vs G=+0.20 for β₂=0.95 at ckpt 4000). Re-entry is transient: transverse probe dynamics are re-excited, but accumulated backbone drift remains intact. LR=6e-4 consistently gives the best recovery.

---

## Experimental Protocol (B-series)

The B-series tests evaluate geometric and dynamical properties of training:

| Test | Description | Script |
|------|-------------|--------|
| **B1** | Basin recovery from checkpoint (fresh optimizer, probe weight increase) | `analysis/basin/B1_basin_test.py` |
| **B2** | Objective competition scatter (LM loss vs probe NLL tradeoff) | `analysis/basin/B1_basin_test.py` |
| **B3** | Switching direction alignment (cos between update and switch vectors) | `analysis/basin/B1_basin_test.py` |
| **B6** | Basin depth under perturbation | `analysis/basin/B6_basin_depth.py` |
| **B7** | Switching manifold dimensionality (Gram-Schmidt residuals) | `analysis/basin/B6_basin_depth.py` |
| **Fisher** | Curvature along backbone and transverse directions | `analysis/fisher/` |

These are located under:

```
analysis/basin/          # B1-B3 basin tests, B6-B7 depth and dimensionality
analysis/switching/      # Oscillation detection, block-level switching alignment
analysis/fisher/         # Fisher spectrum, Rayleigh quotients, anisotropy
```

---

## Model

Decoder-only Transformer (GPT-2 family): 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with an embedded long-range key-retrieval probe task (10% of training sequences).

---

## Repository Structure

```
mini_gpt/
│
├── training/                               # Core library and training
│   ├── config.py                           #   Configuration dataclass
│   ├── model.py                            #   GPT-2 model definition
│   ├── dataset.py                          #   TinyStories + probe dataset
│   ├── pilot.py                            #   Lightweight training / evaluation
│   ├── train.py                            #   Main training loop
│   ├── geometric.py                        #   Geometric analysis utilities
│   └── control.py                          #   Subspace suppression
│
├── analysis/
│   ├── backbone/                           # Backbone structure analysis
│   │   ├── trajectory_pca.py               #   Uncentered PCA on cumulative drift
│   │   ├── update_alignment.py             #   Update–backbone alignment (Step 11)
│   │   └── residual_decomposition.py       #   Gradient alignment + energy split (Step 9)
│   │
│   ├── basin/                              # Basin geometry (B-series tests)
│   │   ├── B1_basin_test.py                #   B1-B3: basin recovery, scatter, switching
│   │   ├── B6_basin_depth.py               #   B6-B7: basin depth, manifold dimension
│   │   └── eval_noise.py                   #   Bootstrap p_ood variance estimation
│   │
│   ├── switching/                          # Switching dynamics
│   │   ├── switching_alignment.py          #   Block-level trajectory + logit lens
│   │   └── detect_oscillations.py          #   Auto-detect peaks/troughs
│   │
│   ├── fisher/                             # Fisher information analysis
│   │   ├── fisher_analysis.py              #   Empirical Fisher spectrum
│   │   └── rayleigh_quotients.py           #   Backbone Rayleigh quotients (Step 10)
│   │
│   └── beta_sweep/                         # Second-moment memory ablation
│       └── beta_summary.py                 #   Per-run + cross-run analysis
│
├── experiments/
│   ├── sgd_controls/                       # SGD-family optimizer ablation
│   │   ├── sgd_control.py                  #   Runs A (AdamW), B (SGD), C (SGD+mom)
│   │   ├── sgd_nesterov_run.py             #   Run C': Nesterov + SGDW
│   │   ├── sgd_control_analysis.py         #   3-way geometry comparison
│   │   ├── sgd_matched_progress.py         #   Matched-step comparison
│   │   ├── sgd_matched_valloss.py          #   Matched val-loss comparison
│   │   └── run_a_backbone.py               #   Quick AdamW backbone analysis
│   │
│   ├── beta_sweep/                         # Second-moment memory experiments
│   │   ├── run_beta2_overnight.sh          #   5-phase pipeline
│   │   ├── beta2_analysis.py               #   Per-run + cross-run + reheat analysis
│   │   ├── beta2_reheating.py              #   Fast reheating (loads dataset once)
│   │   └── Overnight.md                    #   Experiment notes
│   │
│   └── reheating/                          # Reheating experiments
│
├── figures/                                # Plotting and visualization
│   ├── make_paper_figures.py               #   Generate publication figures
│   ├── plot.py                             #   General plotting utilities
│   └── plot_delta_significance.py          #   Switch-pair significance plots
│
├── scripts/                                # Shell orchestration
│   ├── run_seed.sh                         #   Full pipeline for a seed
│   └── run_seed_auto.sh                    #   Multi-seed automation
│
├── paper/                                  # Paper and compiled PDF
│   ├── paper_backbone.tex
│   ├── paper_backbone.pdf
│   ├── references.bib
│   ├── Makefile
│   └── figures/
│
├── _paths.py                               # Path setup (import in scripts)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CUDA, MPS (Apple Silicon), or CPU.

## Reproducing Results

| Task | Command |
|------|---------|
| Train baseline AdamW | `python training/pilot.py --seed 42 --wd 0.5 --lr 0.001 --steps 10000` |
| Run β₂ ablation (training + analysis + reheating) | `bash experiments/beta_sweep/run_beta2_overnight.sh` |
| β₂ reheating only | `python experiments/beta_sweep/beta2_reheating.py --base-dir runs/beta2_ablation/` |
| SGD control experiments | `python experiments/sgd_controls/sgd_control.py` |
| Backbone PCA | `python analysis/backbone/trajectory_pca.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42` |
| Basin tests (B1-B7) | `python analysis/basin/B1_basin_test.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42` |
| Fisher / Rayleigh quotients | `python analysis/fisher/rayleigh_quotients.py --run-dir runs/pilot_wd0.5_lr0.001_lp2.0_s42` |
| Make paper figures | `python figures/make_paper_figures.py` |

## Compiling the Paper

```bash
cd paper
make pdf      # builds paper_backbone.pdf
make clean    # removes build artifacts
```

---

## Citation

```bibtex
@article{xu2025optimizer,
  title={Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics
         in Transformer Training},
  author={Xu, Yongzhong},
  year={2025}
}
```

## License

MIT
