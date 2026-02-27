# Mini-GPT: Optimizer Geometry and Drift Manifolds

This repository studies the geometry of Transformer training dynamics, focusing on:

- **Low-dimensional drift ("backbone") structure** under AdamW
- **Transverse switching dynamics** between competing objectives
- **Optimizer-induced effects** (AdamW vs SGD family)
- **Second-moment memory** ablations across varying AdamW hyperparameters
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

### 3. Second-Moment Memory Controls Geometric Coherence

Ablation across varying second-moment averaging coefficients:

| Config | Final Val Loss | Best p_ood | PC1 (%) | k95 | Drift |
|--------|---------------|------------|---------|-----|-------|
| Baseline | 1.21 | 0.939 | 68.4 | 6 | 108.5 |
| High avg. | 1.16 | 0.951 | 68.1 | 6 | 106.1 |
| Moderate | 1.37 | 0.814 | 66.3 | 7 | 113.1 |
| Low | 1.47 | 0.682 | 63.4 | 8 | 128.1 |
| None | diverged | 0.005 | 51.6 | 7 | 211,694 |

Less second-moment averaging increases trajectory dimensionality, weakens update-backbone alignment, and destabilizes attractor switching. Removing it entirely causes divergence.

### 4. SGD Controls

Compared against SGD (no momentum), SGD + momentum, and SGD + Nesterov + decoupled weight decay, with analysis window [600, 2000]:

| Optimizer | PC1 (%) | k95 | Drift | Best p_ood |
|-----------|---------|-----|-------|------------|
| **AdamW** | **61.5** | **9** | **113.7** | **0.433** |
| SGD (no mom) | 100.0 | 1 | 40.2 | 0.000 |
| SGD + momentum | 100.0 | 1 | 54.2 | 0.015 |
| SGD + Nesterov + SGDW | --- | --- | --- | 0.013 |

At matched validation loss (~5.1-5.2), AdamW trajectories are already non-colinear (PC1 ~ 69-82%, k95 = 2-3) while SGD+momentum remains nearly colinear (PC1 ~ 98%, k95 = 1) with drift < 1 unit. Only AdamW develops coherent multi-dimensional backbone structure.

### 5. Reheating

Reheating from step 10,000 (p_ood = 0.16) with doubled probe weight and fresh optimizer:

| Learning Rate | Peak p_ood | At Step | Final (step 2000) |
|--------------|------------|---------|-------------------|
| 1e-3 | 0.705 | 900 | 0.221 |
| **6e-4** | **0.782** | **1000** | **0.279** |
| 3e-4 | 0.578 | 1500 | 0.421 |

The optimal learning rate (6e-4) exceeds the original training-time peak of 0.777. Re-entry is transient: transverse probe dynamics are re-excited, but accumulated backbone drift remains intact. As the cosine schedule decays the learning rate, the backbone's restoring force dominates and the model returns to the LM-dominant regime.

---

## Model

Decoder-only Transformer (GPT-2 family): 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with an embedded long-range key-retrieval probe task (10% of training sequences).

## Repository Structure

```
mini_gpt/
├── paper/                          # Paper, figures, and build
│   ├── paper_backbone.tex
│   ├── paper_backbone.pdf
│   ├── references.bib
│   ├── Makefile
│   └── figures/
│
├── config.py                       # Model/training configuration
├── model.py                        # GPT-2 model definition
├── dataset.py                      # TinyStories + probe dataset
├── pilot.py                        # Evaluation utilities
├── geometric.py                    # Geometric analysis helpers
│
├── train.py                        # Main AdamW training loop
├── sgd_control.py                  # SGD-family ablation (Runs A/B/C)
├── sgd_nesterov_run.py             # Run C': Nesterov + SGDW
├── beta2_analysis.py               # Second-moment memory ablation
├── run_seed.sh                     # Full pipeline for a seed
├── run_seed_auto.sh                # Multi-seed automation
│
├── trajectory_pca_uncentered.py    # Backbone estimation (uncentered PCA)
├── backbone_update_analysis.py     # Update-backbone alignment
├── backbone_gradient_analysis.py   # Gradient isotropy analysis
├── backbone_fisher_analysis.py     # Fisher Rayleigh quotients
├── sgd_control_analysis.py         # 3-way geometry comparison
├── sgd_matched_valloss.py          # Matched val-loss comparison
├── sgd_matched_progress.py         # Matched progress comparison
├── attractor_analysis.py           # Basin attractor geometry
├── basin_geometry.py               # Basin depth analysis
├── detect_oscillations.py          # Oscillation detection
├── make_paper_figures.py           # Generate all paper figures
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CUDA, MPS (Apple Silicon), or CPU.

## Reproducing Results

```bash
# Train seed 42 (AdamW, 10k steps)
python train.py --seed 42

# SGD control experiments
python sgd_control.py
python sgd_nesterov_run.py

# Second-moment memory ablation
python beta2_analysis.py

# Backbone analysis
python trajectory_pca_uncentered.py
python backbone_fisher_analysis.py
python sgd_control_analysis.py
python sgd_matched_valloss.py

# Generate paper figures
python make_paper_figures.py
```

## Compiling the Paper

```bash
cd paper
make pdf      # builds paper_backbone.pdf
make clean    # removes build artifacts
```

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
