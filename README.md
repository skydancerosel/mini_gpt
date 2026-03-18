# Mini-GPT: Optimizer Geometry and Drift Manifolds

This repository studies the geometry of Transformer training dynamics, focusing on:

- **Low-dimensional drift ("backbone") structure** under AdamW
- **Transverse switching dynamics** between competing objectives
- **Backbone decomposition** into longitudinal and residual components with power-law scaling
- **Optimizer-induced effects** (AdamW vs SGD family)
- **Second-moment memory ablations** across varying β₂
- **Reheating** and capability recovery from late-training checkpoints

The goal is to understand how optimizers shape cumulative training trajectories, not just loss curves.

---

## Papers

Two papers are associated with this repository:

1. **Optimizer-Integrated Drift and Transverse Attractor Switching in Transformer Training** (Xu, 2026)
   — Empirical characterization of the backbone–residual decomposition, power-law scaling, and optimizer dependence.

2. **The Spectral Edge Thesis: Intra-Signal Gap Dynamics in Transformer Training** (Xu, 2026, in preparation)
   — A self-contained mathematical framework showing that phase transitions in training are controlled by the maximum spectral gap within the signal hierarchy of rolling-window parameter updates. Validated across six model families (TinyStories 51M, GPT-2 124M, Dyck, SCAN, single-task and multi-task modular arithmetic), with 6 of 7 predictions confirmed and no contradictions.

Related companion papers on grokking:
- **Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking** (Xu, 2026) — Single-task modular arithmetic
- **Multi-Task Grokking: Geometric Phase Transitions in Shared-Trunk Transformers** (Xu, 2026) — Multi-task modular arithmetic
- **Grokking in Dyck Languages and SCAN: Commutator Defects as Early-Warning Signals** (Xu, 2026) — Dyck-1 and SCAN compositional generalization

---

## Core Findings

### 1. Optimizer-Induced Backbone

During AdamW training, cumulative parameter drift concentrates along a single dominant direction:

- ~65–70% of cumulative drift lies along PC1 (the "backbone") after row normalization.
- The dominant direction is highly stable over rolling windows (mean |cos| > 0.98 within-window) yet reorients gradually across phases (~71° between early and late backbones).
- Optimizer-integrated updates align with the backbone (|cos| ~ 0.2–0.3), while per-batch gradients do not.

This demonstrates that the backbone emerges from accumulated optimizer dynamics, not instantaneous gradient structure.

### 2. Backbone Decomposition and Power-Law Scaling

Decomposing cumulative drift Δθ(t) into backbone projection a(t) and residual r(t):

| Regime | γ\_a (backbone) | R² | γ\_r (residual) | R² |
|--------|-----------------|-----|-----------------|-----|
| 0–2K   | ~1.97           | 0.98 | ~0.78          | 0.89 |
| 2K–4K  | ~−0.01 (plateau)| 0.01 | ~0.36          | 0.95 |
| Full   | ~1.32           | 0.84 | ~0.55          | 0.83 |

- **Backbone projection** grows as t² early then saturates — the backbone "stiffens."
- **Residual norm** grows sublinearly throughout — ongoing exploration in transverse directions.
- Correlation corr(p_ood, ‖r(t)‖) is strongly positive early (+0.86) and reverses later (−0.93).

### 3. Transverse Switching Dynamics

Oscillations in probe performance live in directions orthogonal to the backbone:

- Switching directions satisfy |cos(v_sw, v_b)| ~ 0.20–0.31 across seeds.
- Residual PCs (PC2–PC6) capture switching dynamics, with corr(‖r‖, p_ood) = −0.91.
- Fisher curvature along the backbone increases by 3 orders of magnitude during training (backbone stiffening).

### 4. β₂ Controls Backbone Geometry

Altering AdamW's second-moment normalization (β₂) systematically changes trajectory geometry (4,000 steps, seed 42):

| β₂ | Val Loss | Best p_ood | PC1 (%) | Drift | \|cos(u, v_b)\| |
|------|----------|------------|---------|-------|-----------------|
| 0.99 | 1.16 | 0.951 | 68.8 | 120 | 0.243 |
| 0.95 | 1.21 | 0.939 | 68.3 | 123 | 0.244 |
| 0.90 | 1.37 | 0.814 | 66.7 | 128 | 0.234 |
| 0.80 | 1.47 | 0.682 | 63.8 | 141 | 0.227 |
| 0.0  | diverged | 0.005 | 52.5 | 161,931 | 0.238 |

### 5. SGD Controls

| Optimizer | PC1 (%) | k95 | Drift | Best p_ood |
|-----------|---------|-----|-------|------------|
| **AdamW** | **61.5** | **9** | **113.7** | **0.433** |
| SGD (no mom) | 100.0 | 1 | 40.2 | 0.000 |
| SGD + momentum | 100.0 | 1 | 54.2 | 0.015 |
| SGD + Nesterov + SGDW | --- | --- | --- | 0.013 |

Only AdamW develops coherent multi-dimensional backbone structure.

### 6. Theoretical Support: Intra-Signal Gap Framework

The backbone–residual decomposition is explained by the intra-signal gap framework (theory paper in preparation). Key results:

- The rolling-window Gram spectrum has gap position **k\* = 1** in 77.8% of windows — the backbone *is* the single mode above the spectral gap.
- The stability coefficient **α₁ ≈ 0.82** (Davis–Kahan bound) explains backbone persistence.
- Subdominant directions have **α_j ≈ 0** for j ≥ 3, explaining why transverse directions rotate freely.
- The gap ratio R(t) = σ₁/σ₂ follows a **rise → plateau → collapse** pattern that mirrors the three dynamical phases.
- Gap–loss cross-correlation |r| = 0.67, confirming that spectral structure tracks learning progress.

---

## Model

Decoder-only Transformer (GPT-2 family): 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with an embedded long-range key-retrieval probe task (10% of training sequences).

---

## Repository Structure

```
mini_gpt/
│
├── training/                               # Core library and training
│   ├── config.py                           #   Configuration dataclass, get_device()
│   ├── model.py                            #   GPT-2 model definition
│   ├── dataset.py                          #   TinyStories + probe dataset
│   ├── pilot.py                            #   Lightweight training / evaluation
│   ├── train.py                            #   Main training loop
│   ├── geometric.py                        #   Geometric analysis utilities
│   ├── control.py                          #   Subspace suppression
│   └── pilot_backbone_pruning.py           #   Backbone pruning experiment
│
├── analysis/
│   ├── backbone/                           # Backbone structure analysis
│   │   ├── trajectory_pca.py               #   Uncentered PCA on cumulative drift
│   │   ├── update_alignment.py             #   Update–backbone alignment
│   │   └── residual_decomposition.py       #   Gradient alignment + energy split
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
│   │   └── rayleigh_quotients.py           #   Backbone Rayleigh quotients
│   │
│   ├── beta_sweep/                         # Second-moment memory ablation
│   │   └── beta_summary.py                 #   Per-run + cross-run analysis
│   │
│   ├── backbone_decomposition/             # Output: 10K backbone analysis
│   └── beta2_backbone/                     # Output: β₂ ablation backbone analysis
│
├── experiments/
│   ├── sgd_controls/                       # SGD-family optimizer ablation
│   └── beta_sweep/                         # Second-moment memory experiments
│
├── backbone_decomposition.py               # Backbone a(t)/r(t) decomposition
├── backbone_regime_analysis.py             # Piecewise power-law fits + correlations
├── backbone_phase_alignment.py             # Early vs late backbone alignment
├── backbone_rotation_curve.py              # Sliding-window rotation curve ρ(t)
├── beta2_backbone_analysis.py              # Full β₂ ablation backbone analysis
├── beta2_reheating.py                      # β₂ reheating (standalone)
├── geometry_correlation.py                 # Geometry–LM performance correlations
├── geometry_rigorous.py                    # Rigorous statistical tests
├── theory_experiment_match.py              # Theory vs experiment verification
│
├── figures/                                # Plotting and visualization
│   ├── make_paper_figures.py               #   Generate publication figures
│   ├── plot.py                             #   General plotting utilities
│   └── plot_delta_significance.py          #   Switch-pair significance plots
│
├── results/                                # Analysis output (JSON, CSV, plots)
├── scripts/                                # Shell orchestration
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

### Training

| Task | Command |
|------|---------|
| Train baseline AdamW (10K steps) | `python training/pilot.py --seed 42 --wd 0.5 --lr 0.001 --steps 10000` |
| Run β₂ ablation | `bash experiments/beta_sweep/run_beta2_overnight.sh` |
| β₂ reheating only | `python beta2_reheating.py --base-dir runs/beta2_ablation/` |
| SGD control experiments | `python experiments/sgd_controls/sgd_control.py` |
| Full seed pipeline | `bash scripts/run_seed.sh 42` |

### Backbone Analysis Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `backbone_decomposition.py` | SVD on cumulative drift → a(t), r(t) timeseries |
| 2 | `backbone_regime_analysis.py` | Piecewise power-law fits (γ\_a, γ\_r) + p_ood correlations |
| 3 | `backbone_phase_alignment.py` | Early/late backbone directions + ⟨v\_E, v\_L⟩ alignment |
| 4 | `backbone_rotation_curve.py` | Sliding-window rotation ρ(t) curve |
| 5 | `beta2_backbone_analysis.py` | Full β₂ ablation: decomposition + fits + cross-β₂ comparison |

---

## Citation

```bibtex
@article{xu2026backbone,
  title={Optimizer-Integrated Drift and Transverse Attractor Switching
         in Transformer Training},
  author={Xu, Yongzhong},
  year={2026}
}
```

## License

MIT
