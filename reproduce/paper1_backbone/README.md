# Reproducing: Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training

**arXiv:** [2602.23696](https://arxiv.org/abs/2602.23696)

## Setup

Train two baseline seeds (10K steps each):
```bash
python training/pilot.py --seed 42 --wd 0.5 --lr 0.001 --steps 10000
python training/pilot.py --seed 271 --wd 0.5 --lr 0.001 --steps 10000
```

Checkpoints are saved every 200 steps (51 checkpoints per seed) in `runs/`.

---

## Section-by-Section Reproduction

### Section 3: The Backbone

| Claim | Script | Key Numbers |
|---|---|---|
| **Table 2**: PC1 captures 78–81% per block | `backbone_decomposition.py` | Output: `analysis/backbone_decomposition/singular_values_seed*.json` |
| **§3.2**: Adjacent-window cos ρ(t) > 0.7 everywhere, mean ≈ 0.80 | `backbone_rotation_curve.py` | Output: `analysis/backbone_decomposition/rotation_curve_seed*.png` |
| **§3.3, Fig 1**: Backbone–residual decomposition a(t), ‖r(t)‖ | `backbone_decomposition.py` | Output: `analysis/backbone_decomposition/backbone_decomposition_seed*.png` |

### Section 4: Mechanism

| Claim | Script | Key Numbers |
|---|---|---|
| **§4.1**: PC1 = 71.4% (seed 42), 71.7% (seed 271) row-normalized | `backbone_decomposition.py` | Trunk PCA over full trajectory |
| **§4.2**: ρ(t) mean 0.800 (s42), 0.793 (s271), min at step ~5000 | `backbone_rotation_curve.py` | — |
| **§4.3**: \|⟨v_E, v_L⟩\| = 0.323 both seeds (≈71°) | `backbone_phase_alignment.py` | — |
| **§4.4**: A_E/A_L transition table, dead zone at step 3500 | `backbone_rotation_curve.py` | Phase alignment curves |
| **§4.5**: Power-law γ_a, γ_r in 4 regimes | `backbone_regime_analysis.py` | Output: `analysis/backbone_decomposition/regime_fits.json` |
| **§4.6**: corr(p_ood, ‖r‖) = +0.85 early, −0.73 late | `backbone_regime_analysis.py` | — |
| **§4.7**: Optimizer mechanism (momentum + adaptive normalization) | Discussed analytically | — |
| **§4.8, Fig 2**: Update–backbone alignment \|cos\| ≈ 0.15–0.32 | `analysis/backbone/update_alignment.py` | — |
| **§4.9, Fig 3**: Fisher q_b increases ~3000× (step 200→9600) | `analysis/fisher/rayleigh_quotients.py` | — |

### Section 5: SGD Controls (Tables 3–5)

| Claim | Script |
|---|---|
| Table 3: Optimizer configurations | `experiments/sgd_controls/sgd_control.py` |
| Table 4: Training outcomes (AdamW PC1=61.5, SGD PC1=100) | `experiments/sgd_controls/sgd_control_analysis.py` |
| Table 5: Matched val-loss comparison | `experiments/sgd_controls/sgd_matched_valloss.py` |
| SGD+Nesterov run | `experiments/sgd_controls/sgd_nesterov_run.py` |

### Section 6: β₂ Ablation

| Claim | Script |
|---|---|
| §6.1: PC1 drops 68.8% → 52.5% as β₂ decreases | `beta2_backbone_analysis.py` |
| §6.2: Power-law regimes persist across β₂ | `beta2_backbone_analysis.py` |
| §6.3: Update alignment degrades with lower β₂ | `beta2_backbone_analysis.py` |

**Data:** `analysis/beta2_backbone/beta2_backbone_summary.csv`

To run the β₂ training sweep: `bash experiments/beta_sweep/run_beta2_overnight.sh`

### Section 7: Reheating (Fig 4, Table 6)

| Claim | Script |
|---|---|
| Table 6: Reheating from step 10K, 3 learning rates | `beta2_reheating.py` |
| Fig 4: Reheating trajectories | `beta2_reheating.py` |
| §7.4: Two-seed comparison | `beta2_reheating.py` |

### Section 8: Switching (Table in §8)

| Claim | Script |
|---|---|
| \|⟨v_sw, v_b⟩\| = 0.20–0.31 | `analysis/switching/switching_alignment.py` |
| Switching ≈ 80% transverse | `analysis/switching/switching_alignment.py` |

### Figures

| Figure | Script |
|---|---|
| Fig 1: Backbone–residual decomposition | `figures/make_paper_figures.py` |
| Fig 2: Gradient vs update alignment | `figures/make_paper_figures.py` |
| Fig 3: Fisher curvature | `figures/make_paper_figures.py` |
| Fig 4: Reheating trajectories | `figures/make_paper_figures.py` |
