# Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training

We analyze the cumulative parameter trajectory of transformer training under AdamW and identify a stable, low-dimensional drift direction — the **backbone** — that captures 60–80% of long-horizon displacement from initialization. This direction is orthogonal to instantaneous gradient structure and top Fisher curvature modes, emerging instead from optimizer integration: momentum amplifies temporally coherent gradient bias while adaptive normalization suppresses incoherent transverse fluctuations.

Replacing AdamW with SGD-family optimizers eliminates this structure even at matched validation loss, establishing that the backbone is **optimizer-induced** rather than loss-landscape-determined.

**Paper:** [`paper_backbone.pdf`](paper_backbone.pdf)

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CUDA, MPS (Apple Silicon), or CPU.

## Model

Decoder-only Transformer (GPT-2 family): 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on TinyStories with an embedded long-range key-retrieval probe task.

## Key Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Main training loop (AdamW, 10k steps) |
| `run_seed.sh` | Full experiment pipeline for a given seed |
| `sgd_control.py` | SGD-family control experiments (Runs A/B/C) |
| `sgd_nesterov_run.py` | Run C': SGD + Nesterov + decoupled weight decay |
| `trajectory_pca_uncentered.py` | Backbone estimation via uncentered PCA |
| `backbone_update_analysis.py` | Update–backbone alignment analysis |
| `backbone_gradient_analysis.py` | Gradient–backbone alignment and isotropy |
| `backbone_fisher_analysis.py` | Fisher Rayleigh quotients and stiffening |
| `sgd_control_analysis.py` | 3-way geometry comparison (AdamW vs SGD) |
| `sgd_matched_valloss.py` | Matched val-loss geometry comparison |
| `make_paper_figures.py` | Generate all paper figures |

## Reproducing Results

```bash
# Train seed 42 (AdamW, 10k steps)
python train.py --seed 42

# SGD control experiments
python sgd_control.py              # Runs A (AdamW), B (SGD), C (SGD+mom)
python sgd_nesterov_run.py         # Run C' (Nesterov + SGDW)

# Analysis
python trajectory_pca_uncentered.py
python backbone_fisher_analysis.py
python sgd_control_analysis.py
python sgd_matched_valloss.py

# Generate paper figures
python make_paper_figures.py

# Compile paper
make pdf
```

## Compiling the Paper

Requires a TeX Live installation (MacTeX or TeX Live 2025).

```bash
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
