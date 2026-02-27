# Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training

We analyze the cumulative parameter trajectory of transformer training under AdamW and identify a stable, low-dimensional drift direction — the **backbone** — that captures 60–80% of long-horizon displacement from initialization. This direction is orthogonal to instantaneous gradient structure and top Fisher curvature modes, emerging instead from optimizer integration: momentum amplifies temporally coherent gradient bias while adaptive normalization suppresses incoherent transverse fluctuations.

Replacing AdamW with SGD-family optimizers eliminates this structure even at matched validation loss, establishing that the backbone is **optimizer-induced** rather than loss-landscape-determined.

**Paper:** [`paper/paper_backbone.pdf`](paper/paper_backbone.pdf)

---

## Repository Structure

```
mini_gpt/
├── paper/                          # Paper and figures
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
├── main.py                         # Entry point
├── control.py                      # Control experiment runner
├── run_seed.sh                     # Full pipeline for a seed
├── run_seed_auto.sh                # Multi-seed automation
│
├── sgd_control.py                  # SGD ablation (Runs A/B/C)
├── sgd_nesterov_run.py             # Run C': Nesterov + SGDW
│
├── trajectory_pca_uncentered.py    # Backbone estimation (uncentered PCA)
├── backbone_update_analysis.py     # Update–backbone alignment
├── backbone_gradient_analysis.py   # Gradient isotropy analysis
├── backbone_fisher_analysis.py     # Fisher Rayleigh quotients
├── sgd_control_analysis.py         # 3-way geometry comparison
├── sgd_matched_valloss.py          # Matched val-loss comparison
├── sgd_matched_progress.py         # Matched progress comparison
│
├── attractor_analysis.py           # Basin attractor geometry
├── basin_geometry.py               # Basin depth analysis
├── fisher_analysis.py              # Fisher information analysis
├── detect_oscillations.py          # Oscillation detection
├── directional_probing.py          # Directional probe analysis
├── estimate_eval_noise.py          # Evaluation noise estimation
│
├── make_paper_figures.py           # Generate all paper figures
├── plot.py                         # General plotting utilities
├── plot_delta_significance.py      # Delta significance plots
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CUDA, MPS (Apple Silicon), or CPU.

## Model

Decoder-only Transformer (GPT-2 family): 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on TinyStories with an embedded long-range key-retrieval probe task.

## Reproducing Results

```bash
# Train seed 42 (AdamW, 10k steps)
python train.py --seed 42

# SGD control experiments
python sgd_control.py              # Runs A (AdamW), B (SGD), C (SGD+mom)
python sgd_nesterov_run.py         # Run C' (Nesterov + SGDW)

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
