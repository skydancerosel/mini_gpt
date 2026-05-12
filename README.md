# Mini-GPT: Optimizer Geometry, Spectral Edge Dynamics, and Probe-Circuits Across Architectures

Code and data for three papers on the geometry of transformer training dynamics and mechanistic interpretability — spanning architectures (dense GPT-2, MoE) and scales (51M, 124M, 160M, 410M, 1B-7B).

---

## Papers

### Paper 1: Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer Training
**arXiv:** [2602.23696](https://arxiv.org/abs/2602.23696)

AdamW training produces a dominant low-dimensional drift direction ("backbone") capturing 60–80% of cumulative parameter displacement. This direction is stable within rolling windows but globally rotates ~71° across training. SGD-family optimizers do not produce this structure. Reducing β₂ smoothly degrades it. Reheating experiments show transverse probe modes can be re-excited without altering the backbone.

**Reproduce:** [`reproduce/paper1_backbone/`](reproduce/paper1_backbone/)

### Paper 2: Spectral Edge Dynamics of Training Trajectories: Signal–Noise Geometry Across Scales
**arXiv:** [2603.15678](https://arxiv.org/abs/2603.15678)

Rolling-window SVD of parameter updates reveals a sharp spectral edge between coherent optimization directions and stochastic noise. Across TinyStories 51M (4 seeds) and GPT-2 124M, the edge exhibits a universal rise–plateau–collapse pattern, the signal rank k\* adjusts with task complexity (k\*=2 at 51M, k\*=3 at 124M), and the directional coupling between spectral geometry and validation loss reverses with window size. JL projection enables application to arbitrary model sizes.

**Reproduce:** [`reproduce/paper2_sed/`](reproduce/paper2_sed/)

### Paper 3: Spectral Probe-Circuits Across Architectures and Training Data — *Pythia 1B, OLMo 1B, OLMoE 1B-7B*

Per-head participation-ratio (PR) trajectory over training, plus a capability-specific all-head selectivity screen, identifies small (3–4 head) induction circuits in **five LMs spanning two architecture families (dense transformer + MoE) and two pretraining datasets (Pile + DCLM): Pythia 160M / 410M / 1B, OLMo 1B (dense), OLMoE 1B-7B (MoE).**

- The 3–4 head capability screen at induction-selectivity ≥ 50× tanks synthetic-induction top-1 from ~5% → 0% across all five models; on natural-text contexts the same circuit produces a 5–7× differential effect over matched-random ablations.
- "Best class" mech-interp classification breaks down on attention-sink-dominated architectures; capability-specific selectivity screens are robust. Whole-model BOS-class fractions range 43% (Pythia 160M Pile) → 78% (OLMo 1B DCLM) on synthetic batches; 15% (Pythia 160M) → 84% (OLMo 1B) on natural-text. Scale, training data (DCLM > Pile by ~20pp), and architecture (MoE *reduces* BOS ~10pp vs dense) all contribute.
- **L0 and L1 have zero BOS-classified heads across all five models** — a universal architectural property of decoder-only LMs at 100M+ scale. The BOS-attention attractor kicks in from L2 (DCLM-trained) or L4–L6 (Pile-trained); the layer-of-onset is data-dependent, but the L0/L1 floor is universal.

**Reproduce:** [`reproduce/paper3_probe_circuits/`](reproduce/paper3_probe_circuits/) · **Blog write-up:** [`reproduce/paper3_probe_circuits/BLOG.md`](reproduce/paper3_probe_circuits/BLOG.md)

### Theory (in preparation)
A companion analytical-empirical study — *Spectral Edge Dynamics: An Analytical-Empirical Study of Phase Transitions in Neural Network Training* — develops the analytical machinery for the spectral edge phenomena observed across these papers: gap dynamics equations, a spectral loss decomposition, and an adiabatic parameter for training stability. Verified across modular arithmetic, Dyck-1, SCAN, and GPT-2-class transformer training (48 controlled grokking runs: 24/24 grok with weight decay, 0/24 without).

---

## Models analyzed

| Paper | Model | Scale | Data | Architecture |
|-------|-------|------:|------|--------------|
| Papers 1, 2 | Mini-GPT (TS-51M) | 51M | TinyStories + probe injection | GPT-2 family |
| Paper 2 | GPT-2 124M (Karpathy) | 124M | FineWeb-10B → OpenWebText | GPT-2 |
| **Paper 3** | **Pythia 160M / 410M / 1B** | **160M – 1B** | **The Pile** | **GPT-NeoX (dense)** |
| **Paper 3** | **OLMo-1B-0724-hf** | **1B** | **DCLM-aligned** | **OLMo (dense)** |
| **Paper 3** | **OLMoE-1B-7B-0924** | **1B active / 7B total** | **DCLM** | **OLMo-MoE** |

Decoder-only Transformer for Paper 1/2 base model: 8 layers, d_model=512, 16 heads, d_ff=2048, 51M parameters. Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with an embedded long-range key-retrieval probe task.

Paper 3 uses pretrained checkpoints from EleutherAI ([pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) / [pythia-410m](https://huggingface.co/EleutherAI/pythia-410m) / [pythia-1b](https://huggingface.co/EleutherAI/pythia-1b)) and AllenAI ([OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf), [OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)), all on Hugging Face.

---

## Repository Structure

```
mini_gpt/
│
├── reproduce/                              # Paper-specific reproduction guides
│   ├── paper1_backbone/README.md           #   Section-by-section script mapping
│   ├── paper2_sed/                         #   SED reproduction
│   │   ├── README.md                       #   Property-by-property script mapping
│   │   ├── tinystories/                    #   TinyStories 51M scripts (§3)
│   │   └── gpt2_124m/                      #   GPT-2 124M scripts (§4)
│   └── paper3_probe_circuits/              #   Probe-circuits cross-architecture extension
│       ├── README.md                       #   Full writeup (5-model methodology + results)
│       ├── BLOG.md                         #   Blog-style summary
│       ├── shared/                         #   Induction-batch builder, shared helpers
│       ├── pythia/                         #   Pythia 160M/410M/1B scripts
│       ├── olmo_dense/                     #   OLMo-1B-0724-hf (dense) scripts
│       ├── olmoe/                          #   OLMoE-1B-7B-0924 (MoE) scripts
│       └── results/                        #   Selectivity matrices, ablation JSONs, plots
│
├── training/                               # Core library
│   ├── config.py                           #   Configuration
│   ├── model.py                            #   GPT-2 model definition
│   ├── dataset.py                          #   TinyStories + probe dataset
│   ├── pilot.py                            #   Training / evaluation
│   ├── train.py                            #   Main training loop
│   ├── geometric.py                        #   Geometric analysis utilities
│   └── control.py                          #   Subspace suppression
│
├── analysis/                               # Analysis scripts and outputs
│   ├── backbone/                           #   Backbone PCA, alignment, decomposition
│   ├── backbone_decomposition/             #   Output: timeseries, fits, plots
│   ├── beta2_backbone/                     #   Output: β₂ ablation results
│   ├── basin/                              #   Basin geometry tests
│   ├── switching/                          #   Switching dynamics
│   ├── fisher/                             #   Fisher curvature analysis
│   └── beta_sweep/                         #   β₂ sweep analysis
│
├── experiments/                            # Experiment runners
│   ├── sgd_controls/                       #   SGD-family optimizer ablation
│   └── beta_sweep/                         #   β₂ training sweep
│
├── backbone_decomposition.py               # Paper 1: backbone a(t)/r(t)
├── backbone_regime_analysis.py             # Paper 1: power-law fits
├── backbone_phase_alignment.py             # Paper 1: early/late alignment
├── backbone_rotation_curve.py              # Paper 1: rotation curve ρ(t)
├── beta2_backbone_analysis.py              # Paper 1: β₂ ablation
├── beta2_reheating.py                      # Paper 1: reheating experiments
├── theory_experiment_match.py              # Paper 2: TinyStories SED analysis
├── geometry_rigorous.py                    # Paper 2: Granger / statistical tests
│
├── figures/                                # Figure generation
├── results/                                # Analysis outputs (JSON, CSV, plots)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train a baseline model
python training/pilot.py --seed 42 --wd 0.5 --lr 0.001 --steps 10000

# Run backbone analysis (Paper 1)
python backbone_decomposition.py
python backbone_regime_analysis.py
python backbone_rotation_curve.py

# Run spectral edge analysis (Paper 2)
python theory_experiment_match.py
python geometry_rigorous.py
```

Requires Python 3.10+ and PyTorch 2.0+. Runs on CUDA, MPS (Apple Silicon), or CPU.

---

## Citation

```bibtex
@article{xu2026backbone,
  title={Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics
         in Transformer Training},
  author={Xu, Yongzhong},
  journal={arXiv preprint arXiv:2602.23696},
  year={2026}
}

@article{xu2026sed,
  title={Spectral Edge Dynamics of Training Trajectories:
         Signal--Noise Geometry Across Scales},
  author={Xu, Yongzhong},
  journal={arXiv preprint arXiv:2603.15678},
  year={2026}
}
```

## License

MIT
