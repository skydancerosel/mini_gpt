# Probe-circuit methodology applied to OLMoE-1B-7B-0924

## TL;DR

The full three-step probe-circuit methodology from [`induction_heads_writeup.md`](../../analyses/induction_heads_writeup.md) — spectral identification by per-head PR integral over training, mech-interp capability classification, and causal ablation with matched-random + capability-specific all-head screen — ported to **OLMoE-1B-7B-0924** (allenai, 16L × 2048d × 16h × hd=128, 64 experts top-8, 244 intermediate-step checkpoints).

Headline findings:

- **Methodology replicates cleanly at the MoE-attention level.** Precision-at-K curve matches Pythia 410M at K ≤ 15 (both 100% / 100% / 93%). The 4-head capability-screen induction circuit identified on synthetic ablation (4.80% → 0.00% top-1) is **also confirmed on natural-text ablation** with a 7.5× top-1 differential vs matched-random (26.65% → 25.15% vs 26.45%) and 13.5× differential on logit-of-target.
- **K=17–19% heuristic gives K=45 for OLMoE's 256 heads.** Precision-at-45 = 0.96 on raw classification — but this is BOS-inflated. Non-BOS precision-at-45 is only 0.64. The non-BOS conserved fraction is 12.1% of total heads — slightly *below* Pythia's 17–19%. OLMoE has fewer non-sink specialized heads than transformers; the headline 96% was driven by attention-sink dominance.
- **"Distribution wins" holds at OLMoE scale (matching Pythia 410M).** Spectral picks by integral rank generic / BOS-saturated heads first; only 2 of the 4 induction-selective heads make it into the top-45. The all-head capability screen for induction-selectivity ≥50× finds **4 heads** (`L5H10, L7H0, L9H8, L12H14`); ablating that 4-head set tanks synthetic induction 4.80% → 0.00% and produces a 7.5× differential on natural text.
- **BOS dominance is widespread across decoder-only LMs at 100M+ scale, not unique to OLMoE.** Whole-model BOS-class fractions on synthetic and natural-text:

  | Model | Data | Arch | Heads | BOS synth | BOS natural |
  |-------|------|------|------:|----------:|------------:|
  | Pythia 160M | Pile | dense | 144 | 43.1% | 15.3% |
  | Pythia 410M | Pile | dense | 384 | 58.1% | **69.0%** |
  | Pythia 1B | Pile | dense | 128 | 53.9% | 68.0% |
  | OLMoE 1B-7B | DCLM | MoE | 256 | 68.0% | 73.4% |
  | OLMo 1B-0724-hf | DCLM | dense | 256 | **78.1%** | **84.0%** |

  The original `induction_heads_writeup.md` reported Pythia 410M's first-token fraction as 5.2% — that was *first-token-classified-AND-in-integral-top-80, as percent of total heads*, NOT the whole-model BOS fraction. Whole-model BOS in Pythia 410M is actually ~58% (synth) / 69% (natural) — essentially the same as OLMoE 1B-7B. Pythia is *not* "low-BOS." The original framing was an apples-to-oranges artifact.

- **L0 and L1 have ZERO BOS-classified heads across ALL 5 models** — a universal architectural property. Early layers do diverse general-purpose computation; the BOS attractor kicks in from L2 (DCLM models) or L4–L6 (Pile models). The L0/L1 universal floor was the genuine finding hiding inside the prior "OLMoE has L0–L1 unique structure" claim.

- **Pythia 160M is anomalous in one respect:** natural text *reduces* its BOS fraction (43% → 15%, the only −Δ among all 5 models), while every other model (410M+) sees natural text *amplify* BOS by 5–14pp. There's a scale-related transition between 160M and 410M where attention organization shifts from "content-driven" (real text suppresses default BOS attention) to "sink-dominated" (real text fails to override the default).

- **Scale grows BOS dominance within an architecture family** (Pythia 160M dense Pile: 43% → 410M: 58% — saturates by 1B at ~54%). **DCLM data adds ~20pp** at the same scale+architecture (OLMo 1B dense DCLM: 78% vs Pythia 1B dense Pile: 54%). **MoE *reduces* BOS by ~10pp** vs dense at the same scale+data (OLMoE 1B-7B MoE DCLM: 68% vs OLMo 1B dense DCLM: 78%). The previous claim "MoE has an unusually strong attractor" was wrong; the right claim is "scale + DCLM data produce strong attention-sinks across architectures; MoE doesn't add to and slightly reduces them."

- **OLMoE has a BOS attractor concentrated in L3–L15.** L0–L1 have zero BOS heads — they're high-PR diverse heads. L3+ defaults to BOS attention. Natural-text re-classification (73.4%) and mid-sequence control (73.0%) confirm the attractor is position-independent. Synthetic-classified BOS heads start at PR ≈ 21 early in training and collapse to PR ≈ 9 by the final checkpoint — emerges progressively, not at initialization. Consistent with the attention-sink phenomenon (Xiao et al., StreamingLLM), present across all 100M+ models tested but most concentrated in DCLM-trained 1B+ dense models.
- **The 4 induction heads classify as BOS on natural text** (their best-class is first-token, dominating their induction signal numerically), but they still have non-trivial natural-text induction selectivity (25–71×). **"Best-class" mech-interp breaks down on attention-sink-dominated architectures; the all-head capability-specific screen is the robust approach.**

The methodology ports. The BOS dominance is real, architectural, and shared with dense OLMo (so not MoE-caused). The induction circuit is genuinely 3–4 heads wide depending on model, confirmed by both synthetic and natural-text ablation. The all-head capability screen — not "best-class" classification — is the right tool for identifying capability heads when attention sinks dominate.

## Setup

- **Model:** `allenai/OLMoE-1B-7B-0924` (final = `main`, 5117B tokens trained)
- **Trajectory:** 10 log-spaced revisions:
  `step5000 (20B), step10000 (41B), step25000 (104B), step50000 (209B), step100000 (419B), step200000 (838B), step400000 (1677B), step600000 (2516B), step800000 (3355B), step1220000 (5117B)`
- **Eval batch:** synthetic induction batch (`build_induction_batch`, RNG seed 42): 2000 sequences (500 for Phase 1 spectral, 2000 for Phase 2 mech-interp and Phase 3 ablation), seq=256, vocab range `[100, 10000)`, structure `[filler] A B [more filler] A` with the target B prediction at the second-A position.
- **Inference:** MPS, fp16, batch=4. Per-head attention probe hooks `model.model.layers[L].self_attn.o_proj` input and reshapes `(B,T,hidden)→(B,T,n_head,head_dim)`.

## Phase 1: per-head PR trajectory and integral ranking

For each (layer, head) and each of the 10 revisions, compute participation ratio (`exp(H(p))` over normalized squared singular values) of the per-head attention output at the second-A position across the 500-example batch. Then compute the **integral** of `max(PR − 1, 0)` over `log(tokens)`.

### Aggregate trajectory

| Step | Tokens (B) | PR min | PR max | PR mean |
|------|------------|--------|--------|---------|
| 5000 | 20 | 1.39 | 84.02 | **22.18** |
| 10000 | 41 | 1.36 | 80.35 | 18.52 |
| 25000 | 104 | 1.13 | 79.50 | 18.35 |
| 50000 | 209 | 1.15 | 79.35 | 15.43 |
| 100000 | 419 | 1.06 | 87.79 | 14.60 |
| 200000 | 838 | 1.04 | 84.50 | 12.34 |
| 400000 | 1677 | 1.04 | 86.50 | 12.73 |
| 600000 | 2516 | 1.02 | 88.06 | 12.23 |
| 800000 | 3355 | 1.02 | 86.45 | 11.70 |
| 1220000 | 5117 | 1.01 | 92.55 | **10.90** |

Mean PR halves over training (22→11); max stays saturated (80–93); min approaches 1. Classic specialization-during-training.

### Top-10 heads by integral (final ranking)

| Rank | (L,H) | Integral | Trajectory pattern |
|------|-------|----------|---------------------|
| 1 | L1H5 | 423.9 | `67→84→…→75` (saturated throughout) |
| 2 | L0H14 | 417.3 | `72→87→…→83` (saturated) |
| 3 | L0H15 | 400.7 | `68→80→86→93` (rises to saturation) |
| 4 | L2H7 | 375.9 | `75→70→41` (saturated then drops) |
| 5 | L2H8 | 349.4 | `84→53→51` (drops mid-training) |
| 6 | L1H3 | 339.2 | `76→60→43` (drops monotonically) |
| 7 | L2H9 | 338.8 | `79→62→35` (drops) |
| 8 | L6H9 | 332.0 | `59→66→46` (peaks then drops) |
| 9 | L1H9 | 319.0 | `45→70→67` (rises and stays) |
| 10 | L4H12 | 315.4 | `77→48→41` (drops) |

Notable outlier: **L1H12** (rank 11): `30.8→80.3→90.9` — a *riser* whose PR climbs from 30 to 91. Late-emerging high-PR head; possibly an induction-mechanism emergence candidate but in the saved Phase 2 classification ends up first-token-classified (115× BOS vs only 24× induction).

### K selection

| Method | K |
|--------|---|
| Algorithmic elbow (max second derivative on sorted integral) | 9 |
| Writeup heuristic (17–19% of total heads) | 43–48 |

The algorithmic elbow severely undershoots because OLMoE's sorted integral distribution decays smoothly rather than dropping sharply. Phase 2's precision-at-K curve is decisive: precision stays 0.93–1.00 through K=45 (see below), confirming the writeup heuristic's K=45 is the right cutoff.

## Phase 2: mech-interp classification at final revision

Load `main` with `attn_implementation="eager"`; run the full 2000-example batch with `output_attentions=True`; for each (L,H) collect attention from the second-A position to all positions. Compute mean attention to canonical target positions and selectivity = mean_attn / baseline (baseline = mean attention to 50 random "other" positions, avoiding the canonical targets).

Classify each head by highest selectivity ≥30× into one of six classes:
**induction** (attention to position after first A) · **previous-token** (T−2) · **duplicate-token** (position of first A) · **first-token** (position 0) · **self** (T−1) · **local** (mean T−2 to T−5).

### Precision-at-K (top heads by integral)

| K | %heads | OLMoE | Pythia 410M (writeup) |
|---|--------|-------|------------------------|
| 5 | 2.0% | **1.00** (5/5) | 1.00 |
| 10 | 3.9% | **1.00** (10/10) | 1.00 |
| 15 | 5.9% | 0.93 (14/15) | 0.93 |
| 20 | 7.8% | 0.95 (19/20) | — |
| 30 | 11.7% | 0.97 (29/30) | 0.90 |
| 45 | 17.6% | **0.96** (43/45) | — |
| 60 | 23.4% | 0.95 (57/60) | — |
| 80 | 31.2% | **0.96** (77/80) | **0.81** |
| 100 | 39.1% | 0.93 (93/100) | — |
| 130 | 50.8% | 0.91 (118/130) | — |
| 160 | 62.5% | 0.88 (141/160) | — |
| 200 | 78.1% | 0.87 (174/200) | — |
| 256 | 100% | 0.80 (205/256) | — |

OLMoE tracks Pythia 410M through K=15 then *flattens above* the Pythia curve. Precision stays >0.90 through K=130. The flatness is BOS-driven (see below).

### Class breakdown of top-45 heads

| Class | Count |
|-------|------:|
| **first-token (BOS)** | **23** |
| previous-token | 10 |
| self | 6 |
| unclassified | 2 |
| induction | 2 |
| duplicate-token | 2 |

The two induction-classified heads in top-45: `L7H0` (induction-selectivity 56×, ranked ~38 by integral) and `L9H8` (selectivity 72×, ranked ~41).

### Whole-model class breakdown (all 256 heads)

| Class | Count | % of 256 |
|-------|------:|---------:|
| **first-token (BOS)** | **174** | **68.0%** |
| unclassified | 51 | 19.9% |
| previous-token | 12 | 4.7% |
| self | 11 | 4.3% |
| induction | 4 | 1.6% |
| duplicate-token | 4 | 1.6% |

**174 of 256 heads (68%) are BOS-attending** at the ≥30× threshold. By contrast Pythia 410M's top-80 had 20 first-token heads (the writeup's largest reported model). OLMoE is anomalously BOS-dominated.

### Induction-selectivity distribution

| Selectivity threshold | Heads at that threshold |
|----------------------|------------------------:|
| ≥ 10× | 10 |
| ≥ 30× | 4 |
| ≥ 50× | **4** |
| ≥ 100× | 3 |
| ≥ 200× | 0 |

Only 4 heads have induction-selectivity ≥30×: `L5H10, L7H0, L9H8, L12H14`. The induction circuit is genuinely 4-head wide. Two of those four (`L5H10, L12H14`) are NOT in top-45 by integral.

## Phase 3: causal ablation

Forward-pre-hook on `self_attn.o_proj` to zero per-head slices of the input tensor. Run the 2000-example induction batch on each ablation condition; measure cross-entropy loss / top-1 / top-5 induction accuracy / mean logit of target B.

### K=45 results (writeup-heuristic K)

| Condition | n_ablated | Loss | Top-1 | Top-5 | logit_B |
|-----------|----------:|-----:|------:|------:|--------:|
| baseline | 0 | 8.52 | **4.80%** | 10.75% | +2.15 |
| ablate_top45_by_integral | 45 | 11.60 | **0.00%** | 0.00% | −0.45 |
| ablate_matched_random_same_layers | 45 | 10.79 | 0.10% | 0.35% | +0.16 |
| ablate_full_spec_layers (upper bound) | 240 | 13.00 | 0.00% | 0.00% | −0.81 |
| **ablate_induction_screen ≥50×** | **4** | 10.64 | **0.00%** | 0.25% | +0.01 |
| ablate L7H0 only | 1 | 9.08 | 1.85% | 4.95% | +1.58 |
| ablate L9H8 only | 1 | 9.88 | 0.45% | 1.25% | +0.75 |

### K=6 results (Karpathy/Pythia canonical ablation K)

| Condition | n_ablated | Loss | Top-1 | Top-5 | logit_B |
|-----------|----------:|-----:|------:|------:|--------:|
| baseline | 0 | 8.52 | 4.80% | 10.75% | +2.15 |
| ablate_top6_by_integral | 6 | 8.92 | 3.20% | 6.90% | +1.74 |
| **ablate_matched_random_same_layers** | 6 | **8.34** | **6.80%** | **13.15%** | +2.34 |
| ablate_full_spec_layers (upper bound) | 48 | 8.53 | 4.70% | 10.35% | +2.20 |
| ablate_induction_screen ≥50× | 4 | 10.64 | 0.00% | 0.25% | +0.01 |
| ablate L7H0 only | 1 | 9.08 | 1.85% | 4.95% | +1.58 |
| ablate L9H8 only | 1 | 9.88 | 0.45% | 1.25% | +0.75 |

### Five claims from the ablation data

**1. Spectral picks vs matched-random separates at K=45 by loss differential.** Δloss(spec_picks) = +3.08; Δloss(matched_random) = +2.27. Spec_picks do ~36% more loss damage than random in the same layers — the residual signal that survives an over-aggressive K. Both crush argmax to ~0% because ablating ~7 of 16 heads in some layers (e.g., L1, L2 have 7 picks each) destroys those layers' attention capacity regardless.

**2. At K=6, spec_picks are NOT induction-targeted.** Top-6 picks are all in L0/L1/L2 (the highest-integral BOS/saturated heads). Ablating them drops induction top-1 from 4.80% → 3.20% (~33% drop), which is mild general-LM disruption. Ablating *random* 6 heads in those layers *increases* induction top-1 to 6.80% (random ablation slightly removes interference). Ablating *all* 48 heads in L0+L1+L2 returns top-1 to 4.70% (essentially baseline). **L0–L2 is not where induction lives** in OLMoE.

**3. Distribution wins (Pythia 410M pattern, replicated exactly).** The 4-head capability-specific screen (induction-selectivity ≥50×) tanks induction top-1 from 4.80% → 0.00%. Two of the four heads (L5H10, L12H14) were NOT in top-45 by integral — integral ranking missed them. The all-head screen is the surgical tool, integral ranking is not.

**4. Single induction heads are individually load-bearing.** Ablating L7H0 alone drops top-1 by 61% (4.80 → 1.85); L9H8 alone drops it by 91% (4.80 → 0.45). Neither single head is sufficient to fully tank induction (the 4-head screen is needed for 0%), but each one carries large fractional load.

**5. K=6 fails as a matched-random ablation control on OLMoE.** In Karpathy 124M (writeup), top-6 spectral picks happened to contain 3 induction heads → K=6 ablation killed induction (16.1% → 0.85%) vs matched-random's 10.6%. In OLMoE, top-6 contains zero induction heads → K=6 ablation merely perturbs general LM, and matched_random can even *help*. K=6 is the right ablation-control K only when spec_picks happen to contain capability heads, which is model-dependent.

## Phase 3b: Natural-text causal ablation of the 4-head circuit

The K=45/K=6 synthetic ablations above are on the synthetic random-token batch. To test whether the synthetic-identified circuit extends to natural-text induction, we ran the same Pythia-style ablation against the 2000-example natural-text batch (per-example query positions, per-example target tokens S).

| Condition | n_ablated | Loss | Top-1 | Top-5 | logit_S |
|-----------|----------:|-----:|------:|------:|--------:|
| baseline (natural text) | 0 | 6.55 | **26.65%** | 39.45% | +10.29 |
| **ablate_4_induction_heads** (L5H10, L7H0, L9H8, L12H14) | 4 | 6.63 | **25.15%** | 38.55% | +10.02 |
| ablate_matched_random_same_layers ({L5:H7, L7:H7, L9:H13, L12:H7}) | 4 | 6.59 | 26.45% | 39.10% | +10.27 |
| ablate_all_heads_in_those_4_layers (upper bound) | 64 | 5.87 | 24.95% | 39.20% | +9.70 |
| ablate L5H10 only | 1 | 6.55 | 26.70% | 39.15% | +10.28 |
| ablate L7H0 only | 1 | 6.57 | 26.65% | 39.20% | +10.27 |
| ablate L9H8 only | 1 | 6.58 | 26.10% | 38.95% | +10.21 |
| ablate L12H14 only | 1 | 6.57 | 26.20% | 39.20% | +10.19 |

**The circuit IS confirmed on natural text:**

| Metric | 4-head circuit Δ | matched_random Δ | differential |
|--------|------------------:|------------------:|--------------:|
| Top-1 | −1.50pp | −0.20pp | **7.5×** |
| logit_S | −0.27 | −0.02 | **13.5×** |
| Loss | +0.07 | +0.04 | 1.8× |

The same 4 heads spectrally identified on synthetic produce a clean differential vs matched-random on natural text. Effect-size is much smaller than synthetic (1.5pp natural-text vs 4.80pp synthetic, i.e., ~5.6% vs 100% relative drop) because natural-text induction has many redundant backup paths.

**Individual-head ablations are nearly null on natural text but were large on synthetic:**

| Head | Synthetic Δ top-1 | Natural-text Δ top-1 |
|------|------------------:|---------------------:|
| L7H0 alone | **−2.95pp (61% rel)** | 0pp |
| L9H8 alone | **−4.35pp (91% rel)** | −0.55pp |
| L12H14 alone | (not tested) | −0.45pp |
| L5H10 alone | (not tested) | +0.05pp |
| **All 4 together** | **−4.80pp (100% rel)** | **−1.50pp** |

On synthetic random-token contexts, the model has no backup paths — every contributing head is critical, single-head ablations tank acc by 60–90%. On natural text the model has many backup paths (n-gram statistics, semantic priors, redundant attention heads doing partial induction), and individual head removals are absorbed. The 4-head GROUP effect (1.5pp) exceeds the sum of individuals (≈1.0pp), so the heads compose synergistically — each contributes a slightly different piece of the induction mechanism.

**Upper-bound caveat.** Ablating all 64 heads in layers {5,7,9,12} *reduces* loss (6.55 → 5.87) while modestly reducing top-1 (26.65% → 24.95%). The loss-drops-acc-drops paradox is a softmax-normalization artifact: massive ablation flattens the model's prediction distribution, which mechanically raises probability mass on non-modal targets (~73% of the time S is not the model's top-1) and reduces cross-entropy on those even though raw logit_S also dropped. This does not say "remove these layers and have a better model" — it says the model's natural predictions are confidently miscalibrated for this specific eval metric. The interpretable signal is the 4-head-circuit-vs-matched-random differential, not the absolute loss value.

## OLMoE's BOS dominance — confirmed on natural text

The most architecturally distinctive feature of OLMoE in this analysis is the extreme prevalence of first-token-attending heads. **A natural-text re-classification was run specifically to test whether this is an architectural property or a synthetic-batch artifact. Result: it's architectural.**

### Natural-text setup

A 2000-example batch built from OpenWebText (cached locally; web-text from a similar distribution to OLMoE's DCLM training data). For each context: find a token T that appears at position `i`, followed by some token S at position `i+1`, where T reappears later at position `k` (the latest occurrence in the 256-token window). Use `k` as the per-example query position; canonical target positions are derived per example (`induction = i+1`, `duplicate = i`, `prev = k-1`, etc.). Median first-T position is 2, median query position is 186; the gap is large.

Same OLMoE-1B-7B-0924 `main` model loaded with eager attention; same selectivity-≥30× classification threshold; same `compute_pr` helper. The only difference vs Phase 2 is the eval batch.

### Whole-model class breakdown, synthetic vs natural

| Class | Synthetic (Phase 2) | Natural-text |
|-------|---------------------:|---------------:|
| **first-token (BOS)** | **174 (68.0%)** | **188 (73.4%)** |
| unclassified | 51 (19.9%) | 48 (18.8%) |
| self | 11 (4.3%) | 12 (4.7%) |
| previous-token | 12 (4.7%) | 7 (2.7%) |
| induction | 4 (1.6%) | 0 |
| duplicate-token | 4 (1.6%) | 1 (0.4%) |

**Natural text *amplifies* BOS dominance** (73.4% vs 68.0%) rather than reducing it. The 5pp increase rules out the "synthetic-batch attractor" explanation: if BOS attention were just "default attention when no content is interesting," natural text (which has content) should reduce it. Instead, BOS goes up. The architectural BOS-attractor interpretation is the surviving one.

### Cross-model comparison — corrected whole-model BOS fractions

**Important correction to the original `induction_heads_writeup.md` reporting.** The Pythia "first-token heads / total" numbers in that writeup (e.g., 5.2% at Pythia 410M) are *first-token-classified AND integral-top-K, as percent of total heads* — NOT the whole-model BOS fraction. We re-ran mech-interp classification (fp32 to avoid baseline underflow on smaller models) on Pythia 160M / 410M / 1B with `all_head_selectivity` saved, plus matching natural-text mech-interp on all five models. Final 5-model curve:

| Model | Data | Arch | Heads | BOS % synth | BOS % natural | Δ (nat-synth) |
|-------|------|------|------:|------------:|--------------:|--------------:|
| Pythia 160M | Pile | dense | 144 | 43.1% | **15.3%** | **−27.8pp** |
| Pythia 410M | Pile | dense | 384 | 58.1% | 69.0% | +10.9pp |
| Pythia 1B | Pile | dense | 128 | 53.9% | 68.0% | +14.1pp |
| OLMoE 1B-7B | DCLM | MoE | 256 | 68.0% | 73.4% | +5.5pp |
| OLMo 1B-0724-hf | DCLM | dense | 256 | **78.1%** | **84.0%** | +5.9pp |

**Four readings:**

1. **The corrected scale curve in Pythia: 43% → 58% (160M → 410M) on synthetic; nearly saturates at 1B (54%).** The original writeup's "Pythia ~5% BOS" was integral-top-K-AND-BOS-classified as percent of total — not the whole-model fraction. With proper whole-model counting, Pythia is already ~half BOS-class at 160M.

2. **DCLM data adds ~20pp** at the same scale+arch: OLMo 1B dense DCLM (78%) vs Pythia 1B dense Pile (54%). Pure data effect at fixed architecture.

3. **MoE *reduces* BOS by ~10pp** vs dense at same scale+data: OLMoE (68%) vs OLMo (78%) within DCLM. MoE doesn't cause attention sinks; if anything, suppresses them.

4. **Pythia 160M is the anomaly: natural text REDUCES its BOS** (43% synth → 15% natural, −28pp). All other models *amplify* BOS on natural text by 5–14pp. There's a scale transition between 160M and 410M where attention organization flips from "content-driven" (BOS reduced when real content is present) to "sink-dominated" (real text content competes weakly with the BOS attractor). **Pythia 410M's natural-text BOS (69%) is essentially identical to OLMoE's (73%)**, decisively refuting the "OLMoE is unusually BOS-dominated vs transformers" framing.

Consistent with the attention-sink phenomenon (Xiao et al., StreamingLLM). The methodology is unaffected — the all-head capability screen identifies a 3–4 head induction circuit in every model tested, regardless of how BOS-dominant the rest of the attention is.

### Per-layer BOS structure — UNIVERSAL L0–L1 zero-BOS pattern

The per-layer breakdown of BOS-classified heads on the synthetic batch (all 5 models):

| Layer | Pythia 160M | Pythia 410M | Pythia 1B | OLMoE | OLMo |
|-------|------------:|------------:|----------:|------:|-----:|
| **L0** | **0%** | **0%** | **0%** | **0%** | **0%** |
| **L1** | **0%** | **0%** | **0%** | **0%** | **0%** |
| L2 | 0% | 0% | 0% | 38% | 88% |
| L3 | 0% | 0% | 0% | 69% | 94% |
| L4 | 83% | 0% | 88% | 75% | 100% |
| L5 | 67% | 0% | 75% | 69% | 94% |
| L6 | 58% | 88% | 75% | 75% | 100% |
| L7+ | 50–67% | 69–100% | 50–88% | 80–100% | 80–100% |

**L0 and L1 have ZERO BOS-classified heads across ALL 5 models.** This is a universal architectural property of decoder-only LMs at 100M+ scale: the first two layers do diverse general-purpose computation (high PR throughout training), and the BOS attractor only kicks in from L2 onward.

**Where the transition happens varies by training:**
- OLMoE / OLMo (DCLM-trained): BOS dominance starts at L2 (early transition — likely data-driven).
- Pythia 160M / 1B: BOS starts at L4.
- Pythia 410M: BOS starts at L6 (the latest transition among all models).

So the L0–L1 *floor* is universal, but the L2 vs L4 vs L6 *onset* is model-dependent. DCLM-trained models push the transition earlier; Pile-trained models push it later. The universal "early layers do general computation" finding generalizes across architectures, vendors, and training data.

### Induction selectivity on natural text — and the mid-sequence control

To distinguish "natural-text induction is genuinely harder" from "BOS-overlap measurement artifact," we ran a control: rebuild the natural-text batch with `first_T_pos ≥ 20` (forcing the induction target to position ≥ 21, well outside the BOS-attractor zone) and re-classify.

If the unfiltered natural-text weakening was a BOS-overlap artifact (BOS-attending heads "absorbing" attention that should have gone to the induction target near position 0), the filtered batch should *recover* induction selectivity toward synthetic levels.

| Threshold | Synthetic | Natural (unfiltered) | Natural (first_T ≥ 20) |
|-----------|----------:|---------------------:|------------------------:|
| ≥10× | 10 | 11 | **6** |
| ≥30× | 4 | 4 | **2** |
| ≥50× | **4** | 1 | 1 |
| ≥100× | 3 | 0 | 0 |
| ≥200× | 0 | 0 | 0 |

**Filtering first_T to mid-sequence makes induction selectivity *weaker*, not stronger.** Opposite of what the BOS-overlap hypothesis predicted. The unfiltered natural-text weakening was NOT primarily a measurement artifact from position overlap.

| Class (whole-model 256 heads) | Synthetic | Natural (unfiltered) | Natural (first_T ≥ 20) |
|--|--:|--:|--:|
| first-token (BOS) | 174 (68.0%) | 188 (73.4%) | **187 (73.0%)** |
| unclassified | 51 (19.9%) | 48 (18.8%) | 48 (18.8%) |
| self | 11 (4.3%) | 12 (4.7%) | 13 (5.1%) |
| previous-token | 12 (4.7%) | 7 (2.7%) | 7 (2.7%) |
| induction | 4 (1.6%) | 0 | 0 |
| duplicate-token | 4 (1.6%) | 1 (0.4%) | 1 (0.4%) |

BOS fraction is 73% with or without the filter — independent of first_T position. The BOS attractor is architectural, not position-dependent.

**The settled interpretation:**

1. **The induction circuit is real and 4 heads wide.** Synthetic-batch ablation of `{L5H10, L7H0, L9H8, L12H14}` (the heads with synthetic induction-selectivity ≥50×) tanks induction acc 4.80% → 0.00%.
2. **Natural-text induction is genuinely harder to detect via attention selectivity.** Position overlap is not the issue. Even with first_T pushed to position 20, only 1 head reaches induction-selectivity ≥50× and 0 reach ≥100×. Real-world next-token prediction has more competing content-relevant targets; the attention mass doesn't cleanly concentrate on the induction-target position the way it does on the synthetic random-token batch.
3. **BOS dominance is architectural and position-independent.** ~73% of heads are BOS-class regardless of where the induction targets sit.

This complicates the methodology claim moderately: the spectral → mech-interp → causal pipeline works on synthetic batches; the mech-interp *selectivity numbers* themselves don't transfer cleanly to natural text, but the underlying capability heads (validated by synthetic ablation) are the same. Natural-text validation of these heads would require either (a) a natural-text causal ablation on natural-text induction loss, or (b) a richer behavioral signal than attention-target selectivity (e.g., contribution to logit-of-S at the second-T position).

### What this changes about the OLMoE framing

The BOS finding moves from "interesting OLMoE artifact" to **"architectural property worth flagging as a primary OLMoE result"**:
- OLMoE has a structural BOS attractor at ~73% of heads (on the eval distribution closest to its training data).
- This is qualitatively different from Pythia / Karpathy where BOS-attending heads were ~5% of the model.
- The capability-circuit-identification methodology still works (4-head induction screen → tanks induction), but the rank-ordered classification is dominated by BOS heads in the way Pythia 410M's was *not*.

Connection to existing literature: this looks like an extreme case of attention sinks. Whether it's specifically MoE-driven (sparse routing creates pressure for a stable default attention target?) or training-data-driven (DCLM marker distribution?) or just an OLMoE training-config consequence is open — answering would require running the same probe on a dense model trained on the same data (or an MoE trained on different data).

### BOS-head structure (per-layer distribution)

The 174–188 BOS-classified heads are **NOT uniformly distributed across layers** — there's a sharp L0–L1 cliff.

| Layer | BOS / 16 (synthetic) | BOS / 16 (natural) | BOS / 16 (mid-seq control) |
|------:|---:|---:|---:|
| **L0** | **0** | **0** | **0** |
| **L1** | **0** | **0** | **0** |
| L2 | 6 | 2 | 1 |
| L3 | 11 | 12 | 13 |
| L4 | 12 | 13 | 12 |
| L5 | 11 | 13 | 13 |
| L6 | 12 | 12 | 12 |
| L7 | 12 | 14 | 14 |
| L8 | 14 | 14 | 14 |
| L9 | 14 | 15 | 15 |
| L10 | 14 | 15 | 15 |
| L11 | 15 | 15 | 15 |
| L12 | 13 | 16 | 16 |
| L13 | 16 | 16 | 16 |
| L14 | 15 | 16 | 16 |
| L15 | 9 | 15 | 15 |
| TOTAL | 174 / 256 | 188 / 256 | 187 / 256 |

**L0 and L1 have zero BOS heads.** L2 is a transition. From L3 onward, the vast majority of heads (14–16 of 16) are BOS-class. The integral-ranking top-10 (`L1H5, L0H14, L0H15, L2H7, L2H8, L1H3, ...`) are exactly the non-BOS heads concentrated in these first two layers.

This gives a cleaner architectural picture than "OLMoE is BOS-dominated":
- **L0–L1 do diverse, high-PR general computation** (the "saturated" heads we found in Phase 1's top-integral set).
- **L3–L15 default-attend to BOS** for most heads, with a small minority doing other capabilities (induction, prev-token, self, etc.).

### PR trajectory by class — BOS heads SPECIALIZE during training

Classifying heads by their final-revision mech-interp class, then looking at their PR trajectory over Phase 1's 10 revisions, separates the classes cleanly:

| Class | n | start PR (step 5K) | end PR (step 1.22M) | direction |
|-------|--:|-------------------:|--------------------:|-----------|
| first-token (BOS) | 174 | 20.6 | **8.5** | drops 2.4× |
| previous-token | 12 | 56.8 | 33.9 | drops 1.7× |
| **self** | 11 | 37.4 | **45.7** | **only rising class** |
| induction | 4 | 38.9 | 26.3 | drops |
| duplicate-token | 4 | 39.0 | 18.6 | drops |
| unclassified | 51 | 13.5 | 4.3 | drops sharply |

BOS heads don't start as BOS-attending and stay there — they *start at moderate PR ≈ 21* and *collapse to PR ≈ 9 during training*. Attention-sink behavior emerges over training, not at initialization. Self-attention heads are the only class that *gains* PR. Unclassified heads decay to the lowest PR cluster (4.3), suggesting they're weakly content-dependent diffuse heads with no specific capability.

Figure: `olmoe_tier1_pr_trajectory_by_class.png`.

### The 4 induction heads on natural text: BOS-class but with real induction signal

The Phase-2 mech-interp at the final revision found 0 induction-classified heads on natural text (vs 2 on synthetic). The natural-text re-classification of the 4 capability-screen induction heads explains why:

| Head | Synth `best_class` (sel) | Natural `best_class` (sel; **induction-sel**) | Mid-seq `best_class` (sel; **induction-sel**) |
|------|---|---|---|
| L5H10 | induction (101.8×) | **first-token** (95.4×; ind=**25.9×**) | first-token (156.0×; ind=21.3×) |
| L7H0 | induction (171.7×) | **first-token** (251.8×; ind=**33.4×**) | first-token (453.8×; ind=34.1×) |
| L9H8 | induction (157.9×) | **first-token** (354.0×; ind=**71.1×**) | first-token (579.2×; ind=67.0×) |
| L12H14 | induction (61.6×) | **first-token** (224.8×; ind=**38.0×**) | first-token (262.7×; ind=27.9×) |

**All 4 heads are doing both things on natural text:** strong BOS attention AND non-trivial induction attention (selectivity 25–71×, all above the 30× threshold except L5H10). The "best class" classification puts them in `first-token` because BOS selectivity wins numerically, but the induction signal is present underneath.

**This is a real methodology critique on architectures with strong attention sinks:** "highest-selectivity class" classification doesn't surface capability heads correctly because BOS class wins for too many heads. **The all-head capability-specific screen** (Phase 3's `induction-selectivity ≥50×`) is the robust approach — it measures capability-X in isolation, not relative to BOS. On a model like OLMoE this distinction matters; on Pythia / Karpathy (where BOS class is small) it doesn't.

### Dense-baseline control: `allenai/OLMo-1B-0724-hf`

To test whether the BOS attractor is MoE-driven, data-driven, or training-recipe-driven, we ran the **same 3-step pipeline on the dense AllenAI model** with near-identical architecture to OLMoE except for the FFN:

| Property | OLMoE-1B-7B-0924 | OLMo-1B-0724-hf |
|----------|-------------------|------------------|
| hidden_size | 2048 | 2048 |
| num_hidden_layers | 16 | 16 |
| num_attention_heads | 16 | 16 |
| vocab_size | 50304 | 50304 |
| max_position_embeddings | 4096 | 4096 |
| tokenizer | gpt-neox-20b | gpt-neox-20b |
| FFN | MoE 64 experts top-8 | dense |
| trained on | DCLM (~5T tokens) | DCLM/similar (~3T tokens) |
| released | 2024-09 | 2024-07 |

Phase 1 (10 log-spaced revisions, step1000-tokens2B → step1454000-tokens3048B) + Phase 2 (synthetic and natural-text mech-interp at the final revision) + Phase 3 (K=45 / K=6 / capability-screen ablations on both synthetic and natural-text batches).

#### Whole-model class breakdown, OLMoE vs OLMo (synthetic batch)

| Class | OLMoE | OLMo dense |
|-------|------:|-----------:|
| **first-token (BOS)** | 174 (68.0%) | **200 (78.1%)** |
| unclassified | 51 (19.9%) | 26 (10.2%) |
| previous-token | 12 (4.7%) | 15 (5.9%) |
| self | 11 (4.3%) | 11 (4.3%) |
| induction | 4 (1.6%) | 0 |
| duplicate-token | 4 (1.6%) | 4 (1.6%) |

#### Natural-text class breakdown

| Class | OLMoE | OLMo dense |
|-------|------:|-----------:|
| **first-token (BOS)** | 188 (73.4%) | **215 (84.0%)** |
| unclassified | 48 (18.8%) | 23 (9.0%) |
| self | 12 (4.7%) | 11 (4.3%) |
| previous-token | 7 (2.7%) | 7 (2.7%) |

**Dense OLMo has MORE BOS-class heads than MoE OLMoE on both synthetic (78.1% vs 68.0%) and natural-text (84.0% vs 73.4%) batches.** The attention-sink dominance is *not* MoE-driven — it's even stronger in the dense variant.

#### OLMo induction circuit + ablation

OLMo's all-head capability screen at induction-selectivity ≥50× finds **3 heads**: `L2H11 (ind=134×), L4H12 (ind=117×), L12H8 (ind=56×)` — all best-classified as BOS (same pattern as OLMoE's 4 heads).

**Synthetic K=45 ablation (n=2000):**

| Condition | n_ablated | Loss | Top-1 | logit_B |
|-----------|----------:|-----:|------:|--------:|
| baseline | 0 | 9.48 | **1.00%** | +2.56 |
| ablate_top45_by_integral | 45 | 12.51 | 0.00% | +0.57 |
| ablate_matched_random | ≤45 (capped) | 10.66 | 1.35% | +2.09 |
| ablate_full_spec_layers (upper bound) | 240 | 11.44 | 0.00% | +0.67 |
| **ablate_induction_screen ≥50× (3 heads)** | **3** | 10.50 | **0.05%** | +1.58 |

**Natural-text ablation (n=2000):**

| Condition | n_ablated | Loss | Top-1 | logit_S |
|-----------|----------:|-----:|------:|--------:|
| baseline (natural) | 0 | 6.03 | **27.75%** | +11.44 |
| **ablate_induction_circuit** (L2H11, L4H12, L12H8) | 3 | 6.13 | **26.35%** | +11.10 |
| ablate_matched_random | 3 | 5.98 | 28.05% | +11.45 |
| ablate_all_heads_in_those_3_layers | 48 | 5.09 | 26.00% | +10.13 |
| ablate L2H11 only | 1 | 6.04 | 27.55% | +11.39 |
| ablate L4H12 only | 1 | 6.06 | 27.30% | +11.34 |
| ablate L12H8 only | 1 | 6.07 | 27.40% | +11.32 |

#### Five observations from the dense baseline

**1. MoE EXONERATED for the BOS attractor.** Same probe, same methodology, same data family (DCLM), and dense OLMo has MORE BOS heads. So the architecture choice (MoE vs dense) does *not* cause the attention sink. The cause is data-driven (DCLM) or training-recipe-driven (AllenAI's pretraining setup), and we can't disambiguate those two further without (a) a dense LM on DCLM from a different vendor, or (b) an AllenAI model on different data.

**2. Methodology fully transfers to OLMo dense.** Same 3-step pipeline identifies a 3-head induction circuit (1 fewer than OLMoE's 4) that tanks synthetic induction (1.00% → 0.05%) and produces a clean differential vs matched-random on both synthetic and natural-text ablations. Different layer placement (L2/L4/L12 vs OLMoE's L5/L7/L9/L12) — OLMo's induction circuit is *earlier* in the model.

**3. OLMo dense has 5× lower synthetic induction baseline than OLMoE** (1.00% vs 4.80%). On natural text they're comparable (27.75% vs 26.65%). Most likely cause: OLMo trained on ~3T tokens vs OLMoE's ~5T (single-seed scaling effect), or MoE expressiveness specifically helps synthetic random-token sequences. Doesn't affect the methodology — circuit identification works in both regimes.

**4. K=45 spec_picks-vs-matched-random is CLEAN on OLMo dense** (unlike OLMoE where matched-random also tanked). Reason: OLMo's top-45 picks concentrate in L0 (14 of 16 heads), leaving only 2 non-overlap heads in L0 for matched_random to choose from. The matched-random ablation removes far fewer L0 heads than spec_picks, producing the clean baseline-preserving control we'd want. (The script caps matched-random per layer at eligible count when this happens.)

**5. On natural text, induction-circuit ablation differential is ~5× over matched-random (−1.40pp vs +0.30pp).** Same pattern as OLMoE (−1.50pp vs −0.20pp). Individual head ablations are again null on natural text (each head drops top-1 by 0.20–0.45pp), consistent with the OLMoE finding that natural-text induction is redundant.

If we exclude first-token-classified heads from the integral ranking and recompute precision-at-K on the 82 remaining heads:

| K | Original prec (with BOS) | Non-BOS prec |
|---|--------------------------|--------------|
| 5 | 1.00 | 1.00 |
| 15 | 0.93 | 0.93 |
| 30 | 0.97 | 0.90 |
| 45 | 0.96 | **0.64** |
| 80 | 0.96 | **0.39** |

The headline 96%-at-K=45 was BOS-driven. Non-BOS, OLMoE has 31 classified heads of 82 non-BOS = **37.8% of non-BOS heads = 12.1% of total**. Pythia's writeup conserved fraction is 17–19% of total. **Excluding the attention-sink class, OLMoE has *fewer* specialized non-sink heads than transformers, not more.** The inflated precision-at-K curve and the "more specialized fraction than Pythia" claim are both BOS-driven; the real non-sink specialization is below the transformer baseline.

## What this is and what it isn't

**This is** the full Pythia 410M probe-circuit pipeline (spectral → mech-interp → causal) applied to OLMoE-1B-7B-0924, with all three steps following the writeup's exact methodology — integral ranking, the 17–19% K heuristic, the ≥30× capability-classification threshold, matched-random control, and the all-head capability screen for the capability-specific causal claim. The induction circuit at OLMoE scale is 4 heads wide, located in L5/L7/L9/L12, and 2 of those 4 (L5H10, L12H14) lie outside the top-45 by integral — exactly the "distribution wins" pattern documented for Pythia 410M.

**This is NOT** a per-expert FFN PR result. The MoE-specific axis (per-expert PR over training trajectory) has not been computed here. Earlier exploratory work on the snapshot per-expert PR was discarded as not part of the documented methodology; whether it is a valid axis under the methodology requires defining what "behavioral emergence" the per-expert spectral signal would track. This is open.

**This is NOT** a result on natural-text *induction circuit ablation*. The natural-text re-classification was run (and answered the BOS-vs-artifact question — see above), but the causal-verification ablations (Phase 3) were all done on synthetic. A natural-text ablation re-run would tighten the circuit-identification claim further but is not done here.

## Caveats

- **Single seed (RNG=42 for batch construction).** The induction batch is reproducible but does not characterize batch-to-batch variability of the precision-at-K curve.
- **n=500 in Phase 1, n=2000 in Phases 2 and 3.** The integral ranking is computed at n=500, which is below the writeup's Pythia n=2000 standard. Re-running Phase 1 at n=2000 would slightly tighten the trajectory; the qualitative pattern (mean PR 22→11, max ~80–93) is unlikely to change.
- **Coarse ablation.** Zeroing the full per-head slice in `o_proj`'s input. Mean-ablation, per-token ablation, and activation-patching variants would give finer-grained claims.
- **No second induction-eval check.** All ablation effect sizes are measured on the same 2000-example batch used to compute the selectivity matrix. Held-out batches would prevent the rare case where the same examples drive both selectivity and ablation result.

## Open questions raised but not answered

- ~~Is OLMoE's BOS fraction a model property or a synthetic-batch artifact?~~ **Resolved: architectural.** Natural-text re-classification gives 73.4% (UP from 68% synthetic). Mid-seq control gives 73.0%. Position-independent.
- ~~Why does natural-text induction selectivity weaken? BOS-overlap or genuinely harder task?~~ **Resolved: harder task, not position artifact.** Mid-seq control reduced induction-selective count further. The 4 capability heads do produce natural-text induction selectivity (25–71×), they're just classified as BOS because BOS dominates.
- ~~Why is precision-at-K flatter on OLMoE than Pythia?~~ **Resolved: BOS inflation.** Excluding BOS-classified heads, precision-at-K=45 drops from 96% → 64%, and non-BOS classified fraction is 12.1% of total — below Pythia's 17–19%. OLMoE has FEWER non-sink specialized heads than transformers, not more.
- ~~Is the BOS attractor MoE-specific, data-specific, or training-config-specific?~~ **Mostly resolved: BOS is broad across decoder-only LMs at 100M+ scale; MoE reduces vs dense; DCLM adds ~20pp vs Pile.** Whole-model BOS fractions span 43% (Pythia 160M dense Pile) → 78% (OLMo 1B dense DCLM). The prior "OLMoE has anomalously high BOS" claim came from comparing OLMoE whole-model (68%) against Pythia *integral-top-K* (5%), which was an apples-to-oranges comparison. With apples-to-apples whole-model comparison: scale and DCLM data each contribute ~10–20pp to BOS, and MoE *reduces* BOS by ~10pp vs dense at same scale+data. Still open: whether the DCLM effect is data per se or training-recipe (AllenAI pretraining setup); would need a DCLM-trained LM from a different vendor to fully disambiguate.
- When during training do the BOS heads emerge? Phase 1 PR-trajectory analysis shows BOS-classified heads collapse from PR≈21 to PR≈9, suggesting *progressive* attention-sink formation rather than initialization-driven. A per-revision mech-interp re-classification would resolve emergence timing more directly. **Open (would be the next phase).**
- Does per-expert FFN PR have a behavioral correlate at which it could be validated as a probe-circuit axis? Exploratory snapshot showed an L15 expert-rank collapse, but without a behavioral anchor it stays descriptive. **Open.**
- ~~Natural-text causal ablation of the 4-head induction screen — does ablating those same 4 heads degrade *natural-text* induction loss?~~ **Resolved: yes, with redundancy.** 4-head ablation on n=2000 natural batch drops top-1 by 1.50pp (26.65→25.15%) and logit_S by 0.27, vs matched-random's 0.20pp / 0.02 — **7.5× differential** on top-1, 13.5× on logit_S. Effect size much smaller than synthetic (100% relative drop → 5.6%) because natural-text induction has many redundant backup paths; individual single-head ablations on natural text drop top-1 by 0–0.55pp (vs 60–90% on synthetic). The 4-head group effect (1.5pp) exceeds sum of individuals (~1.0pp), so the heads compose synergistically. The circuit is real on both regimes; synthetic just exposes head-criticality that natural-text redundancy hides.

## Artifacts produced

Scripts (worktree root):
- `mamba2_per_head.py` — contains `build_induction_batch` + `compute_pr` (verbatim from `analyses/induction_heads_per_head_124m.py`)
- `olmoe_per_head.py` — Phase 1 per-head PR per revision
- `olmoe_mechinterp.py` — Phase 2 mech-interp classification (synthetic batch)
- `build_natural_induction_batch.py` — builds OpenWebText induction batch with per-example positions; `--min-first-T-pos` filter for the mid-sequence control
- `olmoe_mechinterp_naturaltext.py` — Phase 2 re-classification on natural-text batch (per-example query and target positions)
- `olmoe_ablation.py` — Phase 3 causal ablation on synthetic batch (K=45 and K=6 runs)
- `olmoe_ablation_naturaltext.py` — Phase 3b natural-text causal ablation (per-example query / target)
- `olmoe_tier1_analysis.py` — non-BOS precision-at-K, BOS layer distribution, PR-trajectory-by-class, 4-head cross-classification
- `olmoe_baseline.py` — Step 0 baseline-induction-acc sanity check
- `bench_olmoe_mps.py` — MPS forward-pass timing

Data:
- `olmoe_phase1_trajectory.json` — per-head PR per revision (10 revisions × 256 heads)
- `olmoe_phase1_features.json` — integral / spread / final_pr / max_pr / min_pr per (L,H)
- `olmoe_mechinterp.json` — synthetic top-45 classifications + full 256-head selectivity matrix
- `olmoe_mechinterp_naturaltext.json` — natural-text (unfiltered) top-45 classifications + full 256-head selectivity matrix
- `olmoe_mechinterp_naturaltext_midseq.json` — natural-text (first_T ≥ 20) top-45 classifications + full 256-head selectivity matrix
- `natural_induction_batch.pt` — 2000-example natural-text batch (unfiltered)
- `natural_induction_batch_midseq.pt` — 2000-example natural-text batch with first_T_pos ≥ 20 (mid-sequence control)
- `olmoe_ablation.json` — K=45 ablation results (7 conditions, synthetic)
- `olmoe_ablation_k6.json` — K=6 ablation results (7 conditions, synthetic)
- `olmoe_ablation_naturaltext.json` — 4-head circuit natural-text ablation (8 conditions)
- `olmoe_tier1_summary.json` — non-BOS fraction, BOS layer distribution, induction-head cross-classification
- `olmoe_tier1_pr_trajectory_by_class.png` — PR trajectory by mech-interp class (synthetic-classified)
