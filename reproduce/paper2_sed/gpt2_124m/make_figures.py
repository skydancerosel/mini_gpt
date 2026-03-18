#!/usr/bin/env python3
"""Generate publication-quality figures for the Spectral Edge Dynamics paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- Style ---
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Overview — 4-panel intuitive introduction to SED
# ═══════════════════════════════════════════════════════════════════════
def make_fig1():
    """4-panel overview: spectrum, ratio profile, three-phase, edge-loss link."""

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(6.5, 5.0))

    # ── Panel A: Singular value spectrum (TinyStories, step 2000) ──────
    # From bbp_threshold_test.json (step 2000, seed 42)
    svs = np.array([198.58, 169.08, 122.47, 110.73, 106.22,
                     92.66,  78.30,  61.27,  44.58,  22.79])
    k_vals = np.arange(1, 11)
    kstar = 2
    mp_edge_sv = 64.8        # sqrt(mp_upper_edge ≈ 4197.75)

    colors_a = []
    for i, sv in enumerate(svs):
        if i < kstar:                        # signal modes (k=1,2)
            colors_a.append('#2166ac')
        elif sv > mp_edge_sv:                # above MP edge
            colors_a.append('#999999')
        else:                                # below MP edge
            colors_a.append('#cccccc')

    ax_a.bar(k_vals, svs, color=colors_a, edgecolor='white',
             linewidth=0.5, width=0.7)

    # MP edge
    ax_a.axhline(mp_edge_sv, color='#d6604d', linestyle='--',
                 linewidth=0.7, alpha=0.6)
    ax_a.text(9.5, mp_edge_sv + 5, 'MP edge', fontsize=6.5,
              color='#d6604d', ha='right')

    # Spectral edge gap annotation
    gap_x = 2.55
    ax_a.annotate('', xy=(gap_x, svs[1] - 1), xytext=(gap_x, svs[2] + 1),
                  arrowprops=dict(arrowstyle='<->', color='#d6604d',
                                  lw=1.5, shrinkA=0, shrinkB=0))
    ax_a.text(gap_x + 0.7, (svs[1] + svs[2]) / 2,
              r'$k^*\!=\!2$', fontsize=8, color='#d6604d', va='center')

    ax_a.text(1.5, svs[0] + 10, 'Signal', ha='center', fontsize=7.5,
              color='#2166ac', fontweight='bold')
    ax_a.text(6.5, 95, 'Noise', ha='center', fontsize=7.5,
              color='#666666', fontstyle='italic')

    ax_a.set_xlabel('Index $k$')
    ax_a.set_ylabel(r'$\sigma_k$')
    ax_a.set_xticks(k_vals)
    ax_a.set_ylim(0, 225)
    ax_a.set_title(r'\textbf{(A)} Singular value spectrum', fontsize=10)

    # ── Panel B: Consecutive ratio profile ─────────────────────────────
    ratios = svs[:-1] / svs[1:]               # σ_k / σ_{k+1}
    k_ratio = np.arange(1, 10)

    colors_b = []
    for i in range(9):
        if i == 1:                             # k=2 (spectral edge)
            colors_b.append('#d6604d')
        elif svs[i + 1] >= mp_edge_sv:         # both sides above MP
            colors_b.append('#999999')
        else:
            colors_b.append('#cccccc')

    ax_b.bar(k_ratio, ratios, color=colors_b, edgecolor='white',
             linewidth=0.5, width=0.6)
    ax_b.axhline(1.0, color='black', linewidth=0.4, alpha=0.3)

    # Annotate k*=2 as signal edge
    ax_b.annotate(r'signal edge ($k^*\!=\!2$)',
                  xy=(2, ratios[1] + 0.02), xytext=(4.0, 1.7),
                  fontsize=7.5, color='#d6604d', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.0))

    # Mark below-MP region
    ax_b.axvline(7.5, color='#999999', linestyle=':', linewidth=0.7, alpha=0.5)
    ax_b.text(8.5, 1.0, 'below\nMP', fontsize=6, color='#999999',
              ha='center', va='center')

    ax_b.set_xlabel(r'Index $k$ in $\sigma_k/\sigma_{k+1}$')
    ax_b.set_ylabel(r'$\sigma_k / \sigma_{k+1}$')
    ax_b.set_xticks(k_ratio)
    ax_b.set_ylim(0.9, 2.1)
    ax_b.set_title(r'\textbf{(B)} Ratio profile $\to$ signal rank', fontsize=10)

    # ── Panel C: TinyStories three-phase pattern (seed 42) ─────────────
    with open('/Users/yongzhongxu/mini_gpt/runs/scale_124M/'
              'pilot_124M_b20.95_s42/results/event_study_results.json') as f:
        ts_data = json.load(f)
    steps_c = np.array(ts_data['42']['steps'])
    r23_c = np.array(ts_data['42']['r23'])

    ax_c.plot(steps_c / 1000, r23_c, '-o', color='#2166ac',
              markersize=2, linewidth=1.2)
    ax_c.axvline(5.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
    ax_c.axvline(7.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
    ax_c.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

    ax_c.text(3.5, 1.82, 'Rise', ha='center', fontsize=7.5,
              fontstyle='italic', color='gray')
    ax_c.text(6.0, 1.82, 'Plateau', ha='center', fontsize=7.5,
              fontstyle='italic', color='gray')
    ax_c.text(8.5, 1.82, 'Collapse', ha='center', fontsize=7.5,
              fontstyle='italic', color='gray')

    ax_c.set_xlabel('Training step (×10³)')
    ax_c.set_ylabel(r'$\sigma_2 / \sigma_3$')
    ax_c.set_xlim(2, 10)
    ax_c.set_ylim(1.0, 1.9)
    ax_c.xaxis.set_major_locator(MultipleLocator(1))
    ax_c.set_title(r'\textbf{(C)} Spectral edge dynamics (51M)', fontsize=10)

    # ── Panel D: GPT-2 spectral edge + val-loss ────────────────────────
    with open('combined_pretrain_finetune/sed_results/'
              'gpt2_sed_validation.json') as f:
        gpt2_data = json.load(f)
    bbp = gpt2_data['test1_bbp']
    steps_d = np.array([e['step'] for e in bbp])
    r34_d = np.array([e['r34'] for e in bbp])

    with open('combined_pretrain_finetune/spectral_detail.json') as f:
        sd = json.load(f)
    vl_dict = sd['val_loss']

    color_r = '#2166ac'
    ax_d.plot(steps_d / 1000, r34_d, '-o', color=color_r, markersize=2,
              linewidth=1.2, label=r'$\sigma_3/\sigma_4$')
    ax_d.set_ylabel(r'$\sigma_3/\sigma_4$', color=color_r)
    ax_d.tick_params(axis='y', labelcolor=color_r)

    ax_d2 = ax_d.twinx()
    color_vl = '#d6604d'
    vl_s = [s for s in steps_d if str(s) in vl_dict]
    vl_v = [vl_dict[str(s)] for s in vl_s]
    ax_d2.plot(np.array(vl_s) / 1000, vl_v, '-s', color=color_vl,
               markersize=1.5, linewidth=1.0, alpha=0.8, label='Val-loss')
    ax_d2.set_ylabel('Validation loss', color=color_vl)
    ax_d2.tick_params(axis='y', labelcolor=color_vl)

    ax_d.axvline(17.8, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_d.text(18.0, 1.125, 'Shift', fontsize=7, va='top')

    ax_d.set_xlabel('Training step (×10³)')
    ax_d.set_xlim(15.5, 26)
    ax_d.set_title(r'\textbf{(D)} Edge tracks loss (GPT-2 124M)', fontsize=10)

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2,
                loc='upper right', fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig('fig_overview.pdf')
    print('Saved fig_overview.pdf')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: TinyStories three-phase pattern (3 seeds)
# ═══════════════════════════════════════════════════════════════════════
def make_fig2():
    with open('/Users/yongzhongxu/mini_gpt/runs/scale_124M/pilot_124M_b20.95_s42/results/event_study_results.json') as f:
        data = json.load(f)

    steps = np.array(data['42']['steps'])
    colors = {'42': '#2166ac', '123': '#d6604d', '149': '#4dac26'}
    labels = {'42': 'Seed 42', '123': 'Seed 123', '149': 'Seed 149'}

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for seed in ['42', '123', '149']:
        r23 = np.array(data[seed]['r23'])
        ax.plot(steps / 1000, r23, '-o', color=colors[seed], label=labels[seed],
                markersize=2.5, linewidth=1.2)

    # Phase region shading
    ax.axvspan(2.0, 5.0, alpha=0.05, color='#4dac26')   # Rise
    ax.axvspan(5.0, 7.0, alpha=0.05, color='#ff7f00')   # Plateau
    ax.axvspan(7.0, 10.0, alpha=0.05, color='#d6604d')  # Collapse

    # Phase boundaries
    ax.axvline(5.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axvline(7.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    # Phase labels
    ax.text(3.5, 1.95, 'Rise', ha='center', fontsize=9, fontstyle='italic', color='#2d7d2d')
    ax.text(6.0, 1.95, 'Plateau', ha='center', fontsize=9, fontstyle='italic', color='#cc6600')
    ax.text(8.5, 1.95, 'Collapse', ha='center', fontsize=9, fontstyle='italic', color='#b5302a')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('Training step (×10³)')
    ax.set_ylabel(r'$\sigma_2 / \sigma_3$')
    ax.set_title('TinyStories (51M): Three-Phase Spectral Edge Pattern')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(2, 10)
    ax.set_ylim(1.0, 2.05)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    fig.savefig('fig_tinystories_threephase.pdf')
    print('Saved fig_tinystories_threephase.pdf')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: GPT-2 spectral edge + val-loss (dual axis)
# ═══════════════════════════════════════════════════════════════════════
def make_fig3():
    # Load r34 time series (W=10)
    with open('combined_pretrain_finetune/sed_results/gpt2_sed_validation.json') as f:
        data = json.load(f)

    steps_r34 = [entry['step'] for entry in data['test1_bbp']]
    r34 = [entry['r34'] for entry in data['test1_bbp']]

    # Load val-loss
    with open('combined_pretrain_finetune/spectral_detail.json') as f:
        sd = json.load(f)
    vl_dict = sd['val_loss']

    # Align val-loss to the same steps
    vl_steps = []
    vl_vals = []
    for s in steps_r34:
        key = str(s)
        if key in vl_dict:
            vl_steps.append(s)
            vl_vals.append(vl_dict[key])

    steps_r34 = np.array(steps_r34)
    r34 = np.array(r34)
    vl_steps = np.array(vl_steps)
    vl_vals = np.array(vl_vals)

    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

    # Spectral edge ratio
    color_r34 = '#2166ac'
    ax1.plot(steps_r34 / 1000, r34, '-o', color=color_r34, markersize=2.5,
             linewidth=1.2, label=r'$\sigma_3/\sigma_4$ (W=10)')
    ax1.set_xlabel('Training step (×10³)')
    ax1.set_ylabel(r'$\sigma_3 / \sigma_4$', color=color_r34)
    ax1.tick_params(axis='y', labelcolor=color_r34)

    # Val-loss on secondary axis
    ax2 = ax1.twinx()
    color_vl = '#d6604d'
    ax2.plot(vl_steps / 1000, vl_vals, '-s', color=color_vl, markersize=2,
             linewidth=1.0, alpha=0.8, label='Val-loss')
    ax2.set_ylabel('Validation loss', color=color_vl)
    ax2.tick_params(axis='y', labelcolor=color_vl)

    # Distribution shift marker (darkened)
    ax1.axvline(17.8, color='black', linestyle='--', linewidth=1.4, alpha=0.9)
    ax1.text(17.9, 1.135, 'Distribution\nshift', fontsize=7.5, rotation=90,
             va='top', fontweight='bold')

    # Spectral reorganization label near the spike
    ax1.annotate('spectral\nreorganization',
                 xy=(17.6, 1.133), xytext=(16.2, 1.10),
                 fontsize=6.5, color='#2166ac', fontstyle='italic',
                 arrowprops=dict(arrowstyle='->', color='#2166ac',
                                 lw=0.8, connectionstyle='arc3,rad=0.2'))

    # Overfit onset marker
    ax1.axvline(22.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax1.text(22.3, 1.135, 'Overfit', fontsize=8, rotation=90, va='top', color='gray')

    ax1.set_title(r'GPT-2 124M: Spectral Edge and Val-Loss')
    ax1.set_xlim(15.5, 26)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

    fig.savefig('fig_gpt2_edge_valloss.pdf')
    print('Saved fig_gpt2_edge_valloss.pdf')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: GPT-2 sliding-window correlation (phase-dependent coupling)
# ═══════════════════════════════════════════════════════════════════════
def make_fig4():
    with open('combined_pretrain_finetune/sed_results/gpt2_sed_enhancements.json') as f:
        data = json.load(f)

    # Sliding window correlation for W=10
    sw = data['enhancement_d']['10']
    steps = np.array(sw['sliding_window_steps'])
    corr = np.array(sw['sliding_window_rs'])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Color by sign: positive = blue, negative = red
    for i in range(len(steps) - 1):
        c = '#2166ac' if corr[i] >= 0 else '#d6604d'
        ax.bar(steps[i] / 1000, corr[i], width=0.18, color=c, alpha=0.8, edgecolor='none')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(17.8, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(17.9, 0.95, 'Shift', fontsize=8, rotation=90, va='top')
    ax.axvline(22.2, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(22.3, 0.95, 'Overfit', fontsize=8, rotation=90, va='top', color='gray')

    # Phase annotations
    ax.annotate('Pre-shift\n$r \\approx -0.95$', xy=(16.8, -0.94),
                fontsize=7.5, ha='center', va='top', color='#d6604d')
    ax.annotate('Post-shift\n$r \\approx +0.83$', xy=(19.3, 0.90),
                fontsize=7.5, ha='center', va='top', color='#2166ac')
    ax.annotate('Overfit\n$r \\approx -0.75$', xy=(24.0, -0.75),
                fontsize=7.5, ha='center', va='top', color='#d6604d')

    ax.set_xlabel('Training step (×10³)')
    ax.set_ylabel(r'Correlation $r$')
    ax.set_title('GPT-2 124M: Phase-Dependent Spectral Edge–Loss Coupling')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(16.2, 25.5)

    fig.savefig('fig_gpt2_sliding_corr.pdf')
    print('Saved fig_gpt2_sliding_corr.pdf')
    plt.close()


if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print('All figures generated.')
