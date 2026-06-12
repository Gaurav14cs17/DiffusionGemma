"""
Generate publication-quality diagrams for the DiffusionGemma visual guide.
Run: python generate_diagrams.py
Outputs PNG images to the current directory.
Requires: pip install matplotlib numpy
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#0f1117',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'font.family': 'sans-serif',
    'font.size': 12,
})

COLORS = {
    'accent': '#6366f1', 'green': '#22c55e', 'red': '#ef4444',
    'orange': '#f97316', 'blue': '#3b82f6', 'cyan': '#06b6d4',
    'pink': '#ec4899', 'yellow': '#eab308', 'purple': '#8b5cf6',
    'muted': '#94a3b8', 'card': '#1a1d2e', 'card2': '#232740',
    'bg': '#0f1117', 'text': '#e2e8f0',
}

OUT = os.path.dirname(os.path.abspath(__file__))


def fig1_roofline():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0.1, 100, 500)
    peak = 50
    bandwidth = 0.8
    ridge = peak / bandwidth
    y = np.minimum(bandwidth * x, peak)
    ax.plot(x, y, color=COLORS['accent'], linewidth=3, zorder=3)
    ax.axhline(y=peak, color=COLORS['green'], linewidth=1.5, linestyle='--', alpha=0.5)
    ax.annotate('Autoregressive\n(B=1, single user)', xy=(2, bandwidth*2),
                fontsize=13, fontweight='bold', color=COLORS['red'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1d2e', edgecolor=COLORS['red'], alpha=0.9))
    ax.plot(2, bandwidth*2, 'o', color=COLORS['red'], markersize=14, zorder=5)
    ax.annotate('DiffusionGemma\n(256 tokens at once)', xy=(ridge-5, peak-3),
                fontsize=13, fontweight='bold', color=COLORS['green'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1d2e', edgecolor=COLORS['green'], alpha=0.9))
    ax.plot(ridge, peak, 'o', color=COLORS['green'], markersize=14, zorder=5)
    ax.annotate('Ridge Point', xy=(ridge, peak+2), fontsize=10, color=COLORS['yellow'],
                ha='center', fontweight='bold')
    ax.annotate('', xy=(ridge-2, peak-5), xytext=(4, bandwidth*4),
                arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=2, linestyle='dashed'))
    ax.fill_between(x[x < ridge], 0, y[x < ridge], alpha=0.05, color=COLORS['accent'])
    ax.text(8, 15, 'Memory-Bound\nRegion', fontsize=11, color=COLORS['accent'], alpha=0.6, fontstyle='italic')
    ax.fill_between(x[x >= ridge], 0, y[x >= ridge], alpha=0.05, color=COLORS['green'])
    ax.text(75, 20, 'Compute-Bound\nRegion', fontsize=11, color=COLORS['green'], alpha=0.6, fontstyle='italic')
    ax.set_xlabel('Arithmetic Intensity (FLOPs / byte)', fontsize=13)
    ax.set_ylabel('Throughput (FLOPS)', fontsize=13)
    ax.set_title('Roofline Model: Why DiffusionGemma is Faster for Single Users', fontsize=15, fontweight='bold', color=COLORS['text'], pad=15)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 150)
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    ax.grid(True, alpha=0.1, color='#475569')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_roofline.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig2_alpha_schedule():
    fig, ax = plt.subplots(figsize=(10, 5))
    T = 1000
    t = np.arange(T)
    beta = np.linspace(1e-4, 0.02, T)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    signal = np.sqrt(alpha_bar)
    noise = np.sqrt(1 - alpha_bar)
    ax.plot(t, signal, color=COLORS['green'], linewidth=3, label='√ᾱₜ (signal coefficient)')
    ax.plot(t, noise, color=COLORS['red'], linewidth=3, label='√(1−ᾱₜ) (noise coefficient)')
    ax.fill_between(t, signal, alpha=0.08, color=COLORS['green'])
    ax.fill_between(t, noise, alpha=0.08, color=COLORS['red'])
    cross_idx = np.argmin(np.abs(signal - noise))
    ax.axvline(x=cross_idx, color=COLORS['yellow'], linewidth=1.5, linestyle='--', alpha=0.6)
    ax.annotate('Crossover\n(50/50)', xy=(cross_idx, signal[cross_idx]),
                xytext=(cross_idx + 100, 0.7), fontsize=11, color=COLORS['yellow'],
                fontweight='bold', arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=1.5))
    ax.set_xlabel('Timestep t', fontsize=13)
    ax.set_ylabel('Coefficient Value', fontsize=13)
    ax.set_title('Forward Diffusion: Signal Decays, Noise Grows', fontsize=15, fontweight='bold', color=COLORS['text'], pad=15)
    ax.legend(fontsize=12, loc='center right', framealpha=0.9, facecolor=COLORS['card'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    ax.grid(True, alpha=0.1, color='#475569')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_noise_schedule.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig3_temperature():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    logits = np.array([3.0, 1.5, 0.5, 0.2, -0.3])
    labels = ['Token A', 'Token B', 'Token C', 'Token D', 'Token E']
    temps = [2.0, 1.0, 0.3]
    titles = ['τ = 2.0 (Early Steps)\nExplore', 'τ = 1.0 (Mid Steps)\nBalanced', 'τ = 0.3 (Late Steps)\nCommit']
    colors_list = [COLORS['pink'], COLORS['orange'], COLORS['green']]
    for ax, tau, title, c in zip(axes, temps, titles, colors_list):
        probs = np.exp(logits / tau) / np.sum(np.exp(logits / tau))
        bars = ax.barh(labels, probs, color=c, alpha=0.7, edgecolor=c, linewidth=1.5)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{p:.2f}', va='center', fontsize=10, color=COLORS['text'])
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontsize=12, fontweight='bold', color=c, pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        ax.grid(True, alpha=0.08, axis='x', color='#475569')
    fig.suptitle('Temperature Effect on Token Probability Distributions',
                 fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_temperature_effect.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig4_attention_masks():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n = len(tokens)
    causal = np.tril(np.ones((n, n)))
    bidir = np.ones((n, n))
    combined = np.ones((n + 4, n + 4))
    combined[:4, :4] = np.tril(np.ones((4, 4)))
    combined[:4, 4:] = 0
    masks = [causal, bidir, combined]
    titles = ['Causal (Encoder Mode)', 'Bidirectional (Denoiser Mode)',
              'Combined (Encoder + Canvas)']
    cmaps = ['Blues', 'Greens', 'Purples']
    tick_labels = [tokens, tokens,
                   ['Enc₁','Enc₂','Enc₃','Enc₄','Can₁','Can₂','Can₃','Can₄','Can₅','Can₆']]
    for ax, mask, title, cmap, tl in zip(axes, masks, titles, cmaps, tick_labels):
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1.5, aspect='equal')
        ax.set_xticks(range(len(tl)))
        ax.set_yticks(range(len(tl)))
        ax.set_xticklabels(tl, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(tl, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                color = '#e2e8f0' if mask[i, j] > 0.5 else '#475569'
                ax.text(j, i, '✓' if mask[i, j] > 0.5 else '·',
                        ha='center', va='center', fontsize=8, color=color)
        ax.spines[:].set_color('#475569')
    fig.suptitle('Attention Masks in DiffusionGemma',
                 fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_attention_masks.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig5_denoising_trace():
    fig, ax = plt.subplots(figsize=(14, 7))
    steps = ['Step 0\n(noise)', 'Step 1', 'Step 2', 'Step 3', 'Step 4\n(done)']
    tokens_per_step = [
        ['xyz', '##', 'qr', '!!', 'zz', 'aa', 'mm', '!?'],
        ['The', 'cat', 'qr', 'on', 'zz', 'mat', 'mm', '!?'],
        ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', '!?'],
        ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'pur'],
        ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'purred'],
    ]
    is_clean = [
        [0,0,0,0,0,0,0,0],
        [1,1,0,1,0,1,0,0],
        [1,1,1,1,1,1,0.5,0],
        [1,1,1,1,1,1,1,0.5],
        [1,1,1,1,1,1,1,1],
    ]
    for si, (step_label, toks, clean) in enumerate(zip(steps, tokens_per_step, is_clean)):
        y = len(steps) - 1 - si
        ax.text(-0.8, y, step_label, fontsize=10, fontweight='bold', va='center',
                color=[COLORS['red'], COLORS['orange'], COLORS['yellow'], COLORS['cyan'], COLORS['green']][si])
        for ti, (tok, c) in enumerate(zip(toks, clean)):
            if c >= 1:
                fc, ec, tc = '#0d3320', COLORS['green'], COLORS['green']
            elif c >= 0.5:
                fc, ec, tc = '#2d1f0a', COLORS['orange'], COLORS['orange']
            else:
                fc, ec, tc = '#2a1215', COLORS['red'], COLORS['red']
            rect = FancyBboxPatch((ti * 1.3, y - 0.3), 1.1, 0.6, boxstyle="round,pad=0.08",
                                  facecolor=fc, edgecolor=ec, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(ti * 1.3 + 0.55, y, tok, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=tc)
    ax.set_xlim(-1.2, 10.5)
    ax.set_ylim(-0.8, len(steps) - 0.2)
    ax.set_title('Canvas Denoising Over Steps', fontsize=15, fontweight='bold',
                 color=COLORS['text'], pad=15)
    ax.axis('off')
    clean_patch = mpatches.Patch(facecolor='#0d3320', edgecolor=COLORS['green'], label='Confident (accepted)')
    partial_patch = mpatches.Patch(facecolor='#2d1f0a', edgecolor=COLORS['orange'], label='Partially confident')
    noise_patch = mpatches.Patch(facecolor='#2a1215', edgecolor=COLORS['red'], label='Noise (rejected)')
    ax.legend(handles=[clean_patch, partial_patch, noise_patch], loc='lower right',
              fontsize=10, framealpha=0.9, facecolor=COLORS['card'])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '05_denoising_trace.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig6_entropy_acceptance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    positions = ['pos 4\n"the"', 'pos 1\n"The"', 'pos 3\n"upon"', 'pos 2\n"crash"',
                 'pos 6\n"shore"', 'pos 5\n"sandy"', 'pos 8\n"rain"', 'pos 7\n"wind"']
    entropies = [0.35, 0.52, 0.85, 1.80, 2.90, 3.20, 5.10, 5.50]
    colors = [COLORS['green']]*6 + [COLORS['red']]*2
    bars = ax1.barh(positions[::-1], entropies[::-1], color=colors[::-1], alpha=0.7,
                    edgecolor=colors[::-1], linewidth=1.5)
    ax1.axvline(x=3.5, color=COLORS['yellow'], linewidth=2, linestyle='--', alpha=0.7)
    ax1.text(3.6, 7.5, 'Acceptance\nthreshold', fontsize=10, color=COLORS['yellow'], fontweight='bold')
    ax1.set_xlabel('Entropy (bits)', fontsize=12)
    ax1.set_title('Entropy per Position\n(sorted by confidence)', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#475569')
    ax1.spines['bottom'].set_color('#475569')
    H_max = 17.97
    cumsum = np.cumsum([h - H_max for h in entropies])
    ax2.plot(range(1, 9), cumsum, 'o-', color=COLORS['accent'], linewidth=2.5, markersize=8)
    ax2.axhline(y=-50, color=COLORS['yellow'], linewidth=2, linestyle='--', alpha=0.7)
    ax2.text(1, -48, 'Budget B = -50', fontsize=10, color=COLORS['yellow'], fontweight='bold')
    ax2.fill_between(range(1, 9), cumsum, -50, where=np.array(cumsum) >= -50,
                     alpha=0.1, color=COLORS['green'])
    ax2.fill_between(range(1, 9), cumsum, -50, where=np.array(cumsum) < -50,
                     alpha=0.1, color=COLORS['green'])
    for i, (cs, pos) in enumerate(zip(cumsum, positions)):
        color = COLORS['green'] if i < 6 else COLORS['red']
        ax2.annotate(f'{cs:.0f}', xy=(i+1, cs), xytext=(i+1.3, cs+5),
                     fontsize=9, color=color, fontweight='bold')
    ax2.set_xlabel('Tokens accepted (sorted)', fontsize=12)
    ax2.set_ylabel('Cumulative (H - H_max)', fontsize=12)
    ax2.set_title('Cumulative Entropy Budget\n(all pass in this example)', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#475569')
    ax2.spines['bottom'].set_color('#475569')
    ax2.grid(True, alpha=0.1, color='#475569')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '06_entropy_acceptance.png'), dpi=200, bbox_inches='tight')
    plt.close()


def fig7_ar_vs_diffusion():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1]})
    n_tokens = 10
    for i in range(n_tokens):
        for j in range(n_tokens):
            if j <= i:
                rect = FancyBboxPatch((j * 1.1, 0.8 - i * 0.08), 0.9, 0.06,
                                      boxstyle="round,pad=0.02",
                                      facecolor=COLORS['accent'], alpha=0.3 + 0.05 * j,
                                      edgecolor=COLORS['accent'], linewidth=0.5)
                ax1.add_patch(rect)
                ax1.text(j * 1.1 + 0.45, 0.83 - i * 0.08, f't{j+1}',
                         ha='center', va='center', fontsize=6, color=COLORS['text'])
    ax1.set_xlim(-0.5, 12)
    ax1.set_ylim(-0.2, 1.0)
    ax1.set_title('Autoregressive: One token per step (N steps for N tokens)',
                   fontsize=13, fontweight='bold', color=COLORS['accent'])
    ax1.axis('off')
    steps = 4
    for s in range(steps):
        for i in range(n_tokens):
            alpha = 0.15 + s * 0.25
            c = COLORS['green'] if s == steps - 1 else COLORS['cyan']
            rect = FancyBboxPatch((i * 1.1, 0.8 - s * 0.2), 0.9, 0.15,
                                  boxstyle="round,pad=0.02",
                                  facecolor=c, alpha=alpha,
                                  edgecolor=c, linewidth=0.5)
            ax2.add_patch(rect)
            ax2.text(i * 1.1 + 0.45, 0.875 - s * 0.2, f't{i+1}',
                     ha='center', va='center', fontsize=6, color=COLORS['text'])
        ax2.text(11.5, 0.87 - s * 0.2, f'Step {s+1}', fontsize=9, color=COLORS['muted'], va='center')
    ax2.set_xlim(-0.5, 13)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Diffusion: ALL tokens refined each step (S ≪ N steps)',
                   fontsize=13, fontweight='bold', color=COLORS['green'])
    ax2.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '07_ar_vs_diffusion.png'), dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Generating diagrams...")
    fig1_roofline()
    print("  ✓ 01_roofline.png")
    fig2_alpha_schedule()
    print("  ✓ 02_noise_schedule.png")
    fig3_temperature()
    print("  ✓ 03_temperature_effect.png")
    fig4_attention_masks()
    print("  ✓ 04_attention_masks.png")
    fig5_denoising_trace()
    print("  ✓ 05_denoising_trace.png")
    fig6_entropy_acceptance()
    print("  ✓ 06_entropy_acceptance.png")
    fig7_ar_vs_diffusion()
    print("  ✓ 07_ar_vs_diffusion.png")
    print(f"\nAll diagrams saved to: {OUT}/")
