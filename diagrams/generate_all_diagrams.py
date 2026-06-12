"""
Comprehensive DiffusionGemma Visual Guide - ALL diagrams.
Run: python generate_all_diagrams.py
Generates 20+ publication-quality PNG diagrams.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import os

plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#0f1117',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'font.family': 'DejaVu Sans',
    'font.size': 12,
})

C = {
    'accent': '#6366f1', 'green': '#22c55e', 'red': '#ef4444',
    'orange': '#f97316', 'blue': '#3b82f6', 'cyan': '#06b6d4',
    'pink': '#ec4899', 'yellow': '#eab308', 'purple': '#8b5cf6',
    'muted': '#94a3b8', 'card': '#1a1d2e', 'card2': '#232740',
    'bg': '#0f1117', 'text': '#e2e8f0', 'teal': '#14b8a6',
}
OUT = os.path.dirname(os.path.abspath(__file__))

def box(ax, x, y, w, h, text, fc, ec, tc='white', fs=10, fw='bold', alpha=0.85, rx=0.08):
    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={rx}",
                        facecolor=fc, edgecolor=ec, linewidth=1.8, alpha=alpha)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, wrap=True)
    return r

def arrow(ax, x1, y1, x2, y2, color=C['yellow'], style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ============================================================
# 01. WHY DIFFUSION? AR vs DIFFUSION comparison
# ============================================================
def fig01_why_diffusion():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('Why Diffusion LLMs?', fontsize=20, fontweight='bold', color=C['text'], pad=20)

    # AR side
    box(ax, 0.3, 5.8, 6, 0.8, 'AUTOREGRESSIVE (GPT, Gemma)', C['card'], C['red'], C['red'], 14)
    toks_ar = ['The','','','','','']
    for step in range(6):
        yy = 5.0 - step * 0.7
        ax.text(0.1, yy + 0.15, f'Step {step+1}', fontsize=8, color=C['muted'])
        for i in range(6):
            if i <= step:
                box(ax, 1.2 + i*0.85, yy, 0.75, 0.35, ['The','cat','sat','on','the','mat'][i],
                    '#0d3320', C['green'], C['green'], 8)
            else:
                box(ax, 1.2 + i*0.85, yy, 0.75, 0.35, '?',
                    C['card2'], '#475569', '#475569', 8, 'normal')
    ax.text(3.5, 0.7, '6 tokens = 6 forward passes', fontsize=11, color=C['red'],
            fontweight='bold', ha='center')
    ax.text(3.5, 0.3, 'Sequential, memory-bound, slow for 1 user',
            fontsize=9, color=C['muted'], ha='center')

    # Diffusion side
    box(ax, 7.5, 5.8, 6, 0.8, 'DIFFUSION (DiffusionGemma)', C['card'], C['green'], C['green'], 14)
    steps_d = [
        (['xyz','##','qr','!!','zz','aa'], [0]*6),
        (['The','cat','qr','on','zz','mat'], [1,1,0,1,0,1]),
        (['The','cat','sat','on','the','mat'], [1]*6),
    ]
    labels_d = ['Step 1', 'Step 2', 'Step 3\n(done!)']
    for step, (toks, clean) in enumerate(steps_d):
        yy = 5.0 - step * 1.3
        ax.text(7.3, yy + 0.15, labels_d[step], fontsize=8, color=C['muted'])
        for i, (t, cl) in enumerate(zip(toks, clean)):
            fc = '#0d3320' if cl else '#2a1215'
            ec = C['green'] if cl else C['red']
            tc = C['green'] if cl else C['red']
            box(ax, 8.4 + i*0.85, yy, 0.75, 0.35, t, fc, ec, tc, 8)
    ax.text(10.8, 0.7, 'ALL tokens refined simultaneously', fontsize=11, color=C['green'],
            fontweight='bold', ha='center')
    ax.text(10.8, 0.3, 'Parallel, compute-bound, fast for 1 user',
            fontsize=9, color=C['muted'], ha='center')

    # VS
    box(ax, 6.5, 3.2, 0.8, 0.5, 'VS', C['accent'], C['accent'], 'white', 14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_why_diffusion.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 02. ROOFLINE MODEL
# ============================================================
def fig02_roofline():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(0.1, 100, 500)
    peak = 50; bw = 0.8; ridge = peak / bw
    y = np.minimum(bw * x, peak)
    ax.plot(x, y, color=C['accent'], linewidth=3.5, zorder=3)
    ax.axhline(y=peak, color=C['green'], linewidth=1.5, linestyle='--', alpha=0.4)
    ax.fill_between(x[x < ridge], 0, y[x < ridge], alpha=0.06, color=C['accent'])
    ax.fill_between(x[x >= ridge], 0, y[x >= ridge], alpha=0.06, color=C['green'])
    ax.plot(2, bw*2, 'o', color=C['red'], markersize=18, zorder=5)
    ax.annotate('Autoregressive\n(B=1, 1 token/step)\nGPU mostly IDLE',
                xy=(2, bw*2), xytext=(5, 12), fontsize=12, fontweight='bold', color=C['red'],
                bbox=dict(boxstyle='round,pad=0.5', fc=C['card'], ec=C['red']),
                arrowprops=dict(arrowstyle='->', color=C['red'], lw=2))
    ax.plot(ridge, peak, 'o', color=C['green'], markersize=18, zorder=5)
    ax.annotate('DiffusionGemma\n(256 tokens/step)\nGPU FULLY used',
                xy=(ridge, peak), xytext=(ridge-30, peak-18), fontsize=12, fontweight='bold', color=C['green'],
                bbox=dict(boxstyle='round,pad=0.5', fc=C['card'], ec=C['green']),
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=2))
    ax.annotate('', xy=(ridge-2, peak-5), xytext=(4, bw*4),
                arrowprops=dict(arrowstyle='->', color=C['yellow'], lw=2.5, linestyle='dashed'))
    ax.text(12, 22, 'DiffusionGemma\nshifts here!', fontsize=11, color=C['yellow'],
            fontweight='bold', ha='center', fontstyle='italic')
    ax.text(6, 8, 'Memory-Bound', fontsize=13, color=C['accent'], alpha=0.5, fontweight='bold')
    ax.text(75, 25, 'Compute-Bound', fontsize=13, color=C['green'], alpha=0.5, fontweight='bold')
    ax.set_xlabel('Arithmetic Intensity (FLOPs / byte)', fontsize=14)
    ax.set_ylabel('Throughput', fontsize=14)
    ax.set_title('Roofline Model: GPU Utilization', fontsize=17, fontweight='bold', color=C['text'], pad=15)
    ax.set_xscale('log'); ax.set_xlim(0.5, 150); ax.set_ylim(0, 60)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#475569')
    ax.grid(True, alpha=0.08, color='#475569')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_roofline.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 03. CONTINUOUS DIFFUSION (images)
# ============================================================
def fig03_continuous_diffusion():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_title('Continuous Diffusion (for Images)', fontsize=16, fontweight='bold', color=C['text'], pad=15)
    noise_levels = [0, 0.2, 0.5, 0.8, 1.0]
    labels = ['Clean\nt=0', 'Light noise\nt=T/4', 'Medium\nt=T/2', 'Heavy\nt=3T/4', 'Pure noise\nt=T']
    cols = [C['green'], C['teal'], C['yellow'], C['orange'], C['red']]
    for i, (nl, label, col) in enumerate(zip(noise_levels, labels, cols)):
        x0 = 0.5 + i * 2.7
        np.random.seed(42)
        grid = np.random.rand(8, 8) * nl + (1 - nl) * np.outer(np.linspace(0.2,0.8,8), np.linspace(0.3,0.9,8))
        for r in range(8):
            for c_idx in range(8):
                rect = Rectangle((x0 + c_idx*0.15, 2.8 - r*0.15), 0.13, 0.13,
                                  facecolor=plt.cm.viridis(grid[r, c_idx]), edgecolor='none')
                ax.add_patch(rect)
        ax.text(x0 + 0.6, 1.3, label, ha='center', fontsize=9, color=col, fontweight='bold')
        if i < 4:
            arrow(ax, x0 + 1.3, 2.2, x0 + 2.3, 2.2, C['accent'])
    ax.text(7, 0.6, 'Forward: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε', fontsize=13,
            color=C['cyan'], ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc=C['card'], ec=C['cyan'], alpha=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_continuous_diffusion.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 04. NOISE SCHEDULE
# ============================================================
def fig04_noise_schedule():
    fig, ax = plt.subplots(figsize=(12, 5))
    T = 1000; t = np.arange(T)
    beta = np.linspace(1e-4, 0.02, T)
    alpha_bar = np.cumprod(1 - beta)
    signal = np.sqrt(alpha_bar); noise = np.sqrt(1 - alpha_bar)
    ax.plot(t, signal, color=C['green'], linewidth=3.5, label='√ᾱ_t  (signal)')
    ax.plot(t, noise, color=C['red'], linewidth=3.5, label='√(1-ᾱ_t)  (noise)')
    ax.fill_between(t, signal, alpha=0.08, color=C['green'])
    ax.fill_between(t, noise, alpha=0.08, color=C['red'])
    ci = np.argmin(np.abs(signal - noise))
    ax.axvline(x=ci, color=C['yellow'], linewidth=2, linestyle='--', alpha=0.6)
    ax.annotate('50/50\ncrossover', xy=(ci, signal[ci]), xytext=(ci+120, 0.75),
                fontsize=12, color=C['yellow'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['yellow'], lw=2))
    ax.set_xlabel('Timestep t →', fontsize=14); ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Noise Schedule: Signal Dies, Noise Grows', fontsize=17, fontweight='bold', color=C['text'], pad=15)
    ax.legend(fontsize=13, loc='center right', framealpha=0.9, facecolor=C['card'])
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#475569')
    ax.grid(True, alpha=0.08, color='#475569')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_noise_schedule.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 05. DISCRETE DIFFUSION: MASKED vs UNIFORM
# ============================================================
def fig05_masked_vs_uniform():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax in axes: ax.set_xlim(0, 8); ax.set_ylim(0, 7); ax.axis('off')

    # MASKED
    ax = axes[0]
    ax.set_title('Masked Diffusion (MDLM)', fontsize=15, fontweight='bold', color=C['blue'], pad=15)
    clean = ['The','cat','sat','on','the','mat']
    rows_m = [
        (['The','cat','sat','on','the','mat'], [1]*6, 'Clean'),
        (['The','[M]','sat','[M]','[M]','mat'], [1,0,1,0,0,1], 't=1'),
        (['[M]','[M]','[M]','[M]','[M]','[M]'], [0]*6, 't=T (all masked)'),
    ]
    for ri, (toks, cl, lbl) in enumerate(rows_m):
        yy = 5.8 - ri * 1.0
        ax.text(0.1, yy+0.12, lbl, fontsize=9, color=C['muted'], fontweight='bold')
        for ti, (t, c) in enumerate(zip(toks, cl)):
            fc = '#0d3320' if c else '#1a1a3e'
            ec = C['green'] if c else C['blue']
            tc = C['green'] if c else C['blue']
            box(ax, 1.3 + ti*1.0, yy, 0.85, 0.4, t, fc, ec, tc, 9)
    ax.text(4, 2.8, '↓ Denoise (reverse) ↓', fontsize=11, color=C['blue'], ha='center', fontweight='bold')
    rows_r = [
        (['[M]','[M]','[M]','[M]','[M]','[M]'], [0]*6, 'Start'),
        (['The','[M]','sat','[M]','the','[M]'], [1,0,1,0,1,0], 'Step 1'),
        (['The','cat','sat','on','the','mat'], [1]*6, 'Done'),
    ]
    for ri, (toks, cl, lbl) in enumerate(rows_r):
        yy = 2.2 - ri * 0.9
        ax.text(0.1, yy+0.12, lbl, fontsize=9, color=C['muted'], fontweight='bold')
        for ti, (t, c) in enumerate(zip(toks, cl)):
            fc = '#0d3320' if c else '#1a1a3e'
            ec = C['green'] if c else C['blue']
            tc = C['green'] if c else C['blue']
            box(ax, 1.3 + ti*1.0, yy, 0.85, 0.4, t, fc, ec, tc, 9)
    box(ax, 1.0, -0.3, 6.0, 0.5, '✗ Once unmasked → CANNOT change back. No self-correction!',
        '#2a1215', C['red'], C['red'], 10)

    # UNIFORM
    ax = axes[1]
    ax.set_title('Uniform Diffusion (DiffusionGemma)', fontsize=15, fontweight='bold', color=C['green'], pad=15)
    rows_u = [
        (['The','cat','sat','on','the','mat'], [1]*6, 'Clean'),
        (['The','dog','sat','xyz','qr','mat'], [1,0,1,0,0,1], 't=1'),
        (['xyz','##','qr','!!','zz','aa'], [0]*6, 't=T (all random)'),
    ]
    for ri, (toks, cl, lbl) in enumerate(rows_u):
        yy = 5.8 - ri * 1.0
        ax.text(0.1, yy+0.12, lbl, fontsize=9, color=C['muted'], fontweight='bold')
        for ti, (t, c) in enumerate(zip(toks, cl)):
            fc = '#0d3320' if c else '#2a1215'
            ec = C['green'] if c else C['red']
            tc = C['green'] if c else C['red']
            box(ax, 1.3 + ti*1.0, yy, 0.85, 0.4, t, fc, ec, tc, 9)
    ax.text(4, 2.8, '↓ Denoise (reverse) ↓', fontsize=11, color=C['green'], ha='center', fontweight='bold')
    rows_r2 = [
        (['xyz','##','qr','!!','zz','aa'], [0]*6, 'Start'),
        (['The','dog','sat','on','zz','mat'], [1,0.5,1,1,0,1], 'Step 1'),
        (['The','cat','sat','on','the','mat'], [1]*6, 'Done'),
    ]
    for ri, (toks, cl, lbl) in enumerate(rows_r2):
        yy = 2.2 - ri * 0.9
        ax.text(0.1, yy+0.12, lbl, fontsize=9, color=C['muted'], fontweight='bold')
        for ti, (t, c) in enumerate(zip(toks, cl)):
            if c >= 1: fc, ec, tc2 = '#0d3320', C['green'], C['green']
            elif c >= 0.5: fc, ec, tc2 = '#2d1f0a', C['orange'], C['orange']
            else: fc, ec, tc2 = '#2a1215', C['red'], C['red']
            box(ax, 1.3 + ti*1.0, yy, 0.85, 0.4, t, fc, ec, tc2, 9)
    box(ax, 1.0, -0.3, 6.0, 0.5, '✓ Any token can change at ANY step. Full self-correction!',
        '#0d3320', C['green'], C['green'], 10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '05_masked_vs_uniform.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 06. UNIFORM DIFFUSION MATH
# ============================================================
def fig06_uniform_math():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Uniform Diffusion: The Math', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    box(ax, 0.5, 4.5, 11, 1.2, '', C['card'], C['accent'], alpha=0.6)
    ax.text(6, 5.3, 'Forward Process', fontsize=14, fontweight='bold', color=C['accent'], ha='center')
    ax.text(6, 4.85, 'q(x_t = j | x_0 = k)  =  ᾱ_t · δ_jk  +  (1 - ᾱ_t) / K',
            fontsize=14, color=C['cyan'], ha='center', fontweight='bold',
            fontfamily='monospace')

    # Visual explanation
    box(ax, 0.8, 3.0, 4.5, 1.2, '', C['card'], C['green'], alpha=0.6)
    ax.text(3.05, 3.95, 'Stay same token (j = k)', fontsize=11, fontweight='bold', color=C['green'], ha='center')
    ax.text(3.05, 3.5, 'P = ᾱ_t + (1-ᾱ_t)/K', fontsize=12, color=C['green'], ha='center', fontfamily='monospace')
    ax.text(3.05, 3.15, 'High early, drops over time', fontsize=9, color=C['muted'], ha='center')

    box(ax, 6.0, 3.0, 5.5, 1.2, '', C['card'], C['red'], alpha=0.6)
    ax.text(8.75, 3.95, 'Jump to random token (j ≠ k)', fontsize=11, fontweight='bold', color=C['red'], ha='center')
    ax.text(8.75, 3.5, 'P = (1-ᾱ_t)/K', fontsize=12, color=C['red'], ha='center', fontfamily='monospace')
    ax.text(8.75, 3.15, 'Uniform over vocab, grows over time', fontsize=9, color=C['muted'], ha='center')

    # Numeric example
    box(ax, 0.8, 1.0, 10.5, 1.7, '', C['card2'], C['yellow'], alpha=0.5)
    ax.text(6, 2.45, 'Example: K=50000 vocab, ᾱ_t=0.7 at step t', fontsize=12, fontweight='bold', color=C['yellow'], ha='center')
    ax.text(3.5, 1.9, 'P(stay "cat") = 0.7 + 0.3/50000\n                        = 0.700006', fontsize=11, color=C['green'], ha='center', fontfamily='monospace')
    ax.text(8.5, 1.9, 'P(become "dog") = 0.3/50000\n                            = 0.000006', fontsize=11, color=C['red'], ha='center', fontfamily='monospace')
    ax.text(6, 1.2, 'At ᾱ_t=0:  P(stay)=1/K, P(any)=1/K → pure uniform noise!', fontsize=10, color=C['muted'], ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '06_uniform_math.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 07. GEMMA 4 BASE MODEL
# ============================================================
def fig07_gemma4():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('Gemma 4 26B A4B: Base Model', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    # Outer box
    box(ax, 0.5, 0.5, 13, 6.2, '', C['card'], C['accent'], alpha=0.3)
    ax.text(7, 6.45, 'Gemma 4 (26B total, 4B active)', fontsize=14, fontweight='bold', color=C['accent'], ha='center')

    # Input
    box(ax, 1, 5.2, 3, 0.7, 'Token Embeddings', '#1a1a3e', C['blue'], C['blue'], 11)
    arrow(ax, 2.5, 5.2, 2.5, 4.8)

    # Transformer layers
    for i in range(3):
        y = 3.8 - i * 1.3
        # Self attention
        box(ax, 1, y, 2.5, 0.5, f'Self-Attention (L{i+1})', '#1a1a3e', C['purple'], C['purple'], 10)
        arrow(ax, 3.5, y+0.25, 4.0, y+0.25, C['muted'])
        # MoE
        box(ax, 4.0, y-0.15, 3.5, 0.8, '', '#1a1a3e', C['orange'], alpha=0.5)
        ax.text(5.75, y+0.45, 'Mixture of Experts (MoE)', fontsize=9, fontweight='bold', color=C['orange'], ha='center')
        # Expert boxes inside
        for e in range(4):
            ex = 4.2 + e * 0.8
            box(ax, ex, y-0.05, 0.7, 0.35, f'E{e+1}', '#2d1f0a', C['orange'], C['orange'], 7)
        ax.text(5.75, y-0.12, 'Router picks 1 of ~64', fontsize=7, color=C['muted'], ha='center')
        arrow(ax, 7.5, y+0.25, 8.0, y+0.25, C['muted'])
        box(ax, 8.0, y, 2, 0.5, 'LayerNorm + Residual', '#0f1a1a', C['teal'], C['teal'], 9)

    # Key stats
    box(ax, 10.5, 4.5, 3.0, 2.0, '', C['card2'], C['cyan'], alpha=0.4)
    ax.text(12, 6.2, 'Key Specs', fontsize=11, fontweight='bold', color=C['cyan'], ha='center')
    stats = ['26B parameters total', '4B active per token', '~64 experts per layer',
             'Router picks 1 expert', 'Causal attention only']
    for si, s in enumerate(stats):
        ax.text(10.7, 5.8 - si*0.35, f'• {s}', fontsize=9, color=C['text'])

    # Problem callout
    box(ax, 10.5, 1.2, 3.0, 1.2, '', '#2a1215', C['red'], alpha=0.6)
    ax.text(12, 2.15, 'THE PROBLEM', fontsize=10, fontweight='bold', color=C['red'], ha='center')
    ax.text(12, 1.75, 'Causal attention\n= can only look LEFT\n= bad for denoising!', fontsize=9, color=C['red'], ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '07_gemma4_base.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 08. ATTENTION MASKS (Causal vs Bidirectional)
# ============================================================
def fig08_attention_masks():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    tokens = ['Write', 'a', 'poem', 'about', 'cats']
    n = len(tokens)
    masks = [np.tril(np.ones((n,n))), np.ones((n,n)), None]
    titles = ['Causal Mask\n(Encoder Mode)', 'Bidirectional Mask\n(Denoiser Mode)', 'Combined Mask\n(Encoder + Canvas)']
    cmaps_c = [C['blue'], C['green'], C['purple']]

    for idx in range(2):
        ax = axes[idx]; mask = masks[idx]
        ax.set_title(titles[idx], fontsize=13, fontweight='bold', color=cmaps_c[idx], pad=15)
        for i in range(n):
            for j in range(n):
                color = cmaps_c[idx] if mask[i,j] else C['card2']
                alpha = 0.6 if mask[i,j] else 0.3
                rect = Rectangle((j, n-1-i), 0.9, 0.9, facecolor=color, alpha=alpha,
                                  edgecolor=C['bg'], linewidth=2)
                ax.add_patch(rect)
                sym = '✓' if mask[i,j] else '✗'
                ax.text(j+0.45, n-1-i+0.45, sym, ha='center', va='center',
                        fontsize=10, color='white' if mask[i,j] else '#475569', fontweight='bold')
        ax.set_xticks([x+0.45 for x in range(n)]); ax.set_xticklabels(tokens, fontsize=9, rotation=30)
        ax.set_yticks([y+0.45 for y in range(n)]); ax.set_yticklabels(tokens[::-1], fontsize=9)
        ax.set_xlim(-0.1, n); ax.set_ylim(-0.1, n)
        ax.set_xlabel('Key (attends to)', fontsize=10); ax.set_ylabel('Query (token)', fontsize=10)
        for s in ax.spines.values(): s.set_visible(False)

    # Combined
    ax = axes[2]
    enc_tokens = ['Enc₁', 'Enc₂', 'Enc₃']
    can_tokens = ['Can₁', 'Can₂', 'Can₃']
    all_tokens = enc_tokens + can_tokens
    m = len(all_tokens)
    combined = np.zeros((m,m))
    combined[:3,:3] = np.tril(np.ones((3,3)))  # encoder: causal
    combined[3:,:3] = 1   # canvas sees encoder
    combined[3:,3:] = 1   # canvas sees canvas (bidirectional)
    ax.set_title(titles[2], fontsize=13, fontweight='bold', color=cmaps_c[2], pad=15)
    for i in range(m):
        for j in range(m):
            if i < 3 and j < 3:
                color = C['blue'] if combined[i,j] else C['card2']
            elif i >= 3:
                color = C['green'] if j >= 3 and combined[i,j] else C['purple'] if combined[i,j] else C['card2']
            else:
                color = C['card2']
            alpha = 0.6 if combined[i,j] else 0.3
            rect = Rectangle((j, m-1-i), 0.9, 0.9, facecolor=color, alpha=alpha,
                              edgecolor=C['bg'], linewidth=2)
            ax.add_patch(rect)
            sym = '✓' if combined[i,j] else '✗'
            ax.text(j+0.45, m-1-i+0.45, sym, ha='center', va='center',
                    fontsize=9, color='white' if combined[i,j] else '#475569', fontweight='bold')
    ax.set_xticks([x+0.45 for x in range(m)]); ax.set_xticklabels(all_tokens, fontsize=8, rotation=30)
    ax.set_yticks([y+0.45 for y in range(m)]); ax.set_yticklabels(all_tokens[::-1], fontsize=8)
    ax.set_xlim(-0.1, m); ax.set_ylim(-0.1, m)
    ax.set_xlabel('Key', fontsize=10); ax.set_ylabel('Query', fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)

    fig.suptitle('Attention Masks in DiffusionGemma', fontsize=17, fontweight='bold', color=C['text'], y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '08_attention_masks.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 09. ENCODER-DENOISER PATCH
# ============================================================
def fig09_encoder_denoiser():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('One Model, Two Roles: The Encoder-Denoiser Patch', fontsize=17, fontweight='bold', color=C['text'], pad=15)

    # Central model
    box(ax, 4.5, 2.8, 5, 1.5, '', C['card'], C['accent'])
    ax.text(7, 3.95, 'Same Gemma 4 Weights', fontsize=13, fontweight='bold', color=C['accent'], ha='center')
    ax.text(7, 3.45, '(26B parameters, shared)', fontsize=10, color=C['muted'], ha='center')
    ax.text(7, 3.05, 'Only the ATTENTION MASK changes!', fontsize=11, fontweight='bold', color=C['yellow'], ha='center')

    # Encoder
    box(ax, 0.3, 5.2, 5, 1.5, '', '#0d1a33', C['blue'])
    ax.text(2.8, 6.35, '🔵 ENCODER MODE', fontsize=14, fontweight='bold', color=C['blue'], ha='center')
    ax.text(2.8, 5.9, 'Input: User query ("Write a poem")', fontsize=10, color=C['text'], ha='center')
    ax.text(2.8, 5.55, 'Mask: CAUSAL (▼ triangle)', fontsize=10, color=C['blue'], ha='center')
    ax.text(2.8, 5.25, 'Runs: ONCE → saves KV cache', fontsize=10, color=C['cyan'], ha='center')
    arrow(ax, 2.8, 5.2, 5.5, 4.3, C['blue'])

    # Denoiser
    box(ax, 8.5, 5.2, 5.2, 1.5, '', '#0d2a15', C['green'])
    ax.text(11.1, 6.35, '🟢 DENOISER MODE', fontsize=14, fontweight='bold', color=C['green'], ha='center')
    ax.text(11.1, 5.9, 'Input: Noisy canvas tokens', fontsize=10, color=C['text'], ha='center')
    ax.text(11.1, 5.55, 'Mask: BIDIRECTIONAL (■ full)', fontsize=10, color=C['green'], ha='center')
    ax.text(11.1, 5.25, 'Runs: S times (denoising loop)', fontsize=10, color=C['cyan'], ha='center')
    arrow(ax, 11.1, 5.2, 8.5, 4.3, C['green'])

    # KV cache arrow
    box(ax, 1.0, 0.8, 4.5, 1.5, '', C['card'], C['yellow'], alpha=0.5)
    ax.text(3.25, 2.0, 'KV Cache', fontsize=13, fontweight='bold', color=C['yellow'], ha='center')
    ax.text(3.25, 1.55, 'Keys & Values from encoder', fontsize=10, color=C['text'], ha='center')
    ax.text(3.25, 1.15, 'Computed once, reused every step', fontsize=9, color=C['muted'], ha='center')
    arrow(ax, 3.5, 2.8, 3.25, 2.3, C['yellow'])
    arrow(ax, 5.5, 1.5, 8.5, 1.5, C['yellow'], lw=3)
    ax.text(7, 1.8, 'shared every step →', fontsize=10, color=C['yellow'], ha='center', fontweight='bold')

    # Output
    box(ax, 8.5, 0.3, 5.2, 1.5, '', '#0d2a15', C['green'], alpha=0.5)
    ax.text(11.1, 1.5, 'Output: Logits', fontsize=12, fontweight='bold', color=C['green'], ha='center')
    ax.text(11.1, 1.05, 'Probability for EVERY token\nat ALL 256 canvas positions', fontsize=10, color=C['text'], ha='center')
    ax.text(11.1, 0.6, '→ Sample → Accept/Reject → Repeat', fontsize=9, color=C['muted'], ha='center')
    arrow(ax, 11.1, 2.8, 11.1, 1.8, C['green'])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '09_encoder_denoiser.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 10. KV-CACHE SHARING
# ============================================================
def fig10_kv_cache():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('KV-Cache Sharing: How Encoder Talks to Denoiser', fontsize=17, fontweight='bold', color=C['text'], pad=15)

    # Encoder KV
    box(ax, 0.5, 3.5, 3.5, 2.0, '', '#0d1a33', C['blue'])
    ax.text(2.25, 5.2, 'Encoder KV Cache', fontsize=12, fontweight='bold', color=C['blue'], ha='center')
    kv_toks = ['Write', 'a', 'poem']
    for i, t in enumerate(kv_toks):
        box(ax, 0.8, 4.4 - i*0.45, 1.2, 0.35, f'K,V "{t}"', '#1a1a3e', C['blue'], C['blue'], 8)

    ax.text(2.25, 3.6, 'Computed ONCE\nReused every step', fontsize=8, color=C['cyan'], ha='center')

    # Canvas KV
    box(ax, 5.0, 3.5, 3.5, 2.0, '', '#0d2a15', C['green'])
    ax.text(6.75, 5.2, 'Canvas KV (step s)', fontsize=12, fontweight='bold', color=C['green'], ha='center')
    can_toks = ['tok₁', 'tok₂', 'tok₃']
    for i, t in enumerate(can_toks):
        box(ax, 5.3, 4.4 - i*0.45, 1.2, 0.35, f'K,V "{t}"', '#0d2a15', C['green'], C['green'], 8)
    ax.text(6.75, 3.6, 'Recomputed\nevery step', fontsize=8, color=C['teal'], ha='center')

    # Concatenation
    arrow(ax, 4.0, 4.5, 4.5, 4.5, C['yellow'], lw=2.5)
    arrow(ax, 8.5, 4.5, 9.0, 4.5, C['yellow'], lw=2.5)

    box(ax, 9.0, 3.2, 4.8, 2.5, '', C['card'], C['purple'])
    ax.text(11.4, 5.4, 'ATTENTION', fontsize=13, fontweight='bold', color=C['purple'], ha='center')
    ax.text(11.4, 4.9, 'Q from canvas tokens', fontsize=10, color=C['text'], ha='center')
    ax.text(11.4, 4.5, 'K = [ Enc_K  |  Canvas_K ]', fontsize=10, color=C['yellow'], ha='center', fontfamily='monospace')
    ax.text(11.4, 4.1, 'V = [ Enc_V  |  Canvas_V ]', fontsize=10, color=C['yellow'], ha='center', fontfamily='monospace')
    ax.text(11.4, 3.6, 'Each canvas token attends\nto BOTH query AND canvas!', fontsize=10, color=C['cyan'], ha='center')

    # Output
    arrow(ax, 11.4, 3.2, 11.4, 2.3, C['purple'])
    box(ax, 9.0, 1.5, 4.8, 0.8, 'Rich hidden states\nblending query context + canvas tokens', C['card'], C['teal'], C['teal'], 10)

    # Savings callout
    box(ax, 0.5, 1.2, 8.0, 1.0, '', '#1a1a2e', C['yellow'], alpha=0.5)
    ax.text(4.5, 2.0, '💡 KEY INSIGHT', fontsize=11, fontweight='bold', color=C['yellow'], ha='center')
    ax.text(4.5, 1.55, 'Encoder KV computed ONCE, reused for ALL S denoising steps → huge speedup!', fontsize=10, color=C['text'], ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '10_kv_cache.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 11. FULL ARCHITECTURE PIPELINE
# ============================================================
def fig11_full_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('Complete DiffusionGemma Architecture', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    # User query
    box(ax, 0.3, 6.5, 3.5, 0.8, '"Write a poem about cats"', C['card'], C['blue'], C['blue'], 10)
    arrow(ax, 2.0, 6.5, 2.0, 6.0, C['blue'])

    # Encoder
    box(ax, 0.5, 4.8, 3.0, 1.2, '', '#0d1a33', C['blue'])
    ax.text(2, 5.75, 'ENCODER', fontsize=13, fontweight='bold', color=C['blue'], ha='center')
    ax.text(2, 5.3, 'Causal attention', fontsize=9, color=C['muted'], ha='center')
    ax.text(2, 5.0, 'Run ONCE', fontsize=9, color=C['cyan'], ha='center')

    # KV Cache
    arrow(ax, 2.0, 4.8, 2.0, 4.3, C['yellow'])
    box(ax, 0.5, 3.5, 3.0, 0.8, 'KV Cache\n(stored)', '#2d1f0a', C['yellow'], C['yellow'], 10)

    # Denoising loop
    box(ax, 4.5, 2.0, 11.0, 5.5, '', C['card'], C['green'], alpha=0.2)
    ax.text(10, 7.2, 'DENOISING LOOP (S steps)', fontsize=14, fontweight='bold', color=C['green'], ha='center')

    # Random init
    box(ax, 5.0, 6.0, 3.0, 0.8, 'Random Canvas\n(256 tokens)', '#2a1215', C['red'], C['red'], 10)
    arrow(ax, 8.0, 6.4, 8.7, 6.4, C['muted'])

    # Self conditioning
    box(ax, 8.7, 6.0, 3.0, 0.8, '① Self-Conditioning\n+ prev predictions', C['card'], C['pink'], C['pink'], 9)
    arrow(ax, 10.2, 6.0, 10.2, 5.5, C['pink'])

    # Denoiser
    box(ax, 8.0, 4.3, 4.5, 1.2, '', '#0d2a15', C['green'])
    ax.text(10.25, 5.25, '② DENOISER (Bidirectional)', fontsize=12, fontweight='bold', color=C['green'], ha='center')
    ax.text(10.25, 4.8, 'Gemma 4 + Encoder KV', fontsize=10, color=C['text'], ha='center')
    ax.text(10.25, 4.45, '→ logits at ALL positions', fontsize=9, color=C['cyan'], ha='center')

    # KV arrow to denoiser
    arrow(ax, 3.5, 3.9, 8.0, 4.7, C['yellow'], lw=2.5)
    ax.text(5.5, 4.5, 'KV Cache →', fontsize=9, color=C['yellow'], fontweight='bold')

    # Temperature
    arrow(ax, 10.25, 4.3, 10.25, 3.8, C['pink'])
    box(ax, 8.5, 3.0, 3.5, 0.8, '③ Temperature Scaling\nlogits / τ', C['card'], C['pink'], C['pink'], 9)

    # Sampling
    arrow(ax, 10.25, 3.0, 10.25, 2.6, C['orange'])
    box(ax, 8.5, 2.0, 3.5, 0.6, '④ Sample tokens', C['card'], C['orange'], C['orange'], 9)

    # Accept/reject
    arrow(ax, 12.0, 2.3, 12.8, 2.3, C['yellow'])
    box(ax, 12.8, 2.0, 2.5, 0.6, '⑤ Accept/Reject\n(entropy check)', C['card'], C['yellow'], C['yellow'], 8)

    # Loop back
    ax.annotate('', xy=(14.5, 6.4), xytext=(14.5, 2.6),
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=2, connectionstyle='arc3,rad=0.3'))
    ax.text(14.8, 4.5, 'loop\nback', fontsize=9, color=C['green'], ha='center', fontweight='bold', rotation=90)

    # Adaptive stop
    box(ax, 5.0, 2.0, 3.0, 0.8, '⑥ All stable?\n→ STOP early', '#0d2a15', C['teal'], C['teal'], 9)
    arrow(ax, 5.0, 2.0, 5.0, 1.2, C['teal'])

    # Output
    box(ax, 4.0, 0.3, 5.0, 0.8, '✅ Final Output: "The cat sat on the mat..."', '#0d3320', C['green'], C['green'], 11)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '11_full_architecture.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 12. SELF-CONDITIONING
# ============================================================
def fig12_self_conditioning():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Self-Conditioning: Memory Between Steps', fontsize=17, fontweight='bold', color=C['text'], pad=15)

    # Step S
    box(ax, 0.3, 3.0, 5.5, 2.7, '', C['card'], C['accent'])
    ax.text(3.05, 5.4, 'Step s: Model predicts', fontsize=12, fontweight='bold', color=C['accent'], ha='center')

    # Prob distribution
    probs = [('cat', 0.60, C['green']), ('dog', 0.15, C['teal']), ('hat', 0.10, C['cyan']),
             ('bat', 0.08, C['blue']), ('...', 0.07, C['muted'])]
    for i, (tok, p, col) in enumerate(probs):
        bw = p * 4
        box(ax, 0.8, 4.8 - i*0.38, bw, 0.3, f'{tok}: {p:.0%}', col, col, 'white', 8, alpha=0.7)

    # Soft embedding
    arrow(ax, 3.05, 3.0, 3.05, 2.5, C['yellow'])
    box(ax, 1.2, 1.7, 3.7, 0.7, '', '#2d1f0a', C['yellow'])
    ax.text(3.05, 2.2, 'Soft Embedding', fontsize=11, fontweight='bold', color=C['yellow'], ha='center')
    ax.text(3.05, 1.85, 'ẽ = Σ p(tok) × Embed(tok)', fontsize=9, color=C['text'], ha='center')

    # FFNN
    arrow(ax, 3.05, 1.7, 3.05, 1.2, C['purple'])
    box(ax, 1.5, 0.5, 3.1, 0.7, 'FFNN (trainable)', '#1a1a3e', C['purple'], C['purple'], 10)
    ax.text(3.05, 0.3, 'c = FFNN(ẽ)', fontsize=9, color=C['muted'], ha='center')

    # Arrow to step S+1
    arrow(ax, 5.8, 0.85, 7.2, 0.85, C['yellow'], lw=3)
    ax.text(6.5, 1.1, 'carry\nmemory', fontsize=9, color=C['yellow'], ha='center', fontweight='bold')

    # Step S+1
    box(ax, 7.2, 2.5, 6.5, 3.2, '', C['card'], C['green'])
    ax.text(10.45, 5.4, 'Step s+1: Enriched Input', fontsize=12, fontweight='bold', color=C['green'], ha='center')

    box(ax, 7.8, 4.3, 5.3, 0.7, 'Token embedding: e = Embed("random_tok")', '#0d2a15', C['green'], C['text'], 9, 'normal')
    ax.text(10.45, 3.8, '+', fontsize=20, color=C['yellow'], ha='center', fontweight='bold')
    box(ax, 7.8, 3.0, 5.3, 0.7, 'Self-conditioning: c (from step s)', '#2d1f0a', C['yellow'], C['yellow'], 9, 'normal')
    ax.text(10.45, 2.55, '═══════════════════════', fontsize=10, color=C['muted'], ha='center')

    box(ax, 8.5, 0.3, 3.9, 0.8, 'ê = e + c  (enriched!)\nModel knows what it predicted before',
        '#0d3320', C['green'], C['green'], 9)
    arrow(ax, 10.45, 2.5, 10.45, 1.1, C['green'])

    # Insight
    box(ax, 7.5, 1.3, 5.8, 0.9, '', '#0d2a15', C['cyan'], alpha=0.4)
    ax.text(10.4, 1.95, '💡 Even if the current token is random noise,', fontsize=9, color=C['text'], ha='center')
    ax.text(10.4, 1.6, 'the model remembers: "last step I thought this was cat (60%)"', fontsize=9, color=C['cyan'], ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '12_self_conditioning.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 13. TEMPERATURE SCHEDULE
# ============================================================
def fig13_temperature():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    logits = np.array([3.0, 1.5, 0.5, 0.2, -0.3])
    labels = ['A', 'B', 'C', 'D', 'E']
    temps = [3.0, 1.5, 0.7, 0.2]
    titles = ['τ=3.0\n(Very early)', 'τ=1.5\n(Early)', 'τ=0.7\n(Late)', 'τ=0.2\n(Final)']
    colors_list = [C['red'], C['pink'], C['orange'], C['green']]
    subtitles = ['EXPLORE\nwildly', 'EXPLORE\nbroadly', 'REFINE\nchoices', 'COMMIT\nto best']

    for ax, tau, title, col, sub in zip(axes, temps, titles, colors_list, subtitles):
        probs = np.exp(logits / tau) / np.sum(np.exp(logits / tau))
        bars = ax.barh(labels, probs, color=col, alpha=0.75, edgecolor=col, linewidth=2)
        for bar, p in zip(bars, probs):
            if p > 0.01:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{p:.0%}', va='center', fontsize=11, color=C['text'], fontweight='bold')
        ax.set_xlim(0, 1.15)
        ax.set_title(title, fontsize=12, fontweight='bold', color=col, pad=10)
        ax.text(0.5, -0.8, sub, fontsize=11, color=col, ha='center', fontweight='bold',
                transform=ax.transAxes)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#475569')
        ax.grid(True, alpha=0.06, axis='x', color='#475569')
    fig.suptitle('Temperature Schedule: Explore → Commit', fontsize=16, fontweight='bold', color=C['text'], y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '13_temperature.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 14. ENTROPY-BOUNDED ACCEPTANCE
# ============================================================
def fig14_entropy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    positions = ['pos₁ "The"', 'pos₃ "sat"', 'pos₅ "the"', 'pos₄ "on"',
                 'pos₆ "mat"', 'pos₂ "dog"', 'pos₇ "and"', 'pos₈ "!?"']
    entropies = [0.2, 0.4, 0.5, 0.7, 1.5, 3.2, 4.8, 6.1]
    threshold = 3.5
    colors = [C['green'] if e < threshold else C['red'] for e in entropies]
    statuses = ['✓ Accept' if e < threshold else '✗ Reject → re-noise' for e in entropies]

    bars = ax1.barh(range(len(positions)), entropies, color=colors, alpha=0.7, edgecolor=colors, linewidth=2)
    ax1.set_yticks(range(len(positions))); ax1.set_yticklabels(positions[::-1], fontsize=10)
    ax1.axvline(x=threshold, color=C['yellow'], linewidth=2.5, linestyle='--')
    ax1.text(threshold + 0.2, 7.2, f'Threshold', fontsize=11, color=C['yellow'], fontweight='bold')
    for i, (e, s) in enumerate(zip(entropies, statuses)):
        ax1.text(max(e + 0.15, 0.3), i, s, va='center', fontsize=9, color=colors[i], fontweight='bold')
    ax1.set_xlabel('Entropy (bits)', fontsize=13)
    ax1.set_title('Token Confidence\n(sorted by entropy)', fontsize=14, fontweight='bold', color=C['text'])
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    for s in ['left','bottom']: ax1.spines[s].set_color('#475569')
    ax1.grid(True, alpha=0.06, axis='x', color='#475569')

    # After acceptance
    canvas_before = [('The', 1), ('dog', 0), ('sat', 1), ('on', 1), ('the', 1), ('mat', 1), ('and', 0), ('!?', 0)]
    canvas_after = [('The', 1), ('xyz', -1), ('sat', 1), ('on', 1), ('the', 1), ('mat', 1), ('qr', -1), ('##', -1)]

    ax2.set_xlim(0, 10); ax2.set_ylim(0, 6); ax2.axis('off')
    ax2.set_title('Accept/Reject Result', fontsize=14, fontweight='bold', color=C['text'], pad=15)

    ax2.text(5, 5.5, 'BEFORE acceptance check:', fontsize=11, color=C['muted'], ha='center')
    for i, (tok, cl) in enumerate(canvas_before):
        fc = '#0d3320' if cl == 1 else '#2d1f0a'
        ec = C['green'] if cl == 1 else C['orange']
        tc = C['green'] if cl == 1 else C['orange']
        box(ax2, 0.5 + i*1.1, 4.7, 0.95, 0.5, tok, fc, ec, tc, 9)

    ax2.text(5, 4.0, '↓  Entropy check  ↓', fontsize=11, color=C['yellow'], ha='center', fontweight='bold')

    ax2.text(5, 3.3, 'AFTER acceptance check:', fontsize=11, color=C['muted'], ha='center')
    for i, (tok, cl) in enumerate(canvas_after):
        if cl == 1:
            fc, ec, tc = '#0d3320', C['green'], C['green']
            lbl = tok
        elif cl == -1:
            fc, ec, tc = '#2a1215', C['red'], C['red']
            lbl = tok
        else:
            fc, ec, tc = '#2d1f0a', C['orange'], C['orange']
            lbl = tok
        box(ax2, 0.5 + i*1.1, 2.5, 0.95, 0.5, lbl, fc, ec, tc, 9)

    ax2.text(5, 1.8, 'Rejected tokens → replaced with FRESH random tokens', fontsize=10, color=C['red'], ha='center')
    ax2.text(5, 1.3, 'This maintains the noise distribution the model expects!', fontsize=9, color=C['muted'], ha='center')

    green_p = mpatches.Patch(fc='#0d3320', ec=C['green'], lw=2, label='Accepted (confident)')
    red_p = mpatches.Patch(fc='#2a1215', ec=C['red'], lw=2, label='Rejected → re-noised')
    ax2.legend(handles=[green_p, red_p], loc='lower center', fontsize=10, framealpha=0.9, facecolor=C['card'])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '14_entropy_acceptance.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 15. DENOISING TRACE (step by step)
# ============================================================
def fig15_denoising_trace():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-1.5, 14); ax.set_ylim(-0.5, 9); ax.axis('off')
    ax.set_title('Step-by-Step Canvas Denoising', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    steps_data = [
        ('Step 0\n(noise)', ['xyz','##','qr','!!','zz','aa','mm','!?'], [0]*8,
         'Start with 8 random tokens'),
        ('Step 1', ['The','cat','qr','on','zz','mat','mm','!?'], [1,1,0,1,0,1,0,0],
         '4 tokens accepted, 4 rejected → re-noised'),
        ('Step 2', ['The','cat','sat','on','the','mat','and','!?'], [1,1,1,1,1,1,0.5,0],
         '6 tokens stable, "and" partially confident'),
        ('Step 3', ['The','cat','sat','on','the','mat','and','pur'], [1,1,1,1,1,1,1,0.5],
         '7 tokens stable, last partially confident'),
        ('Step 4\n(done!)', ['The','cat','sat','on','the','mat','and','purred'], [1]*8,
         'All tokens stable + low entropy → STOP'),
    ]
    step_colors = [C['red'], C['orange'], C['yellow'], C['cyan'], C['green']]

    for si, (label, toks, clean, desc) in enumerate(steps_data):
        y = 7.8 - si * 1.6
        ax.text(-1.2, y + 0.15, label, fontsize=10, fontweight='bold', color=step_colors[si],
                va='center', ha='center')
        for ti, (t, c) in enumerate(zip(toks, clean)):
            if c >= 1: fc, ec, tc = '#0d3320', C['green'], C['green']
            elif c >= 0.5: fc, ec, tc = '#2d1f0a', C['orange'], C['orange']
            else: fc, ec, tc = '#2a1215', C['red'], C['red']
            box(ax, 0.3 + ti * 1.35, y - 0.25, 1.15, 0.55, t, fc, ec, tc, 9)
        ax.text(11.5, y, desc, fontsize=9, color=C['muted'], va='center')
        if si < len(steps_data) - 1:
            ax.text(5.5, y - 0.65, '↓', fontsize=16, color=step_colors[si], ha='center')

    green_p = mpatches.Patch(fc='#0d3320', ec=C['green'], lw=2, label='Confident (accepted)')
    orange_p = mpatches.Patch(fc='#2d1f0a', ec=C['orange'], lw=2, label='Partially confident')
    red_p = mpatches.Patch(fc='#2a1215', ec=C['red'], lw=2, label='Noise / rejected')
    ax.legend(handles=[green_p, orange_p, red_p], loc='lower right', fontsize=11,
              framealpha=0.9, facecolor=C['card'])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '15_denoising_trace.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 16. MULTI-CANVAS (Block Diffusion)
# ============================================================
def fig16_multi_canvas():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
    ax.set_title('Multi-Canvas (Block Diffusion): Long Sequences', fontsize=17, fontweight='bold', color=C['text'], pad=15)

    for bi in range(3):
        x0 = 0.5 + bi * 4.5
        # Block box
        box(ax, x0, 1.5, 3.8, 2.5, '', C['card'], C['accent'], alpha=0.5)
        ax.text(x0 + 1.9, 3.8, f'Canvas {bi+1}', fontsize=13, fontweight='bold', color=C['accent'], ha='center')
        # Tokens
        for t in range(4):
            box(ax, x0 + 0.15 + t*0.9, 3.0, 0.8, 0.4, f't{bi*4+t+1}', '#0d3320', C['green'], C['green'], 8)
        # Diffusion label
        ax.text(x0 + 1.9, 2.5, 'Diffusion\n(S steps, parallel)', fontsize=9, color=C['green'], ha='center')
        ax.text(x0 + 1.9, 1.8, 'bidirectional attention', fontsize=8, color=C['muted'], ha='center')
        if bi < 2:
            arrow(ax, x0 + 3.8, 2.75, x0 + 4.5, 2.75, C['yellow'], lw=3)
            ax.text(x0 + 4.15, 3.1, 'KV→', fontsize=9, color=C['yellow'], fontweight='bold', ha='center')

    # Labels
    box(ax, 0.5, 0.3, 6.0, 0.8, '⟵ Diffusion WITHIN each canvas (parallel) ⟶',
        '#0d2a15', C['green'], C['green'], 10)
    box(ax, 7.5, 0.3, 6.0, 0.8, '⟵ Autoregressive BETWEEN canvases (sequential) ⟶',
        '#0d1a33', C['blue'], C['blue'], 10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '16_multi_canvas.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 17. TRAINING PROCESS
# ============================================================
def fig17_training():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title('Training DiffusionGemma', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    # Clean sequence
    ax.text(0.5, 5.5, '① Take a clean sequence:', fontsize=12, fontweight='bold', color=C['blue'])
    clean = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    for i, t in enumerate(clean):
        box(ax, 1.0 + i * 1.3, 4.8, 1.1, 0.45, t, '#0d3320', C['green'], C['green'], 10)

    # Corrupt
    arrow(ax, 5, 4.8, 5, 4.3, C['yellow'])
    ax.text(6.5, 4.45, '② Corrupt at random noise level t', fontsize=12, fontweight='bold', color=C['yellow'])
    noisy = ['The', 'xyz', 'sat', '##', 'the', 'mat']
    noisy_cl = [1, 0, 1, 0, 1, 1]
    for i, (t, c) in enumerate(zip(noisy, noisy_cl)):
        fc = '#0d3320' if c else '#2a1215'
        ec = C['green'] if c else C['red']
        tc = C['green'] if c else C['red']
        box(ax, 1.0 + i * 1.3, 3.6, 1.1, 0.45, t, fc, ec, tc, 10)

    # Predict
    arrow(ax, 5, 3.6, 5, 3.1, C['purple'])
    ax.text(6.5, 3.25, '③ Model predicts original at ALL positions', fontsize=12, fontweight='bold', color=C['purple'])
    pred = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    for i, t in enumerate(pred):
        box(ax, 1.0 + i * 1.3, 2.4, 1.1, 0.45, t, '#1a1a3e', C['purple'], C['purple'], 10)
        if noisy_cl[i] == 0:
            ax.text(1.55 + i * 1.3, 2.2, '↑ must predict!', fontsize=7, color=C['orange'], ha='center')

    # Loss
    arrow(ax, 5, 2.4, 5, 1.8, C['accent'])
    box(ax, 1.0, 0.8, 12, 0.9, '', C['card'], C['accent'])
    ax.text(7, 1.5, '④ Loss = weighted cross-entropy at ALL positions', fontsize=12, fontweight='bold', color=C['accent'], ha='center')
    ax.text(7, 1.05, 'L = w(t) × Σᵢ -log p_θ(x₀ⁱ | x_t)       Backprop → update weights', fontsize=11, color=C['cyan'], ha='center', fontfamily='monospace')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '17_training.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 18. ONE DENOISING STEP (detailed pipeline)
# ============================================================
def fig18_one_step():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_title('Anatomy of ONE Denoising Step', fontsize=18, fontweight='bold', color=C['text'], pad=15)

    stages = [
        (0.5, 7, 3.5, 0.7, '① Self-Conditioning\nAdd previous predictions to input', C['pink']),
        (0.5, 5.8, 3.5, 0.7, '② Embed + Timestep\nToken embed + time embed + conditioning', C['blue']),
        (5, 5.8, 4.5, 0.7, '③ Forward Pass (Denoiser)\nBidirectional attention + Encoder KV', C['green']),
        (5, 4.6, 4.5, 0.7, '④ Logits → Temperature\nlogits / τ(step) → softmax', C['pink']),
        (5, 3.4, 4.5, 0.7, '⑤ Sample Tokens\nMultinomial sampling from probabilities', C['orange']),
        (5, 2.2, 4.5, 0.7, '⑥ Entropy Check\nSort by confidence, accept within budget', C['yellow']),
        (10.5, 2.2, 5, 0.7, '⑦ Re-noise Rejected\nReplace with fresh random tokens', C['red']),
        (10.5, 3.4, 5, 0.7, '⑧ Stability Check\nAll tokens same as last step?', C['teal']),
        (10.5, 4.6, 5, 0.7, '⑨ Entropy Check\nAvg entropy < 0.005?', C['cyan']),
        (10.5, 5.8, 5, 0.7, '⑩ Decision\nBoth pass → STOP, else → next step', C['green']),
    ]
    for x, y, w, h, text, col in stages:
        box(ax, x, y, w, h, text, C['card'], col, col, 9)

    # Arrows
    arrow(ax, 2.25, 7, 2.25, 6.5, C['pink'])
    arrow(ax, 4, 6.15, 5, 6.15, C['muted'])
    arrow(ax, 7.25, 5.8, 7.25, 5.3, C['green'])
    arrow(ax, 7.25, 4.6, 7.25, 4.1, C['pink'])
    arrow(ax, 7.25, 3.4, 7.25, 2.9, C['orange'])
    arrow(ax, 9.5, 2.55, 10.5, 2.55, C['yellow'])
    arrow(ax, 13, 2.9, 13, 3.4, C['red'])
    arrow(ax, 13, 4.1, 13, 4.6, C['teal'])
    arrow(ax, 13, 5.3, 13, 5.8, C['cyan'])

    # Labels
    box(ax, 0.5, 0.5, 7, 1.0, '', '#0d3320', C['green'], alpha=0.3)
    ax.text(4, 1.25, '💡 All 256 tokens processed simultaneously in one forward pass', fontsize=11, color=C['green'], ha='center', fontweight='bold')
    ax.text(4, 0.75, 'This is why diffusion is compute-bound (good!) instead of memory-bound', fontsize=10, color=C['muted'], ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '18_one_step.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 19. ADAPTIVE STOPPING
# ============================================================
def fig19_adaptive_stopping():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stability over steps
    steps = range(1, 11)
    changed = [200, 120, 60, 25, 10, 3, 0, 0, 0, 0]
    ax1.bar(steps, changed, color=[C['red'] if c > 0 else C['green'] for c in changed],
            alpha=0.7, edgecolor=[C['red'] if c > 0 else C['green'] for c in changed], linewidth=2)
    ax1.axhline(y=0.5, color=C['green'], linewidth=2, linestyle='--', alpha=0.5)
    ax1.annotate('Stable!\n(0 changes)', xy=(7, 0), xytext=(7.5, 80),
                fontsize=12, color=C['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=2))
    ax1.set_xlabel('Denoising Step', fontsize=13)
    ax1.set_ylabel('Tokens Changed', fontsize=13)
    ax1.set_title('Stability Check\n(tokens changing per step)', fontsize=14, fontweight='bold', color=C['text'])
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    for s in ['left','bottom']: ax1.spines[s].set_color('#475569')
    ax1.grid(True, alpha=0.06, color='#475569')

    # Entropy over steps
    avg_entropy = [8.5, 5.2, 2.8, 1.2, 0.4, 0.05, 0.003, 0.002, 0.001, 0.001]
    ax2.plot(steps, avg_entropy, 'o-', color=C['cyan'], linewidth=2.5, markersize=8)
    ax2.axhline(y=0.005, color=C['yellow'], linewidth=2, linestyle='--')
    ax2.text(1.5, 0.02, 'threshold = 0.005', fontsize=10, color=C['yellow'], fontweight='bold')
    ax2.annotate('Below threshold!\nSTOP', xy=(7, 0.003), xytext=(8, 2),
                fontsize=12, color=C['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=2))
    ax2.set_xlabel('Denoising Step', fontsize=13)
    ax2.set_ylabel('Average Entropy', fontsize=13)
    ax2.set_title('Confidence Check\n(average prediction entropy)', fontsize=14, fontweight='bold', color=C['text'])
    ax2.set_yscale('log')
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    for s in ['left','bottom']: ax2.spines[s].set_color('#475569')
    ax2.grid(True, alpha=0.06, color='#475569')

    fig.suptitle('Adaptive Stopping: Stop Early When Converged', fontsize=16, fontweight='bold', color=C['text'], y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '19_adaptive_stopping.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 20. SELF-CORRECTION (the magic)
# ============================================================
def fig20_self_correction():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
    ax.set_title('Self-Correction: The Magic of Uniform Diffusion', fontsize=17, fontweight='bold', color=C['text'], pad=15)

    # Step 1: wrong token
    ax.text(0.3, 4.3, 'Step 2:', fontsize=11, fontweight='bold', color=C['orange'])
    toks1 = ['The', 'dog', 'sat', 'on', 'the', 'mat']
    cl1 = [1, 0.5, 1, 1, 1, 1]
    for i, (t, c) in enumerate(zip(toks1, cl1)):
        if c >= 1: fc, ec, tc = '#0d3320', C['green'], C['green']
        else: fc, ec, tc = '#2d1f0a', C['orange'], C['orange']
        box(ax, 1.5 + i*1.3, 4.0, 1.1, 0.5, t, fc, ec, tc, 10)
    ax.text(9.5, 4.2, '"dog" was predicted but\nentropy was HIGH (3.2 bits)\n→ rejected & re-noised!', fontsize=9, color=C['orange'])

    # Arrow down
    arrow(ax, 5, 4.0, 5, 3.5, C['yellow'])

    # Step 2: re-noised
    ax.text(0.3, 3.0, 'Re-noise:', fontsize=11, fontweight='bold', color=C['red'])
    toks2 = ['The', 'xyz', 'sat', 'on', 'the', 'mat']
    cl2 = [1, 0, 1, 1, 1, 1]
    for i, (t, c) in enumerate(zip(toks2, cl2)):
        if c >= 1: fc, ec, tc = '#0d3320', C['green'], C['green']
        else: fc, ec, tc = '#2a1215', C['red'], C['red']
        box(ax, 1.5 + i*1.3, 2.6, 1.1, 0.5, t, fc, ec, tc, 10)
    ax.text(9.5, 2.8, '"dog" → replaced with\nrandom token "xyz"', fontsize=9, color=C['red'])

    # Arrow down
    arrow(ax, 5, 2.6, 5, 2.1, C['green'])

    # Step 3: corrected
    ax.text(0.3, 1.6, 'Step 3:', fontsize=11, fontweight='bold', color=C['green'])
    toks3 = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    for i, t in enumerate(toks3):
        box(ax, 1.5 + i*1.3, 1.2, 1.1, 0.5, t, '#0d3320', C['green'], C['green'], 10)
    ax.text(9.5, 1.4, 'With more context, model\nnow predicts "cat" with\nhigh confidence → accepted! ✓', fontsize=9, color=C['green'])

    # Key insight
    box(ax, 1.0, 0.1, 12, 0.6, '💡 Unlike masked diffusion, uniform diffusion can FIX mistakes by re-noising and trying again!',
        '#0d3320', C['green'], C['green'], 11)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '20_self_correction.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# 21. SUMMARY COMPARISON TABLE
# ============================================================
def fig21_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    ax.set_title('DiffusionGemma vs Autoregressive: Full Comparison', fontsize=17, fontweight='bold', color=C['text'], pad=20)

    props = ['Tokens per forward pass', 'Forward passes for N tokens', 'Single-user latency',
             'Self-correction', 'Attention (generation)', 'GPU bottleneck',
             'Multi-user throughput', 'Can fix mistakes']
    ar = ['1', 'N (one per token)', 'HIGH', 'No ✗', 'Causal (left-only)', 'Memory-bound',
          'High ✓', 'No ✗']
    diff = ['256', 'S ≪ N (few steps)', 'LOW ✓', 'Yes ✓', 'Bidirectional (all)', 'Compute-bound ✓',
            'Lower', 'Yes ✓']
    ar_c = [C['red']]*3 + [C['red'], C['muted'], C['red'], C['green'], C['red']]
    di_c = [C['green']]*3 + [C['green'], C['green'], C['green'], C['orange'], C['green']]

    rows = len(props)
    col_w = [4.0, 3.5, 3.5]
    for r in range(rows + 1):
        y = 5.0 - r * 0.55
        if r == 0:
            fc = C['card2']
            texts = ['Property', 'Autoregressive', 'DiffusionGemma']
            cols = [C['text'], C['red'], C['green']]
            fw = 'bold'
        else:
            fc = C['card'] if r % 2 == 0 else C['bg']
            texts = [props[r-1], ar[r-1], diff[r-1]]
            cols = [C['muted'], ar_c[r-1], di_c[r-1]]
            fw = 'normal'
        x = 1.5
        for ci, (txt, col, w) in enumerate(zip(texts, cols, col_w)):
            rect = FancyBboxPatch((x, y), w, 0.5, boxstyle='round,pad=0.05',
                                  facecolor=fc, edgecolor='#1a1d2e', linewidth=1)
            ax.add_patch(rect)
            ax.text(x + w/2, y + 0.25, txt, ha='center', va='center',
                    fontsize=10, fontweight=fw, color=col)
            x += w + 0.15

    ax.set_xlim(0, 14); ax.set_ylim(-0.5, 5.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '21_comparison.png'), dpi=180, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    funcs = [
        (fig01_why_diffusion, '01_why_diffusion.png'),
        (fig02_roofline, '02_roofline.png'),
        (fig03_continuous_diffusion, '03_continuous_diffusion.png'),
        (fig04_noise_schedule, '04_noise_schedule.png'),
        (fig05_masked_vs_uniform, '05_masked_vs_uniform.png'),
        (fig06_uniform_math, '06_uniform_math.png'),
        (fig07_gemma4, '07_gemma4_base.png'),
        (fig08_attention_masks, '08_attention_masks.png'),
        (fig09_encoder_denoiser, '09_encoder_denoiser.png'),
        (fig10_kv_cache, '10_kv_cache.png'),
        (fig11_full_architecture, '11_full_architecture.png'),
        (fig12_self_conditioning, '12_self_conditioning.png'),
        (fig13_temperature, '13_temperature.png'),
        (fig14_entropy, '14_entropy_acceptance.png'),
        (fig15_denoising_trace, '15_denoising_trace.png'),
        (fig16_multi_canvas, '16_multi_canvas.png'),
        (fig17_training, '17_training.png'),
        (fig18_one_step, '18_one_step_anatomy.png'),
        (fig19_adaptive_stopping, '19_adaptive_stopping.png'),
        (fig20_self_correction, '20_self_correction.png'),
        (fig21_comparison, '21_comparison.png'),
    ]
    print(f"Generating {len(funcs)} diagrams...")
    for fn, name in funcs:
        fn()
        print(f"  ✓ {name}")
    print(f"\nAll {len(funcs)} diagrams saved to: {OUT}/")
