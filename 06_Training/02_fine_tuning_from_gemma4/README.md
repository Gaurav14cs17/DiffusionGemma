# Chapter 6.2: Fine-Tuning from Gemma 4 — Standing on the Shoulders of Giants

> *"Don't train a diffusion model from scratch — adapt one that already understands language."*

![Gemma 4 Base Model](../../diagrams/07_gemma4_base.png)

![Training Process](../../diagrams/17_training.png)

---

## 6.2.1 Why Not Train from Scratch?

Training a discrete diffusion model from scratch faces several challenges:

```
┌──────────────────────────────────────────────────────────────┐
│               CHALLENGES OF TRAINING FROM SCRATCH             │
│                                                               │
│  1. COMPUTE COST                                             │
│     Training a 26B model from scratch requires               │
│     thousands of TPU/GPU hours on trillions of tokens        │
│                                                               │
│  2. HARDER OBJECTIVE                                         │
│     Uniform-state diffusion requires the model to:           │
│     - Identify which tokens are noise                         │
│     - Predict correct replacements                            │
│     - Handle variable noise levels                            │
│     This is harder than simple next-token prediction          │
│                                                               │
│  3. LANGUAGE UNDERSTANDING                                   │
│     The model needs deep linguistic knowledge                 │
│     that takes billions of examples to develop                │
│                                                               │
│  4. DATA REQUIREMENTS                                        │
│     Would need the same massive datasets                      │
│     used for Gemma 4's pre-training                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Solution**: Start from Gemma 4's pre-trained checkpoint and fine-tune it for the diffusion task.

---

## 6.2.2 What Changes vs. What Stays

```
  GEMMA 4 26B A4B (PRE-TRAINED)
  ┌──────────────────────────────────────────┐
  │                                           │
  │  ┌─────────────────────────┐              │
  │  │   Token Embeddings      │  SHARED ✓    │
  │  │   (vocabulary → d_model)│  (unchanged) │
  │  └─────────────────────────┘              │
  │                                           │
  │  ┌─────────────────────────┐              │
  │  │   Transformer Layers    │  FINE-TUNED  │
  │  │   (attention + MoE FFN) │              │
  │  │                         │  Adapts to:  │
  │  │   W_Q, W_K, W_V, W_O   │  - Bidir attn│
  │  │   Expert weights        │  - Noisy input│
  │  │   Router weights        │  - All-pos   │
  │  │                         │    prediction │
  │  └─────────────────────────┘              │
  │                                           │
  │  ┌─────────────────────────┐              │
  │  │   LM Head               │  SHARED ✓    │
  │  │   (d_model → vocab)     │  (unchanged) │
  │  └─────────────────────────┘              │
  │                                           │
  │  ┌─────────────────────────┐              │
  │  │   NEW: Self-Conditioning│  TRAINED     │
  │  │   FFNN (d → d)          │  from scratch│
  │  └─────────────────────────┘              │
  │                                           │
  │  ┌─────────────────────────┐              │
  │  │   NEW: Timestep         │  TRAINED     │
  │  │   Embedding (1 → d)     │  from scratch│
  │  └─────────────────────────┘              │
  │                                           │
  └──────────────────────────────────────────┘
```

### Summary of Changes

| Component | Status | Details |
|-----------|--------|---------|
| Token embeddings | Unchanged | Shared with pre-trained model |
| LM Head | Unchanged | Same vocabulary projection |
| Attention weights | Fine-tuned | Learn bidirectional patterns |
| MoE experts | Fine-tuned | Adapt to noisy input distribution |
| Router weights | Fine-tuned | May route differently for noisy tokens |
| Self-conditioning FFNN | New | Small network, trained from scratch |
| Timestep embedding | New | Encodes noise level $t$ |

---

## 6.2.3 Timestep Conditioning

The model needs to know **how noisy** the canvas is (the noise level $t$). This is injected via a **timestep embedding**:

$$
\text{emb}(t) = \text{MLP}\left(\text{sinusoidal}(t)\right) \in \mathbb{R}^d
$$

The sinusoidal encoding (similar to positional encoding in transformers):

$$
\text{sinusoidal}(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \qquad \text{sinusoidal}(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

This timestep embedding is added to the hidden states or used to modulate layer norms (adaptive layer norm):

$$
h^{(\ell)} \leftarrow \text{LayerNorm}(h^{(\ell)}) \cdot (1 + \gamma(t)) + \beta(t)
$$

where $\gamma(t), \beta(t)$ are learned functions of the timestep.

---

## 6.2.3b Timestep Conditioning: Full Mathematical Trace

This section computes the **entire timestep conditioning pipeline** numerically — from raw scalar $t$ to its effect inside every transformer layer.

### Part A: Sinusoidal Encoding (Dimension by Dimension)

**Input:** noise level $t = 0.4$, embedding dimension $d = 8$.

**Formula** (same structure as positional encoding in the original Transformer):

$$
\text{sinusoidal}(t)_{2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad \text{sinusoidal}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

Each pair $(2i, 2i+1)$ shares the same frequency $\omega_i = 1 / 10000^{2i/d}$.

| Index | Freq. denominator $10000^{2i/8}$ | Argument $t / \cdot$ | sin | cos |
|-------|--------------------------------------|--------------------------|-----|-----|
| 0 | $10000^{0} = 1$ | $0.4 / 1 = 0.4$ | $\sin(0.4) = 0.389$ | $\cos(0.4) = 0.921$ |
| 1 | $10000^{0} = 1$ | $0.4 / 1 = 0.4$ | (paired with index 0) | (paired with index 0) |
| 2 | $10000^{2/8} = 10$ | $0.4 / 10 = 0.04$ | $\sin(0.04) = 0.040$ | $\cos(0.04) = 0.999$ |
| 3 | $10000^{2/8} = 10$ | $0.4 / 10 = 0.04$ | (paired with index 2) | (paired with index 2) |
| 4 | $10000^{4/8} = 100$ | $0.4 / 100 = 0.004$ | $\sin(0.004) = 0.004$ | $\cos(0.004) = 1.000$ |
| 5 | $10000^{4/8} = 100$ | $0.4 / 100 = 0.004$ | (paired with index 4) | (paired with index 4) |
| 6 | $10000^{6/8} = 1000$ | $0.4 / 1000 = 0.0004$ | $\sin(0.0004) = 0.0004$ | $\cos(0.0004) = 1.000$ |
| 7 | $10000^{6/8} = 1000$ | $0.4 / 1000 = 0.0004$ | (paired with index 6) | (paired with index 6) |

**Resulting 8-dimensional sinusoidal vector:**

$$
\boxed{\text{sinusoidal}(0.4) = [0.389,\; 0.921,\; 0.040,\; 0.999,\; 0.004,\; 1.000,\; 0.0004,\; 1.000]}
$$

**Why multiple frequencies?** Low-frequency dimensions (indices 0–1) encode coarse noise-level information; high-frequency dimensions (indices 6–7) let the model distinguish nearby values of $t$ (e.g., 0.39 vs 0.41). This mirrors how positional encodings separate nearby positions.

---

### Part B: Timestep MLP — From 8-D to Model Dimension

The sinusoidal vector is passed through a small MLP to produce a $d$-dimensional conditioning vector:

$$
\text{emb}(t) = W_2 \cdot \text{GELU}(W_1 \cdot \text{sinusoidal}(t) + b_1) + b_2
$$

where $W_1 \in \mathbb{R}^{d \times 8}$, $W_2 \in \mathbb{R}^{d \times d}$, $b_1 \in \mathbb{R}^d$, $b_2 \in \mathbb{R}^d$.

**Numerical trace** (illustrative with $d = 4$ for readability; real model uses $d \approx 5376$):

```
  Input:  sinusoidal(0.4) = [0.389, 0.921, 0.040, 0.999, 0.004, 1.000, 0.0004, 1.000]  ∈ ℝ⁸

  Step B.1 — Linear projection:
    z = W₁ · sinusoidal(0.4) + b₁

    Suppose W₁ · [...] + b₁ = [0.82, -0.31, 0.55, 0.17]   (example 4-dim hidden pre-activation)

  Step B.2 — GELU activation:
    GELU(x) ≈ x · Φ(x)   (Φ = standard normal CDF)

    GELU(0.82)  ≈ 0.67
    GELU(-0.31) ≈ -0.11
    GELU(0.55)  ≈ 0.39
    GELU(0.17)  ≈ 0.10

    h = [0.67, -0.11, 0.39, 0.10]

  Step B.3 — Output projection:
    emb(0.4) = W₂ · h + b₂

    Suppose result:  emb(0.4) ≈ [0.15, -0.08, 0.22, 0.11]  ∈ ℝ⁴
```

In the full model, `emb(t)` is a **single vector** broadcast to all $L$ canvas positions (same noise level everywhere on the canvas).

---

### Part C: Injecting `emb(t)` into the Transformer

Two common mechanisms (DiffusionGemma may use one or a combination):

#### Option 1: Additive Injection (Input Layer)

$$
\hat{e}_i = \text{Embed}(x_t^i) + \text{emb}(t) + c_i
$$

```
  Position i input construction (t = 0.4):

    Embed("dog")     = [0.7,  0.3,  0.0, -0.2]    ← token embedding (shared with Gemma 4)
  + emb(0.4)         = [0.15, -0.08, 0.22, 0.11]   ← "40% noise" signal
  + c_i              = [0.0,   0.0,  0.0,  0.0]    ← self-conditioning (zero on first pass)
  ─────────────────────────────────────────────────
    ê_i              = [0.85,  0.22, 0.22, -0.09]
```

Every token embedding is **shifted** by the same `emb(t)`, telling all positions "the canvas is at noise level 0.4."

#### Option 2: Adaptive Layer Norm (AdaLN) — Per Layer

At each transformer layer $\ell$:

$$
h^{(\ell)} \leftarrow \gamma^{(\ell)}(t) \odot \text{LayerNorm}(h^{(\ell)}) + \beta^{(\ell)}(t)
$$

where:

$$
\gamma^{(\ell)}(t) = W_\gamma^{(\ell)} \cdot \text{emb}(t), \qquad \beta^{(\ell)}(t) = W_\beta^{(\ell)} \cdot \text{emb}(t)
$$

```
  Layer ℓ, t = 0.4:

    h^(ℓ)           = [1.2, -0.5, 0.8, 0.3, ...]     ← pre-norm hidden state
    LayerNorm(h)    = [0.9, -0.4, 0.6, 0.2, ...]     ← normalized

    γ(t)            = [1.05, 0.98, 1.02, 1.01, ...]  ← scale (near 1.0 at moderate noise)
    β(t)            = [0.02, -0.01, 0.03, 0.00, ...] ← shift

    h_out           = γ(t) ⊙ LayerNorm(h) + β(t)
                    = [0.95, -0.39, 0.61, 0.20, ...]
```

AdaLN lets each layer **modulate** its normalization differently depending on $t$, giving finer control than a single additive shift.

---

### Part D: Why the Model Must Know the Noise Level

The denoising task is **fundamentally non-stationary**: the optimal prediction strategy depends on how corrupted the input is.

| Noise level $t$ | $\bar{\alpha}_t$ | Fraction of clean tokens | What the model should do |
|---------------------|----------------------|--------------------------|--------------------------|
| **0.1** (light) | ≈ 0.90 | ~90% original | **Trust the canvas.** Most tokens are correct; predict the same token or minor corrections. Low entropy outputs. |
| **0.3** | ≈ 0.70 | ~70% original | **Selective denoising.** Identify the ~30% corrupted positions using bidirectional context; leave clean positions alone. |
| **0.5** (medium) | ≈ 0.50 | ~50% original | **Balanced inference.** Equal mix of noise and signal — rely on both local context and global language priors. |
| **0.7** | ≈ 0.30 | ~30% original | **Prior-driven.** Most tokens are random; lean heavily on learned grammar, semantics, and encoder query context. |
| **0.9** (heavy) | ≈ 0.10 | ~10% original | **Generate from scratch.** Canvas is nearly pure noise; treat it like a blank slate conditioned on the encoder KV cache. |

**Without timestep conditioning**, the model sees `["The", "dog", "sat", "The"]` and cannot tell whether:
- $t = 0.1$: "dog" is a rare corruption to fix → predict "cat"
- $t = 0.9$: "dog" might be correct because almost everything is random → predict from context

The timestep embedding **disambiguates** these scenarios, enabling a single network to handle all noise levels.

---

### Part E: Behavioral Table Across Noise Levels

Concrete example: canvas `["The", "???", "sat", "???"]` with query *"Complete the sentence: The cat sat on the mat."*

| $t$ | $\bar{\alpha}_t$ | `emb(t)` character (informal) | Model behavior at pos 1 (`???`) | Model behavior at pos 3 (`???`) | Expected confidence |
|---------|----------------------|-------------------------------|--------------------------------|--------------------------------|---------------------|
| 0.1 | 0.90 | "mostly clean" | Predict "cat" (fix corruption) | Predict "on" (fix corruption) | High (0.85+) |
| 0.3 | 0.70 | "lightly noisy" | Predict "cat" from surrounding "The...sat" | Predict "on" from grammar | Medium-high (0.70+) |
| 0.5 | 0.50 | "half noise" | Use encoder + canvas context | Use phrase patterns ("sat on") | Medium (0.50–0.70) |
| 0.7 | 0.30 | "mostly noise" | Rely on "The ___ sat" template | Rely on "sat ___ the" template | Medium-low (0.35–0.55) |
| 0.9 | 0.10 | "nearly pure noise" | Generate from language prior + query | Generate from language prior + query | Low-moderate (0.25–0.45) |

**Training implication:** Because $t \sim \mathcal{U}(0,1)$ during training, the timestep MLP and AdaLN weights see **every row** of this table millions of times, learning to smoothly interpolate between "trust the input" and "ignore the noise."

**Inference mapping:** During denoising step $s$ of $S$, the inference code sets:

$$
t_s = 1 - \frac{s}{S}
$$

so early steps use $t \approx 1$ (heavy noise, trust priors) and late steps use $t \approx 0$ (light noise, trust canvas). The timestep embedding is what makes this schedule work.

---

## 6.2.4 Benefits of Fine-Tuning from Pre-trained Checkpoint

```
  ┌─────────────────────────────────────────────────────────────┐
  │              WHAT GEMMA 4 BRINGS TO DIFFUSION                │
  │                                                              │
  │  1. LANGUAGE UNDERSTANDING                                   │
  │     ✓ Grammar, syntax, semantics already learned            │
  │     ✓ World knowledge encoded in weights                    │
  │     ✓ Instruction following capabilities                    │
  │                                                              │
  │  2. EFFICIENT REPRESENTATIONS                                │
  │     ✓ Token embeddings are already meaningful               │
  │     ✓ Attention patterns capture linguistic structure        │
  │     ✓ MoE routing already specialized                       │
  │                                                              │
  │  3. FASTER CONVERGENCE                                       │
  │     ✓ Model starts "close" to a good solution               │
  │     ✓ Only needs to learn:                                   │
  │       - How to handle bidirectional attention                │
  │       - How to predict all positions simultaneously          │
  │       - How to work with noisy input                        │
  │     ✓ Much less training data needed                        │
  │                                                              │
  │  4. SHARED ENCODER                                           │
  │     ✓ Encoder mode uses original causal attention           │
  │     ✓ No additional training needed for encoder             │
  │     ✓ Already excellent at processing queries               │
  │                                                              │
  └─────────────────────────────────────────────────────────────┘
```

---

## 6.2.5 Training Data and Process

The fine-tuning process uses:
- **Data**: Supervised instruction-following data (query-response pairs)
- **Noise schedule**: Continuous-time schedule $\sigma(t)$ with log-linear or cosine shape
- **Batch**: Each training example consists of a (query, response) pair with a random noise level
- **Mixed training**: The model alternates between encoder mode (processing query) and denoiser mode (denoising response)

```
  TRAINING EXAMPLE:
  
  Query:    "What is the capital of France?"
  Response: "The capital of France is Paris."
  
  Noise level: t = 0.4  →  ᾱ_t = 0.67
  
  Encoder input:  "What is the capital of France?"
                   (processed with causal attention → KV cache)
  
  Denoiser input: "The xyz of France rand Paris ."
                   ↑ clean  ↑ noise  ↑ clean ↑noise ↑clean ↑clean
  
  Target:         "The capital of France is Paris ."
  
  Loss: Cross-entropy at all positions, weighted by w(t)
```

---

**Next**: [../07_Full_Pipeline/01_end_to_end_walkthrough.md](../../07_Full_Pipeline/01_end_to_end_walkthrough/) — Complete inference trace.
