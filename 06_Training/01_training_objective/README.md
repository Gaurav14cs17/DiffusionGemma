# Chapter 6.1: Training Objective — The Discrete Diffusion ELBO

> *"Teach the model to denoise by showing it noise at every level."*

![Training Process](../../diagrams/17_training.png)

---

## 6.1.1 What the Model Learns

DiffusionGemma's denoiser learns a single skill: given a **noisy canvas** at noise level $t$, predict the **clean tokens** $x_0$.

$$
p_\theta(x_0 \mid x_t, t) = \prod_{i=1}^{L} p_\theta(x_0^i \mid x_t, t)
$$

Each position is predicted independently (given the full noisy context through bidirectional attention).

---

## 6.1.2 The Discrete ELBO Derivation

### Starting Point

We want to maximize the log-likelihood of the data:

$$
\log p_\theta(x_0) = \log \sum_{x_1, \ldots, x_T} p_\theta(x_0, x_1, \ldots, x_T)
$$

### Step 1: Introduce the Forward Process

$$
\log p_\theta(x_0) = \log \sum_{x_{1:T}} \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \cdot q(x_{1:T} \mid x_0)
$$

### Step 2: Apply Jensen's Inequality

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] = -\mathcal{L}_{\text{ELBO}}
$$

### Step 3: Decompose for Discrete State Space

For discrete diffusion with uniform noise, the ELBO decomposes as:

$$
\boxed{\mathcal{L}_{\text{ELBO}} = \underbrace{D_{\text{KL}}(q(x_T \mid x_0)\, \|\, p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(x_t \mid x_0)}\left[D_{\text{KL}}\big(q(x_{t-1} \mid x_t, x_0)\, \|\, p_\theta(x_{t-1} \mid x_t)\big)\right]}_{L_{t-1}} + \underbrace{\mathbb{E}_{q(x_1 \mid x_0)}\left[-\log p_\theta(x_0 \mid x_1)\right]}_{L_0}}
$$

### Each Term Explained

```
┌──────────────────────────────────────────────────────────────────┐
│                    ELBO TERMS (DISCRETE)                          │
├──────────┬───────────────────────────────────────────────────────┤
│          │                                                       │
│  L_T     │  KL between q(x_T|x₀) and prior p(x_T)             │
│  (prior) │  For uniform diffusion, as T→∞:                      │
│          │  q(x_T|x₀) → Uniform(1,...,K) = p(x_T)              │
│          │  So L_T → 0 (no learnable parameters)                │
│          │                                                       │
├──────────┼───────────────────────────────────────────────────────┤
│          │                                                       │
│  L_{t-1} │  Can the model reverse step t → t-1?                │
│  (denoi- │  For discrete states, q(x_{t-1}|x_t,x₀) can be     │
│  sing)   │  computed exactly using Bayes' rule:                  │
│          │                                                       │
│          │       q(x_t|x_{t-1}) · q(x_{t-1}|x₀)               │
│          │  q(x_{t-1}|x_t,x₀) = ─────────────────────          │
│          │                         q(x_t|x₀)                    │
│          │                                                       │
│          │  All three terms are known from the transition        │
│          │  matrices Q_t and Q̄_t                                │
│          │                                                       │
├──────────┼───────────────────────────────────────────────────────┤
│          │                                                       │
│  L_0     │  Reconstruction: given barely-noisy x₁,              │
│  (recon) │  recover x₀. This is a cross-entropy loss.           │
│          │                                                       │
└──────────┴───────────────────────────────────────────────────────┘
```

---

## 6.1.3 The Concrete Reverse Posterior

For uniform-noise discrete diffusion, the reverse posterior has a closed form:

$$
q(x_{t-1} = j \mid x_t = i, x_0 = k) = \frac{q(x_t = i \mid x_{t-1} = j) \cdot q(x_{t-1} = j \mid x_0 = k)}{q(x_t = i \mid x_0 = k)}
$$

### Computing Each Term

**Forward single step** $q(x_t = i \mid x_{t-1} = j)$:

$$
= \begin{cases}
1 - \beta_t + \beta_t / K & \text{if } i = j \\
\beta_t / K & \text{if } i \neq j
\end{cases}
$$

**Forward marginal** $q(x_{t-1} = j \mid x_0 = k)$:

$$
= \begin{cases}
\bar{\alpha}_{t-1} + (1 - \bar{\alpha}_{t-1})/K & \text{if } j = k \\
(1 - \bar{\alpha}_{t-1})/K & \text{if } j \neq k
\end{cases}
$$

**Forward marginal** $q(x_t = i \mid x_0 = k)$:

$$
= \begin{cases}
\bar{\alpha}_t + (1 - \bar{\alpha}_t)/K & \text{if } i = k \\
(1 - \bar{\alpha}_t)/K & \text{if } i \neq k
\end{cases}
$$

The ratio gives a tractable posterior that the model learns to approximate.

---

## 6.1.3b Step-by-Step: Computing the Loss for One Training Example

This section walks through **one complete training example** with real numbers — from corruption to backpropagation — so the abstract ELBO becomes concrete.

### Setup

| Symbol | Value |
|--------|-------|
| Clean sequence $x_0$ | `["The", "cat", "sat", "on"]` — length $L = 4$ |
| Vocabulary $K = 5$ | `{The=0, cat=1, sat=2, on=3, dog=4}` |
| Noise level $t$ | `0.4` |
| Survival probability $\bar{\alpha}_t = \bar{\alpha}_{0.4}$ | `0.6` (token kept with prob 0.6, corrupted with prob 0.4) |
| Corruption rule | With prob $1 - \bar{\alpha}_t = 0.4$: replace with `Uniform{0,…,4}` |

**Token IDs for reference:**

```
  The=0   cat=1   sat=2   on=3   dog=4
```

---

### Step 1: Corrupt Each Position (Forward Process)

Each position is corrupted **independently** via the marginal:

$$
q(x_t^i = j \mid x_0^i = k) = \bar{\alpha}_t \cdot \delta_{jk} + \frac{1 - \bar{\alpha}_t}{K}
$$

| Position $i$ | $x_0^i$ | Keep? (prob 0.6) | Corrupt? (prob 0.4) | Draw | $x_t^i$ |
|------------------|-------------|------------------|---------------------|------|-------------|
| 0 | The (0) | ✓ keep | — | — | **The** (0) |
| 1 | cat (1) | — | ✓ corrupt | `Uniform(0,4)` → 4 | **dog** (4) |
| 2 | sat (2) | ✓ keep | — | — | **sat** (2) |
| 3 | on (3) | — | ✓ corrupt | `Uniform(0,4)` → 0 | **The** (0) |

**Noisy canvas:**

$$
x_t = \texttt{["The", "dog", "sat", "The"]}
$$

**Corruption mask** (whether $x_t^i \neq x_0^i$):

```
  Position:     0       1       2       3
  x₀:          The     cat     sat     on
  x_t:         The     dog     sat     The
  Corrupted?    NO      YES     NO      YES
```

---

### Step 2: Forward Pass → Logits → Softmax

The denoiser receives $(x_t, t)$ plus encoder KV cache and outputs logits $\ell_i \in \mathbb{R}^K$ at each position. We apply softmax to get $p_\theta(x_0^i = k \mid x_t)$.

**Position 0** — noisy token is "The" (correct), model should predict $x_0^0 =$ The (0):

```
  logits₀ = [4.0,  0.1,  0.2, -0.1, -0.5]

  softmax:  exp(4.0)=54.60,  exp(0.1)=1.11,  exp(0.2)=1.22,  exp(-0.1)=0.90,  exp(-0.5)=0.61
            sum = 58.43

  p₀ = [54.60/58.43,  1.11/58.43,  1.22/58.43,  0.90/58.43,  0.61/58.43]
     = [0.93,  0.02,  0.02,  0.02,  0.01]
```

**Position 1** — noisy token is "dog" (wrong), target is $x_0^1 =$ cat (1):

```
  logits₁ = [0.5,  2.5,  0.3,  0.1,  1.8]

  p₁ = softmax(logits₁) = [0.11,  0.52,  0.09,  0.07,  0.21]
                              ↑
                         model correctly ranks "cat" highest (52%)
```

**Position 2** — noisy token is "sat" (correct), target is sat (2):

```
  logits₂ = [0.1,  0.2,  3.8,  0.1,  0.0]

  p₂ = [0.02,  0.02,  0.90,  0.02,  0.02]
```

**Position 3** — noisy token is "The" (wrong), target is on (3):

```
  logits₃ = [0.3,  0.1,  0.2,  3.2,  0.5]

  p₃ = [0.05,  0.04,  0.04,  0.82,  0.05]
                              ↑
                         model correctly ranks "on" highest (82%)
```

---

### Step 3: Cross-Entropy Loss at Each Position

The target is always the **clean** token $x_0^i$, regardless of whether $x_t^i$ is corrupted:

$$
\text{CE}_i = -\log p_\theta(x_0^i \mid x_t) = -\log p_i[x_0^i]
$$

| Position | Target $x_0^i$ | $p_i[x_0^i]$ | $\text{CE}_i = -\log p_i[x_0^i]$ |
|----------|-------------------|------------------|--------------------------------------|
| 0 | The (0) | 0.93 | $-\log(0.93) = 0.073$ |
| 1 | cat (1) | 0.52 | $-\log(0.52) = 0.654$ |
| 2 | sat (2) | 0.90 | $-\log(0.90) = 0.105$ |
| 3 | on (3) | 0.82 | $-\log(0.82) = 0.198$ |

**Unweighted sum:** $0.073 + 0.654 + 0.105 + 0.198 = 1.030$

**Interpretation:**
- Position 0 and 2 have **low loss** — the noisy token already matches $x_0$, so the model mainly confirms it.
- Position 1 has **high loss** — the model must infer "cat" from context despite seeing "dog".
- Position 3 has **moderate loss** — the model predicts "on" correctly but with only 82% confidence (noisy "The" is misleading).

---

### Step 4: Apply Per-Position ELBO Weights $w_t^i$

From the continuous-time ELBO (Section 6.1.4), each position receives a weight based on whether it was corrupted. For uniform noise with vocabulary size $K$:

$$
w_t^i = \begin{cases}
\dfrac{K - 1}{K} = \dfrac{4}{5} = 0.8 & \text{if } x_t^i \neq x_0^i \quad \text{(corrupted — hard denoising task)} \\[8pt]
\dfrac{1}{K} = \dfrac{1}{5} = 0.2 & \text{if } x_t^i = x_0^i \quad \text{(clean — easy confirmation task)}
\end{cases}
$$

**Derivation sketch:** The ELBO ratio term for position $i$ is:

$$
\frac{q(x_t^i = j \mid x_0^i)}{q(x_t^i = x_0^i \mid x_0^i)} = \frac{(1-\bar{\alpha}_t)/K}{\bar{\alpha}_t + (1-\bar{\alpha}_t)/K}
$$

When $x_t^i \neq x_0^i$, the observed noisy token $j$ is a corruption, and the numerator equals $(1-\bar{\alpha}_t)/K$ while the denominator is the marginal at the clean token. Summing over all $j \neq x_0^i$ gives a factor of $(K-1)$, yielding weight $\propto (K-1)/K$. When $x_t^i = x_0^i$, only the $j = x_0^i$ term contributes, giving weight $\propto 1/K$.

| Position | Corrupted? | Weight $w_t^i$ | $w_t^i \times \text{CE}_i$ |
|----------|------------|-------------------|----------------------------------|
| 0 | No | 0.2 | $0.2 \times 0.073 = 0.0146$ |
| 1 | **Yes** | **0.8** | $0.8 \times 0.654 = 0.5232$ |
| 2 | No | 0.2 | $0.2 \times 0.105 = 0.0210$ |
| 3 | **Yes** | **0.8** | $0.8 \times 0.198 = 0.1584$ |

**Weighted loss (before schedule factor):**

$$
\sum_{i=1}^{L} w_t^i \cdot \text{CE}_i = 0.0146 + 0.5232 + 0.0210 + 0.1584 = \boxed{0.715}
$$

**Key insight:** Corrupted positions (1 and 3) contribute **88%** of the weighted loss despite being only half the positions. Training focuses gradient signal on the hard denoising task.

---

### Step 5: Multiply by Schedule Weight $\sigma'(t)$

The full continuous-time ELBO includes the derivative of the noise schedule:

$$
\mathcal{L}_{\text{example}} = \sigma'(t) \sum_{i=1}^{L} w_t^i \cdot \left(-\log p_\theta(x_0^i \mid x_t)\right)
$$

**Noise schedule (log-linear):** DiffusionGemma uses a continuous schedule $\sigma(t)$ with $\bar{\alpha}_t = e^{-\sigma(t)}$. For the log-linear choice $\bar{\alpha}_t = 1 - t$:

$$
\sigma(t) = -\log(1 - t), \qquad \sigma'(t) = \frac{1}{1 - t}
$$

At $t = 0.4$:

$$
\sigma'(0.4) = \frac{1}{1 - 0.4} = \frac{1}{0.6} \approx 1.667
$$

**Total loss for this single example:**

$$
\mathcal{L}_{\text{example}} = \sigma'(0.4) \times 0.715 = 1.667 \times 0.715 \approx \boxed{1.192}
$$

In full training, this is averaged over batches, sequences, and random draws of $t \sim \mathcal{U}(0,1)$:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{b=1}^{B} \sigma'(t_b) \sum_{i=1}^{L} w_{t_b}^i \cdot \left(-\log p_\theta(x_0^{i,b} \mid x_t^{(b)})\right)
$$

---

### Step 6: How Gradients Flow — $\partial \mathcal{L} / \partial \theta$

The loss depends on model parameters $\theta$ only through the predicted probabilities $p_\theta(x_0^i \mid x_t)$. The gradient path is:

```
  x₀ (fixed target, no grad)
    │
    ▼
  L = σ'(t) · Σᵢ w_t^i · [-log p_θ(x₀ⁱ | x_t)]
    │
    │  ∂L/∂p_i[k] = -σ'(t) · w_t^i / p_i[k]   (when k = x₀ⁱ)
    ▼
  p_i = softmax(logits_i)          ← differentiable
    │
    │  ∂p_i[k]/∂ℓ_i[j] = p_i[k]·(δ_{kj} - p_i[j])
    ▼
  logits_i = W_head · h_i^(L) + b_head    ← LM head (shared with Gemma 4)
    │
    │  ∂L/∂h_i^(L) flows through W_head
    ▼
  h_i^(L) = Transformer(..., emb(t), self_cond, encoder_KV)
    │
    ├──→ ∂L/∂W_Q, ∂L/∂W_K, ∂L/∂W_V, ∂L/∂W_O   (attention, fine-tuned)
    ├──→ ∂L/∂W_expert, ∂L/∂W_router          (MoE FFN, fine-tuned)
    ├──→ ∂L/∂W_timestep                       (timestep MLP, trained from scratch)
    └──→ ∂L/∂W_selfcond                       (self-conditioning FFNN, trained from scratch)
```

**Per-position gradient (cross-entropy + softmax):**

For position $i$ with target $y = x_0^i$:

$$
\frac{\partial \mathcal{L}}{\partial \ell_i[k]} = \sigma'(t) \cdot w_t^i \cdot \left(p_i[k] - \mathbb{1}[k = y]\right)
$$

**Numerical check at position 1** ($y = 1$ = cat, $w = 0.8$, $\sigma' = 1.667$):

```
  p₁ = [0.11, 0.52, 0.09, 0.07, 0.21]
  target y = 1

  ∂L/∂ℓ₁[k] = 1.667 × 0.8 × (p₁[k] - 𝟙[k=1])

  k=0: 1.667 × 0.8 × (0.11 - 0) = 0.147
  k=1: 1.667 × 0.8 × (0.52 - 1) = -0.640   ← largest magnitude (push logit[cat] up)
  k=2: 1.667 × 0.8 × (0.09 - 0) = 0.120
  k=3: 1.667 × 0.8 × (0.07 - 0) = 0.093
  k=4: 1.667 × 0.8 × (0.21 - 0) = 0.280
```

The gradient **increases** the logit for "cat" (target) and **decreases** relative pressure on other tokens. Because position 1 is corrupted and has weight 0.8, this gradient is **4× stronger** than an identical error at a clean position (weight 0.2).

**Backpropagation step:**

```
  1. Compute ∂L/∂ℓ_i for all i = 1,…,L
  2. Backprop through LM head → ∂L/∂h_i^(L)
  3. Backprop through L transformer layers (bidirectional attention + MoE)
  4. Accumulate ∂L/∂θ for all trainable parameters
  5. Optimizer step: θ ← θ - η · ∇_θ L     (η = learning rate)
```

After many such examples across all noise levels $t \in [0,1]$, the model learns to denoise at **every** corruption level — which is exactly what inference needs.

---

## 6.1.4 The Continuous-Time ELBO (What DiffusionGemma Uses)

In practice, DiffusionGemma uses the **continuous-time** formulation, which avoids discretizing into $T$ steps:

$$
\boxed{\mathcal{L}_{\text{CT}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\sigma'(t) \sum_{i=1}^{L} \sum_{j \neq x_0^i} \frac{q(x_t^i = j \mid x_0^i)}{q(x_t^i = x_0^i \mid x_0^i)} \left(-\log p_\theta(x_0^i \mid x_t)\right)\right]}
$$

### Simplified Form

For uniform noise, the ratio simplifies:

$$
\frac{q(x_t^i = j \mid x_0^i)}{q(x_t^i = x_0^i \mid x_0^i)} = \frac{(1 - \bar{\alpha}_t)/K}{\bar{\alpha}_t + (1 - \bar{\alpha}_t)/K}
$$

When $K$ is large (e.g., 256,000), this approximately equals:

$$
\approx \frac{1 - \bar{\alpha}_t}{K \bar{\alpha}_t}
$$

The overall loss then becomes a **weighted cross-entropy** over all positions:

$$
\mathcal{L} \approx \mathbb{E}_{t, x_t}\left[w(t) \sum_{i=1}^{L} -\log p_\theta(x_0^i \mid x_t)\right]
$$

where $w(t)$ is a time-dependent weight derived from the noise schedule.

---

## 6.1.5 Training Algorithm

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    TRAINING STEP                                │
  │                                                                 │
  │  1. Sample a batch of clean sequences:                          │
  │     x₀ ~ p_data                                                │
  │     Each x₀ is a sequence of L tokens                          │
  │                                                                 │
  │  2. Sample a noise level:                                       │
  │     t ~ Uniform(0, 1)                                           │
  │                                                                 │
  │  3. Corrupt each token independently:                           │
  │     For each position i:                                        │
  │       With probability (1 - ᾱ_t):                              │
  │         x_t^i ← random token from Uniform(1,...,K)             │
  │       With probability ᾱ_t:                                    │
  │         x_t^i ← x₀^i (keep original)                          │
  │                                                                 │
  │  4. Optional: Self-conditioning                                 │
  │     With 50% probability, run one forward pass first            │
  │     and use its predictions as conditioning for the real pass   │
  │                                                                 │
  │  5. Run the denoiser (bidirectional Gemma 4):                  │
  │     logits = model(x_t, t, encoder_KV, self_cond)             │
  │                                                                 │
  │  6. Compute weighted cross-entropy loss:                        │
  │     L = w(t) · Σᵢ [-log softmax(logits_i)[x₀^i]]             │
  │                                                                 │
  │  7. Backpropagate and update θ                                  │
  │                                                                 │
  └────────────────────────────────────────────────────────────────┘
```

---

## 6.1.6 Comparison with Autoregressive Training

```
  AUTOREGRESSIVE TRAINING:             DIFFUSION TRAINING:
  
  Input:  [The]  [cat]  [sat]          Input:  [The] [rand] [sat] [rand] [the]
  Target: [cat]  [sat]  [on]           Target: [The] [cat]  [sat] [on]   [the]
  
  Loss: CE at each shifted position    Loss: Weighted CE at ALL positions
  
  Attention: Causal (lower triangular) Attention: Bidirectional (full)
  
  1 training example per sequence      1 training example per (sequence, t) pair
```

| Property | Autoregressive | Diffusion |
|----------|---------------|-----------|
| Positions in loss | All (shifted by 1) | All (weighted by noise level) |
| Noise in input | None | Random tokens at level $t$ |
| Attention type | Causal | Bidirectional |
| Conditioning | Previous clean tokens | Noisy canvas + encoder KV |

---

**Next**: [02_fine_tuning_from_gemma4](../02_fine_tuning_from_gemma4/) — Adapting the pre-trained model.
