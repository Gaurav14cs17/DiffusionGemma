# Chapter 3.2: Masked Diffusion Language Models (MDLM)

> *"What if [MASK] is the noise?"*

![Masked vs Uniform Diffusion](../../diagrams/05_masked_vs_uniform.png)

---

## 3.2.1 Core Idea

In **Masked Diffusion**, the "noise" is the replacement of real tokens with a special **[MASK]** token. The forward process progressively masks more and more tokens until the entire sequence is masked. The reverse process learns to unmask them.

```
  Forward (Masking):
  ┌───────────────────────────────────────────────────┐
  │ t=0: "The  cat  sat  on   the  mat"               │
  │ t=1: "The  [M]  sat  on   the  mat"               │
  │ t=2: "The  [M]  [M]  on   [M]  mat"               │
  │ t=3: "[M]  [M]  [M]  [M]  [M]  mat"               │
  │ t=4: "[M]  [M]  [M]  [M]  [M]  [M]"               │
  └───────────────────────────────────────────────────┘

  Reverse (Unmasking):
  ┌───────────────────────────────────────────────────┐
  │ t=4: "[M]  [M]  [M]  [M]  [M]  [M]"               │
  │ t=3: "[M]  [M]  [M]  on   [M]  [M]"               │
  │ t=2: "[M]  cat  [M]  on   [M]  mat"               │
  │ t=1: "The  cat  sat  on   [M]  mat"               │
  │ t=0: "The  cat  sat  on   the  mat"               │
  └───────────────────────────────────────────────────┘
```

---

## 3.2.2 The Absorbing-State Transition Matrix

The vocabulary is extended to $K + 1$ states (adding [MASK] as state $K+1$). The transition matrix at each step is:

$$
\mathbf{Q}_t^{\text{mask}} = \begin{bmatrix}
\alpha_t & 0 & \cdots & 0 & \beta_t \\
0 & \alpha_t & \cdots & 0 & \beta_t \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & \alpha_t & \beta_t \\
0 & 0 & \cdots & 0 & 1
\end{bmatrix}
$$

where:
- $\alpha_t = 1 - \beta_t$: probability of staying the same
- $\beta_t$: probability of transitioning to [MASK]
- Last row: [MASK] is **absorbing** — once masked, always masked

### Cumulative Transition

$$
\bar{\mathbf{Q}}_t^{\text{mask}} = \begin{bmatrix}
\bar{\alpha}_t & 0 & \cdots & 0 & 1 - \bar{\alpha}_t \\
0 & \bar{\alpha}_t & \cdots & 0 & 1 - \bar{\alpha}_t \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & \bar{\alpha}_t & 1 - \bar{\alpha}_t \\
0 & 0 & \cdots & 0 & 1
\end{bmatrix}
$$

For any token $x_0 = k$ (not [MASK]):

$$
q(x_t = k \mid x_0 = k) = \bar{\alpha}_t, \qquad q(x_t = \text{[MASK]} \mid x_0 = k) = 1 - \bar{\alpha}_t
$$

```
  Probability
  1.0 ──┤●─────────────────────────────
        │ ╲  P(x_t = [MASK])           ╱●
  0.8 ──┤  ╲                         ╱
        │   ╲                       ╱
  0.6 ──┤    ╲                    ╱
        │     ╲                 ╱
  0.4 ──┤      ╲ P(x_t = x₀) ╱
        │       ╲           ╱
  0.2 ──┤        ╲        ╱
        │         ╲     ╱
  0.0 ──┤──────────●───────────────────
        └──┬──┬──┬──┬──┬──┬──→ t
           0     T/2     T
```

### Numerical Trace: Masking a 6-Token Sentence

Let's make the absorbing-state dynamics concrete. We use a vocabulary of size $K = 5$:

| Token | ID |
|-------|-----|
| The   | 0  |
| cat   | 1  |
| sat   | 2  |
| on    | 3  |
| mat   | 4  |

The sentence **"The cat sat on the mat"** has 6 positions. We treat the second **"the"** as token ID 0 (same type as **"The"**; casing is ignored here).

#### At $t = 0$: One-Hot Vectors

Each position is a one-hot vector in $\mathbb{R}^5$:

| Position | Token | $\mathbf{x}_0$ |
|----------|-------|----------------|
| 1        | The   | $[1, 0, 0, 0, 0]^\top$ |
| 2        | cat   | $[0, 1, 0, 0, 0]^\top$ |
| 3        | sat   | $[0, 0, 1, 0, 0]^\top$ |
| 4        | on    | $[0, 0, 0, 1, 0]^\top$ |
| 5        | the   | $[1, 0, 0, 0, 0]^\top$ |
| 6        | mat   | $[0, 0, 0, 0, 1]^\top$ |

#### Computing $\bar{\alpha}_t$ with Schedule $\beta_t = [0.2, 0.3, 0.4, 0.6]$

For each discrete step $s$, $\alpha_s = 1 - \beta_s$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:

| Step $t$ | $\beta_t$ | $\alpha_t = 1-\beta_t$ | $\bar{\alpha}_t = \prod_{s \leq t} \alpha_s$ |
|----------|-----------|------------------------|-----------------------------------------------|
| 1        | 0.2       | 0.8                    | $0.8$                                         |
| 2        | 0.3       | 0.7                    | $0.8 \times 0.7 = 0.56$                       |
| 3        | 0.4       | 0.6                    | $0.56 \times 0.6 = 0.336$                     |
| 4        | 0.6       | 0.4                    | $0.336 \times 0.4 = 0.1344$                   |

At any step $t$, every **non-[MASK]** token independently satisfies:

$$
P(\text{stay}) = q(x_t^i = x_0^i \mid x_0^i) = \bar{\alpha}_t, \qquad P(\text{mask}) = 1 - \bar{\alpha}_t
$$

#### At $t = 2$: Per-Position Probabilities ($\bar{\alpha}_2 = 0.56$)

| Position | $x_0^i$ | $P(\text{stay} = x_0^i)$ | $P(\text{mask})$ |
|----------|---------|--------------------------|------------------|
| 1        | The     | 0.56                     | 0.44             |
| 2        | cat     | 0.56                     | 0.44             |
| 3        | sat     | 0.56                     | 0.44             |
| 4        | on      | 0.56                     | 0.44             |
| 5        | the     | 0.56                     | 0.44             |
| 6        | mat     | 0.56                     | 0.44             |

**Independence check:** Because each position draws from its own Bernoulli trial with the same rate, the joint probability of any particular mask pattern factorizes:

$$
P(x_t \mid x_0) = \prod_{i=1}^{6} \left[\bar{\alpha}_2 \cdot \mathbb{1}[x_t^i = x_0^i] + (1-\bar{\alpha}_2) \cdot \mathbb{1}[x_t^i = \text{[MASK]}]\right]
$$

For example, masking only positions 2 and 4:

$$
P(\text{mask at 2,4 only}) = (0.56)^4 \times (0.44)^2 \approx 0.056 \times 0.194 \approx 0.0109
$$

#### One Concrete Draw

Suppose we sample independently at $t=2$:

| Position | Draw | Outcome |
|----------|------|---------|
| 1        | stay | The     |
| 2        | **mask** | [M] |
| 3        | stay | sat     |
| 4        | **mask** | [M] |
| 5        | stay | the     |
| 6        | stay | mat     |

**Resulting sequence:**

```
  t=0:  "The   cat   sat   on    the   mat"
  t=2:  "The   [M]   sat   [M]   the   mat"
```

Positions 2 and 4 are now **[MASK]**; the model must later recover **cat** (ID 1) and **on** (ID 3). All other positions remain clean.

---

## 3.2.3 CTMC Rate Matrix for Masked Diffusion

In continuous time, the rate matrix is:

$$
\mathbf{R}^{\text{mask}} = \begin{bmatrix}
-1 & 0 & \cdots & 0 & 1 \\
0 & -1 & \cdots & 0 & 1 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & -1 & 1 \\
0 & 0 & \cdots & 0 & 0
\end{bmatrix}
$$

The forward marginal at time $t \in [0, 1]$:

$$
q(x_t = k \mid x_0 = k) = e^{-\sigma(t)}, \qquad q(x_t = \text{[MASK]} \mid x_0 = k) = 1 - e^{-\sigma(t)}
$$

where $\sigma(t)$ is a monotonically increasing noise schedule with $\sigma(0) = 0$ and $\sigma(1) \rightarrow \infty$.

---

## 3.2.4 Training Objective for MDLM

The model $p_\theta(x_0^i \mid x_t)$ predicts the original token at each masked position. The loss is:

$$
\boxed{\mathcal{L}_{\text{MDLM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1)}\; \mathbb{E}_{x_t \sim q(x_t \mid x_0)}\left[\sigma'(t) \sum_{i : x_t^i = \text{[MASK]}} -\log p_\theta(x_0^i \mid x_t)\right]}
$$

**Breaking this down:**
- $t \sim \mathcal{U}(0,1)$: sample a random time
- $x_t \sim q(x_t \mid x_0)$: mask tokens according to the noise schedule
- $\sigma'(t)$: weighting factor (derivative of the noise schedule)
- Sum over masked positions: only compute loss on [MASK] tokens
- $-\log p_\theta$: cross-entropy loss for each masked position

### Worked Example: Computing MDLM Loss

We continue the masked sequence from above at $t=2$:

```
  x_t:  "The   [M]   sat   [M]   the   mat"
              pos2        pos4
```

The model outputs logits **only at masked positions** (positions 2 and 4). Clean positions contribute zero loss.

#### Position 2 — Target: **cat** (ID = 1)

Model logits: $\mathbf{z}_2 = [1.2,\; 3.5,\; 0.8,\; 0.3,\; -0.5]$

Softmax probabilities $p_k = e^{z_k} / \sum_j e^{z_j}$:

| Token | ID | Logit | $e^{z_k}$ | $p_k$ |
|-------|-----|-------|-----------|-------|
| The   | 0   | 1.2   | 3.320     | 0.082 |
| **cat** | **1** | **3.5** | **33.115** | **0.815** |
| sat   | 2   | 0.8   | 2.226     | 0.055 |
| on    | 3   | 0.3   | 1.350     | 0.033 |
| mat   | 4   | −0.5  | 0.607     | 0.015 |

Cross-entropy at position 2:

$$
\ell_2 = -\log p_\theta(x_0^2 = \text{cat} \mid x_t) = -\log(0.815) \approx 0.204
$$

#### Position 4 — Target: **on** (ID = 3)

Model logits: $\mathbf{z}_4 = [0.5,\; 0.1,\; 0.3,\; 2.8,\; 1.0]$

| Token | ID | Logit | $e^{z_k}$ | $p_k$ |
|-------|-----|-------|-----------|-------|
| The   | 0   | 0.5   | 1.649     | 0.071 |
| cat   | 1   | 0.1   | 1.105     | 0.048 |
| sat   | 2   | 0.3   | 1.350     | 0.058 |
| **on** | **3** | **2.8** | **16.445** | **0.707** |
| mat   | 4   | 1.0   | 2.718     | 0.117 |

$$
\ell_4 = -\log(0.707) \approx 0.346
$$

#### Total Loss with $\sigma'(t)$ Weighting

Suppose we are in continuous time with noise schedule $\sigma(t) = -\log(1 - 0.9t)$, so that $e^{-\sigma(t)} = 1 - 0.9t$ matches a 90% mask rate at $t=1$. At $t=0.5$:

$$
\sigma(0.5) = -\log(0.55) \approx 0.598, \qquad \sigma'(0.5) = \frac{0.9}{0.55} \approx 1.636
$$

The per-example loss is:

$$
\mathcal{L} = \sigma'(t) \sum_{i : x_t^i = \text{[MASK]}} \ell_i = 1.636 \times (0.204 + 0.346) = 1.636 \times 0.550 \approx \mathbf{0.900}
$$

#### Contrast with Uniform Diffusion

| Aspect | MDLM (Masked) | Uniform Diffusion (UDLM) |
|--------|---------------|--------------------------|
| Positions with loss | Only masked (2 of 6 here) | **All 6 positions** |
| Target | Original token $x_0^i$ | Original token $x_0^i$ |
| Noise | Replace with [MASK] | Replace with random token |
| Efficiency | Fewer CE terms per step | More CE terms, but enables self-correction |

In uniform diffusion, even clean-looking positions like **"sat"** at position 3 might have been corrupted and would still receive a loss term. MDLM's sparsity is computationally cheaper but structurally commits to predictions once unmasked.

---

## 3.2.5 The Limitation: No Self-Correction

```
  Step 1: [M] [M] [M] [M] [M] [M]
             ↓ model predicts "dog" with 60% confidence
  Step 2: [M] dog [M] [M] [M] [M]
                        ↑
                  LOCKED IN! Can't change "dog" to "cat"
                  even if later context suggests "cat"

  Step 3: [M] dog [M] on  [M] [M]
                 ↑
          Now "dog sat on" makes sense, but what if
          the full sentence should be "cat sat on the mat"?
          
  ✗ Cannot go back and fix "dog" → "cat"
```

Once a [MASK] token is replaced, it becomes a real token and cannot be masked again (since masking only happens in the forward direction). This is structurally identical to the autoregressive commitment problem.

### Connection to BERT

BERT's **Masked Language Modeling (MLM)** objective looks remarkably similar to MDLM — and for good reason. BERT is essentially MDLM with a **fixed, single noise level**.

#### BERT's MLM Objective

During pretraining, BERT selects 15% of tokens and replaces them with [MASK] (80%), a random token (10%), or keeps them unchanged (10%). The model predicts the original token at masked positions:

$$
\mathcal{L}_{\text{BERT}} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_0^i \mid x_{\mathcal{M}})
$$

where $\mathcal{M}$ is the set of masked positions.

#### MDLM as a Generalization

MDLM replaces the fixed 15% mask rate with a **continuous schedule** $\sigma(t)$:

| Feature | BERT MLM | MDLM |
|---------|----------|------|
| Mask rate | Fixed $\approx 15\%$ | Varies: $1 - e^{-\sigma(t)}$ from 0% to 100% |
| Time $t$ | Implicit (one step) | Explicit $t \sim \mathcal{U}(0,1)$ |
| Loss weighting | Uniform | $\sigma'(t)$ — emphasizes informative noise levels |
| Training | Single corruption pass | Full diffusion trajectory |

#### Reduction at the "Right" Noise Level

Set BERT's effective mask probability to $p_{\text{mask}} = 0.15$. In MDLM, choose $t^*$ such that:

$$
1 - e^{-\sigma(t^*)} = p_{\text{mask}} \quad \Longrightarrow \quad e^{-\sigma(t^*)} = 0.85
$$

At this noise level, each token is independently [MASK] with probability 0.15 — exactly BERT's corruption distribution (ignoring BERT's 10% random-token and 10% unchanged quirks). The MDLM loss at $t^*$ with $\sigma'(t^*)$ absorbed into the learning rate becomes:

$$
\mathcal{L}_{\text{MDLM}} \Big|_{t = t^*} \;\approx\; \mathbb{E}_{x_t}\left[-\sum_{i : x_t^i = \text{[MASK]}} \log p_\theta(x_0^i \mid x_t)\right] \;\cong\; \mathcal{L}_{\text{BERT}}
$$

**Key insight:** BERT trains the model to denoise at **one** difficulty level. MDLM trains across **all** difficulty levels, preparing the model for iterative unmasking during generation — but inherits BERT's inability to revise committed tokens.

---

## 3.2.6 Reverse Process: How Unmasking Works

Training teaches $p_\theta(x_0^i \mid x_t)$ — given a partially masked sequence, predict the original token at each [MASK]. At **inference**, we run this in reverse: start fully masked and progressively reveal tokens.

### Algorithm Overview

```
  Input:   prompt tokens (fixed) + [M] [M] [M] ... (to generate)
  Output:  complete sequence

  for t = T, T-1, ..., 1:
      1. Run model on current sequence x_t
      2. At each [MASK] position i, get p_θ(x_0^i = k | x_t) for all k
      3. Unmask a subset of positions (see criterion below)
      4. Replace chosen [MASK] tokens with argmax or sampled token
```

### Confidence-Based Unmasking Criterion

A common strategy unmasks the most confident predictions first. Define:

$$
c_i = \max_k p_\theta(x_0^i = k \mid x_t)
$$

**Unmask position $i$** if:

$$
c_i > \tau
$$

where $\tau$ is a confidence threshold (often annealed: low at early steps, high at late steps). Alternatively, unmask the top-$k$ most confident positions per step.

### 3-Step Unmasking Trace

Start from the fully masked state of our 6-token example (positions 1–6 all [MASK]):

```
  Step 0 (t=4):  [M]  [M]  [M]  [M]  [M]  [M]
```

**Step 1** — Model predictions (confidence $c_i$):

| Pos | Top prediction | $c_i$ | Action ($\tau_1 = 0.5$) |
|-----|----------------|-------|-------------------------|
| 1   | The            | 0.72  | ✓ unmask                |
| 2   | cat            | 0.68  | ✓ unmask                |
| 3   | sat            | 0.81  | ✓ unmask                |
| 4   | on             | 0.45  | ✗ stay masked           |
| 5   | the            | 0.38  | ✗ stay masked           |
| 6   | mat            | 0.55  | ✓ unmask                |

```
  Step 1 (t=3):  The  cat  sat  [M]  [M]  mat
```

**Step 2** — Remaining masks; context now includes **"The cat sat … mat"**:

| Pos | Top prediction | $c_i$ | Action ($\tau_2 = 0.4$) |
|-----|----------------|-------|-------------------------|
| 4   | on             | 0.88  | ✓ unmask                |
| 5   | the            | 0.52  | ✓ unmask                |

```
  Step 2 (t=2):  The  cat  sat  on   the  mat
```

**Step 3** — No [MASK] tokens remain; stop.

```
  Step 3 (t=1):  The  cat  sat  on   the  mat   ✓ done
```

### Why Order Matters (and Why It's a Liability)

In Step 1, position 2 was unmasked as **cat** with 68% confidence. If the true sentence required **dog**, the model cannot revisit that decision — the token is locked. Later context (**"… sat on the mat"**) might have disambiguated **cat** vs **dog**, but the reverse process never re-corrupts position 2.

This greedy unmasking is fast and simple, but it is the operational mirror of the **no self-correction** limitation from Section 3.2.5. Uniform-state diffusion (next chapter) keeps tokens corruptible throughout, allowing the model to revise mistakes.

---

**Next**: [03_uniform_state_diffusion.md](../../03_uniform_state_diffusion/03_uniform_state_diffusion/) — How DiffusionGemma solves this with uniform noise.
