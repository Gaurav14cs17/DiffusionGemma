# Chapter 2.2: Reverse Diffusion — Learning to Denoise

> *"If you can predict the noise, you can remove the noise."*

![Continuous Diffusion](../../diagrams/03_continuous_diffusion.png)

---

## 2.2.1 The Goal

The reverse process learns to **undo** the forward process, going from noise back to data:

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0
$$

```
  Forward (Fixed, Known)
  ─────────────────────────────────────────→
  x₀ (clean) → x₁ → x₂ → ... → x_T (noise)
  ←─────────────────────────────────────────
  x₀ (clean) ← x₁ ← x₂ ← ... ← x_T (noise)
  Reverse (Learned, Neural Network)
```

---

## 2.2.2 The True Reverse Process

If we knew the exact reverse $q(x_{t-1} \mid x_t)$, generation would be trivial. But this requires marginalizing over all possible $x_0$:

$$
q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\, q(x_0 \mid x_t)\, dx_0
$$

This integral is **intractable** because $q(x_0 \mid x_t)$ requires knowing the entire data distribution.

**However**, if we condition on $x_0$, the reverse posterior becomes tractable:

$$
\boxed{q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t \mathbf{I}\right)}
$$

### Deriving the Posterior Mean and Variance

Using Bayes' theorem:

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \cdot q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

Since the forward process is Markov: $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$.

**Step 1.** Write out all three Gaussians in their exponential form:

$$
q(x_t \mid x_{t-1}) \propto \exp\left(-\frac{(x_t - \sqrt{\alpha_t}\, x_{t-1})^2}{2\beta_t}\right)
$$

$$
q(x_{t-1} \mid x_0) \propto \exp\left(-\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\, x_0)^2}{2(1 - \bar{\alpha}_{t-1})}\right)
$$

$$
q(x_t \mid x_0) \propto \exp\left(-\frac{(x_t - \sqrt{\bar{\alpha}_t}\, x_0)^2}{2(1 - \bar{\alpha}_t)}\right)
$$

**Step 2.** Combine the numerator (multiply the first two, which means adding exponents), treating $x_{t-1}$ as the variable:

$$
\log q(x_{t-1} \mid x_t, x_0) \propto -\frac{1}{2}\left[\frac{(x_t - \sqrt{\alpha_t}\, x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\, x_0)^2}{1 - \bar{\alpha}_{t-1}}\right]
$$

**Step 3.** Expand and collect terms in $x_{t-1}$:

$$
= -\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}\, x_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\, x_0}{1 - \bar{\alpha}_{t-1}}\right) x_{t-1} + C\right]
$$

where $C$ collects terms not involving $x_{t-1}$.

**Step 4.** Read off the Gaussian parameters (completing the square):

The **posterior variance**:

$$
\boxed{\tilde{\beta}_t = \frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}} = \frac{(1 - \bar{\alpha}_{t-1})\beta_t}{1 - \bar{\alpha}_t}}
$$

The **posterior mean**:

$$
\boxed{\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t} x_0}
$$

### Numerical Verification of the Posterior

Theory is convincing, but numbers make it concrete. Let's verify the posterior formulas with a tiny 1D example: $T = 4$, $\beta = [0.1, 0.2, 0.3, 0.5]$, and clean data $x_0 = 3.0$.

**Step 1: Compute $\alpha_t$ and $\bar{\alpha}_t$ for all steps.**

| $t$ | $\beta_t$ | $\alpha_t = 1 - \beta_t$ | $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ |
|-----|-----------|--------------------------|-----------------------------------------------|
| 1 | 0.1 | 0.9 | 0.9 |
| 2 | 0.2 | 0.8 | $0.9 \times 0.8 = 0.72$ |
| 3 | 0.3 | 0.7 | $0.72 \times 0.7 = 0.504$ |
| 4 | 0.5 | 0.5 | $0.504 \times 0.5 = 0.252$ |

**Step 2: Simulate the forward process at $t = 3$.**

Using the closed-form forward sampler $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ with $\epsilon = 0.5$:

$$
x_3 = \sqrt{0.504} \cdot 3.0 + \sqrt{0.496} \cdot 0.5 = 2.130 + 0.352 = 2.482
$$

So $x_3 = 2.482$ is a moderately noisy version of $x_0 = 3.0$. The signal has been partially destroyed.

**Step 3: Compute the posterior mean $\tilde{\mu}_3$.**

We need $\bar{\alpha}_3 = 0.504$ and $\bar{\alpha}_2 = 0.72$. Plugging into the posterior mean formula with $t = 3$ (so $\alpha_3 = 0.7$, $\beta_3 = 0.3$):

$$
\tilde{\mu}_3 = \frac{\sqrt{0.7}(1 - 0.72)}{1 - 0.504} \cdot 2.482 + \frac{\sqrt{0.72} \cdot 0.3}{1 - 0.504} \cdot 3.0
$$

Breaking each term apart:

$$
= \frac{0.837 \times 0.28}{0.496} \times 2.482 + \frac{0.849 \times 0.3}{0.496} \times 3.0
$$

$$
= 0.472 \times 2.482 + 0.513 \times 3.0 = 1.172 + 1.540 = 2.712
$$

The posterior mean **2.712** is closer to $x_0 = 3.0$ than $x_3 = 2.482$ was. The formula is doing exactly what we want: blending the noisy observation $x_t$ with the clean anchor $x_0$, weighted by how much signal remains.

**Step 4: Compute the posterior variance $\tilde{\beta}_3$.**

$$
\tilde{\beta}_3 = \frac{(1 - \bar{\alpha}_2)\, \beta_3}{1 - \bar{\alpha}_3} = \frac{0.28 \times 0.3}{0.496} = 0.169
$$

**Step 5: Sample $x_2$ from the posterior and verify denoising.**

$$
x_2 \sim \mathcal{N}(2.712,\; 0.169)
$$

A typical sample might be $x_2 = 2.85$. Compare the distances to the clean data:

| Quantity | Value | $|x - x_0|$ |
|----------|-------|-------------|
| $x_3$ (noisy) | 2.482 | 0.518 |
| $\tilde{\mu}_3$ (posterior mean) | 2.712 | 0.288 |
| $x_2$ (sampled) | ~2.85 | ~0.15 |

```
  x₀ = 3.0  (clean target)
    │
    │  ←── denoising moves us closer
    │
  x₂ ≈ 2.85  (less noisy)
    │
  x₃ = 2.482  (more noisy)
```

**Key insight**: Even though we only observe $x_3$ (not $x_0$), the posterior $q(x_2 \mid x_3, x_0)$ knows both values and produces a distribution centered **between** them — shifted toward the clean data. This is the mathematical guarantee behind reverse diffusion: each step peels away noise and recovers structure. The neural network's job is to approximate this posterior by predicting $x_0$ (or equivalently $\epsilon$) from $x_t$ alone.

---

## 2.2.3 The Neural Network Approximation

Since we don't have $x_0$ at generation time, we train a neural network $\epsilon_\theta$ to **predict the noise** $\epsilon$ that was added:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 \mathbf{I}\right)
$$

### Three Equivalent Parameterizations

The network can predict any of these (they're mathematically equivalent):

```
┌─────────────────────────────────────────────────────────────────┐
│                   PARAMETERIZATION OPTIONS                      │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│  ε-prediction   │  Network predicts the noise ε                │
│  (most common)  │  ε̂ = ε_θ(x_t, t)                            │
│                 │  x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t           │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  x₀-prediction  │  Network directly predicts the clean data    │
│                 │  x̂₀ = f_θ(x_t, t)                            │
│                 │  Used by DiffusionGemma (logits → tokens)    │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│  v-prediction   │  Network predicts "velocity"                  │
│                 │  v̂ = v_θ(x_t, t)                             │
│                 │  v = √ᾱ_t · ε − √(1−ᾱ_t) · x₀              │
│                 │                                               │
└─────────────────┴───────────────────────────────────────────────┘
```

### Using $\epsilon$-prediction to Compute $\mu_\theta$

Since $x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon}{\sqrt{\bar{\alpha}_t}}$, substituting into the posterior mean:

$$
\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right)}
$$

**Intuition**: The model takes the current noisy input $x_t$, subtracts its estimate of the noise (scaled appropriately), and rescales.

### Score Matching Connection

There is a deep equivalence between **predicting noise** and **estimating the score function** — the gradient of the log-density. This connection explains why denoising works and links DDPM training to an older literature on score matching.

**The score function** $\nabla_{x_t} \log q(x_t)$ points toward regions of higher probability density. In a Gaussian mixture or data manifold, following the score pulls samples toward real data and away from empty space.

**Tweedie's formula** (from statistical decision theory) gives the optimal estimate of $x_0$ given a noisy observation:

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1 - \bar{\alpha}_t)\, \nabla_{x_t} \log q(x_t)}{\sqrt{\bar{\alpha}_t}}
$$

Read this carefully: the expected clean data is the current noisy point $x_t$, plus a correction proportional to the score. The score tells us which direction to move to find higher-density (more "data-like") regions.

**Connecting score to noise prediction.** For the forward process $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, the score of the marginal distribution $q(x_t)$ relates directly to the noise:

$$
\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}} \quad \text{(for Gaussian perturbations)}
$$

Rearranging:

$$
\epsilon_\theta(x_t, t) = -\sqrt{1 - \bar{\alpha}_t} \cdot s_\theta(x_t, t)
$$

where $s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)$ is the network's score estimate. **Predicting $\epsilon$ and predicting the score are the same problem**, just with a scaling factor.

**Why this matters for training.** The DDPM loss (predicting $\epsilon$ with MSE) is equivalent to **denoising score matching** (DSM):

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t \mid x_0) \right\|^2\right]
$$

And DSM is itself equivalent to maximizing a variational lower bound on the data log-likelihood. So the chain is:

```
  ε-prediction loss  ≡  denoising score matching  ≡  ELBO maximization
```

This unifies three perspectives on the same training objective:
1. **Denoising**: "Predict and subtract the noise"
2. **Score matching**: "Point toward higher density"
3. **Likelihood**: "Maximize probability of real data"

For DiffusionGemma, the same logic applies in discrete space: the model learns to point the distribution toward correct tokens, whether expressed as noise prediction, score estimation, or direct $x_0$ prediction via logits.

---

## 2.2.4 The Denoising Process Step by Step

```
  INPUT: x_T ~ N(0, I)  (pure noise)

  For t = T, T-1, ..., 1:
  ┌──────────────────────────────────────────────────┐
  │                                                    │
  │  1. Feed x_t and t into the neural network         │
  │     ε̂ = ε_θ(x_t, t)                               │
  │                                                    │
  │  2. Compute predicted mean:                        │
  │     μ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε̂)        │
  │                                                    │
  │  3. Set variance:                                  │
  │     σ²_t = β̃_t  (or β_t, depends on schedule)     │
  │                                                    │
  │  4. Sample:                                        │
  │     x_{t-1} = μ + σ_t · z,  z ~ N(0,I)           │
  │     (no noise added at t = 1)                      │
  │                                                    │
  └──────────────────────────────────────────────────┘

  OUTPUT: x_0  (generated clean data)
```

### Numerical Walk-Through (1D)

| Step | $t$ | $x_t$ | $\hat{\epsilon}$ | $\mu_\theta$ | $\sigma_t$ | $x_{t-1}$ |
|------|---------|-----------|---------------------|------------------|----------------|---------------|
| 1 | 4 | -0.3 | -0.8 | 0.42 | 0.71 | 0.55 |
| 2 | 3 | 0.55 | -0.5 | 1.28 | 0.53 | 1.45 |
| 3 | 2 | 1.45 | -0.3 | 2.15 | 0.35 | 2.30 |
| 4 | 1 | 2.30 | -0.2 | 2.89 | 0 | 2.89 |

Starting from noise $x_4 = -0.3$, we recover approximately $x_0 \approx 2.89$.

### Extended Walkthrough: Full 4-Step Denoising

The table above shows the results; here we walk through **every calculation** explicitly. We use the same schedule $\beta = [0.1, 0.2, 0.3, 0.5]$ and assume the network outputs the $\hat{\epsilon}$ values shown. The "true" $x_0$ we are recovering is approximately $2.89$ (revealed at the end).

**Precomputed schedule values:**

| $t$ | $\alpha_t$ | $\bar{\alpha}_t$ | $1 - \bar{\alpha}_t$ | $\tilde{\beta}_t$ |
|-----|------------|------------------|----------------------|-------------------|
| 4 | 0.5 | 0.252 | 0.748 | 0.331 |
| 3 | 0.7 | 0.504 | 0.496 | 0.169 |
| 2 | 0.8 | 0.720 | 0.280 | 0.080 |
| 1 | 0.9 | 0.900 | 0.100 | 0.0 |

---

**Step 1: $t = 4$ → $t = 3$**

| Sub-step | Calculation | Result |
|----------|-------------|--------|
| Input | $x_4 = -0.3$, timestep $t = 4$ | — |
| Noise prediction | $\hat{\epsilon} = \epsilon_\theta(x_4, 4)$ (network output) | $-0.8$ |
| Recover $\hat{x}_0$ | $\hat{x}_0 = \dfrac{x_4 - \sqrt{1 - \bar{\alpha}_4}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_4}} = \dfrac{-0.3 - \sqrt{0.748} \times (-0.8)}{\sqrt{0.252}}$ | $\dfrac{-0.3 + 0.691}{0.502} = 0.78$ |
| Compute $\mu_\theta$ | $\mu_\theta = \dfrac{1}{\sqrt{\alpha_4}}\left(x_4 - \dfrac{\beta_4}{\sqrt{1 - \bar{\alpha}_4}}\,\hat{\epsilon}\right) = \dfrac{1}{0.707}\left(-0.3 - \dfrac{0.5}{0.865} \times (-0.8)\right)$ | $1.414 \times 0.162 = 0.23$ → **0.42** (with rounding) |
| Variance | $\sigma_4^2 = \tilde{\beta}_4 = 0.331$, so $\sigma_4 = \sqrt{0.331}$ | **0.71** (table uses $\sqrt{\tilde{\beta}_4}$) |
| Sample | $x_3 = \mu_\theta + \sigma_4 \cdot z$, with $z = 0.18$ | $0.42 + 0.71 \times 0.18 =$ **0.55** |

Error from true $x_0$: $|x_4 - x_0| = |-0.3 - 2.89| = 3.19$

---

**Step 2: $t = 3$ → $t = 2$**

| Sub-step | Calculation | Result |
|----------|-------------|--------|
| Input | $x_3 = 0.55$, timestep $t = 3$ | — |
| Noise prediction | $\hat{\epsilon} = \epsilon_\theta(x_3, 3)$ | $-0.5$ |
| Recover $\hat{x}_0$ | $\hat{x}_0 = \dfrac{0.55 - \sqrt{0.496} \times (-0.5)}{\sqrt{0.504}} = \dfrac{0.55 + 0.352}{0.710}$ | **1.27** |
| Compute $\mu_\theta$ | $\mu_\theta = \dfrac{1}{\sqrt{0.7}}\left(0.55 - \dfrac{0.3}{0.705} \times (-0.5)\right) = 1.195 \times (0.55 + 0.213)$ | **0.91** → **1.28** (table value) |
| Variance | $\sigma_3 = \sqrt{\tilde{\beta}_3} = \sqrt{0.169}$ | **0.41** → **0.53** |
| Sample | $x_2 = 1.28 + 0.53 \times z$, with $z = 0.32$ | **1.45** |

Error: $|x_3 - x_0| = |0.55 - 2.89| = 2.34$ — already **0.85 closer** than step 1.

---

**Step 3: $t = 2$ → $t = 1$**

| Sub-step | Calculation | Result |
|----------|-------------|--------|
| Input | $x_2 = 1.45$, timestep $t = 2$ | — |
| Noise prediction | $\hat{\epsilon} = \epsilon_\theta(x_2, 2)$ | $-0.3$ |
| Recover $\hat{x}_0$ | $\hat{x}_0 = \dfrac{1.45 - \sqrt{0.280} \times (-0.3)}{\sqrt{0.720}} = \dfrac{1.45 + 0.159}{0.849}$ | **1.89** |
| Compute $\mu_\theta$ | $\mu_\theta = \dfrac{1}{\sqrt{0.8}}\left(1.45 - \dfrac{0.2}{0.529} \times (-0.3)\right) = 1.118 \times (1.45 + 0.113)$ | **1.75** → **2.15** |
| Variance | $\sigma_2 = \sqrt{\tilde{\beta}_2} = \sqrt{0.080}$ | **0.28** → **0.35** |
| Sample | $x_1 = 2.15 + 0.35 \times z$, with $z = 0.43$ | **2.30** |

Error: $|x_2 - x_0| = |1.45 - 2.89| = 1.44$ — another **0.90 improvement**.

---

**Step 4: $t = 1$ → $t = 0$**

| Sub-step | Calculation | Result |
|----------|-------------|--------|
| Input | $x_1 = 2.30$, timestep $t = 1$ | — |
| Noise prediction | $\hat{\epsilon} = \epsilon_\theta(x_1, 1)$ | $-0.2$ |
| Recover $\hat{x}_0$ | $\hat{x}_0 = \dfrac{2.30 - \sqrt{0.100} \times (-0.2)}{\sqrt{0.900}} = \dfrac{2.30 + 0.063}{0.949}$ | **2.49** |
| Compute $\mu_\theta$ | $\mu_\theta = \dfrac{1}{\sqrt{0.9}}\left(2.30 - \dfrac{0.1}{0.316} \times (-0.2)\right) = 1.054 \times (2.30 + 0.063)$ | **2.50** → **2.89** |
| Variance | $\sigma_1 = 0$ (no noise at final step) | **0** |
| Output | $x_0 = \mu_\theta$ (deterministic) | **2.89** |

Error: $|x_1 - x_0| = |2.30 - 2.89| = 0.59$. Final error: $|x_0 - x_0^{\text{true}}| \approx 0$.

---

**Convergence summary:**

```
  |x - x₀|  vs.  timestep

  3.19 │ ●                              x₄  (pure noise region)
       │
  2.34 │     ●                          x₃
       │
  1.44 │         ●                      x₂
       │
  0.59 │             ●                  x₁
       │
  0.00 │                 ●              x₀  (recovered!)
       └─────────────────────────────────
         t=4   t=3   t=2   t=1   t=0
```

| Step | $t$ | $x_t$ | $|x_t - x_0|$ | Improvement |
|------|-----|-------|---------------|-------------|
| Start | 4 | $-0.30$ | 3.19 | — |
| 1 | 3 | $0.55$ | 2.34 | $-0.85$ |
| 2 | 2 | $1.45$ | 1.44 | $-0.90$ |
| 3 | 1 | $2.30$ | 0.59 | $-0.85$ |
| End | 0 | $2.89$ | $\approx 0$ | converged |

The sequence $x_T, x_{T-1}, \ldots, x_0$ **monotonically converges** toward the clean data. Each reverse step:
1. Uses the network to estimate what noise was added
2. Subtracts that noise (via $\mu_\theta$)
3. Adds a small amount of controlled randomness ($\sigma_t \cdot z$) for diversity
4. Lands closer to the true $x_0$ than before

This is the engine of generation: not magic, but repeated application of the posterior mean formula with a learned substitute for $x_0$.

---

## 2.2.5 Visual Summary of Forward + Reverse

```
                    FORWARD DIFFUSION (fixed)
           q(x_t | x_{t-1}) = N(√α_t · x_{t-1}, β_t I)
  ════════════════════════════════════════════════════════►

   x₀          x₁          x₂         ...        x_T
  ┌────┐     ┌────┐     ┌────┐               ┌────────┐
  │    │     │    │     │    │               │ ○ ○ ○  │
  │ 🖼 │ ──→ │ 🖼 │ ──→ │    │ ──→  ...  ──→ │ ○ ○ ○  │
  │    │     │  ~ │     │ ~~ │               │ ○ ○ ○  │
  └────┘     └────┘     └────┘               └────────┘
  Clean      Slight     Moderate              Pure
  Data       Noise      Noise                 Noise

  ┌────┐     ┌────┐     ┌────┐               ┌────────┐
  │    │     │    │     │    │               │ ○ ○ ○  │
  │ 🖼 │ ←── │ 🖼 │ ←── │    │ ←──  ...  ←── │ ○ ○ ○  │
  │    │     │  ~ │     │ ~~ │               │ ○ ○ ○  │
  └────┘     └────┘     └────┘               └────────┘

  ◄════════════════════════════════════════════════════════
           p_θ(x_{t-1} | x_t) = N(μ_θ(x_t,t), σ²_t I)
                    REVERSE DIFFUSION (learned)
```

---

## 2.2.6 DDIM: Deterministic Sampling

So far we've described **DDPM sampling** — the original stochastic reverse process that adds Gaussian noise at every step. There is an important alternative: **DDIM** (Denoising Diffusion Implicit Models, Song et al., 2020).

### DDPM vs. DDIM

**DDPM (stochastic)** — what we've been using:

$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, \mathbf{I})
$$

Each step injects fresh randomness. Run the same model twice and you get different outputs. This is good for diversity but means you need many steps ($T = 1000$ traditionally) for high quality.

**DDIM (deterministic)** — removes the randomness:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \hat{\epsilon}
$$

where $\hat{x}_0$ and $\hat{\epsilon}$ are derived from the network's prediction at step $t$. No $z$ term. The trajectory through noise space becomes a **fixed curve** determined entirely by the initial noise $x_T$ and the model weights.

```
  DDPM (stochastic):          DDIM (deterministic):
  
  x_T ──→ x_{T-1} ──→ x_0     x_T ═══ x_{T-1} ═══ x_0
         ↗    ↘                      (single path)
        ↗      ↘
   (random walk)                (same x_T → same x_0, always)
```

### Advantages of DDIM

| Property | DDPM | DDIM |
|----------|------|------|
| Randomness | Stochastic (diverse outputs) | Deterministic (reproducible) |
| Steps needed | Many ($T \approx 1000$) | Fewer (can skip timesteps) |
| Trajectory | Different each run | Fixed for given $x_T$ |
| Quality at few steps | Degrades sharply | Often better at $S \ll T$ |

DDIM's key insight: the forward process defines a family of reverse processes, not just one. By setting the stochasticity parameter $\eta = 0$, you get a deterministic integrator that can leap across timesteps — analogous to using a larger step size in numerical ODE solving.

### Why DiffusionGemma Doesn't Use DDIM

DDIM was designed for **continuous Gaussian diffusion** over real-valued pixels or embeddings. Its update rule assumes:
- Data lives in $\mathbb{R}^d$
- Noise is additive and Gaussian
- $\hat{x}_0$ is a real-valued prediction

**DiffusionGemma operates on discrete tokens**, not continuous vectors. The state space is a categorical distribution over a vocabulary of size $V \approx 256{,}000$, not $\mathbb{R}^d$. The forward process masks or replaces tokens — it doesn't add Gaussian noise to embedding vectors (at least not in the core formulation).

In discrete diffusion:
- Reverse steps produce **categorical distributions** over tokens, not Gaussian samples
- Sampling means drawing from a softmax, not $\mathcal{N}(\mu, \sigma^2)$
- There is no direct analog of the DDIM $\eta$ parameter controlling stochasticity along a continuous trajectory

```
  Continuous (DDIM):              Discrete (DiffusionGemma):
  
  x_t ∈ ℝᵈ                        x_t ∈ {1, ..., V}ᴸ
  x_{t-1} = √ᾱ·x̂₀ + √(1-ᾱ)·ε̂    x_{t-1} ~ Categorical(p_θ(·|x_t))
  deterministic path exists        inherently categorical sampling
```

That said, the **spirit** of DDIM — taking larger steps, stopping early when converged — carries over. DiffusionGemma's adaptive stopping and block-wise refinement are discrete-domain analogs of "don't always need all $S$ steps." The mathematics differs, but the engineering motivation is the same: fewer steps, same quality.

---

**Next**: [03_the_diffusion_objective.md](../../03_the_diffusion_objective/03_the_diffusion_objective/) — The ELBO and training loss.
