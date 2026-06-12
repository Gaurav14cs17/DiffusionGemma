# Chapter 5.4: The Entropy-Bounded Sampler — Accept, Reject, Re-noise

> *"Only keep the tokens you're sure about."*

![Entropy-Bounded Acceptance](../../diagrams/14_entropy_acceptance.png)

---

## 5.4.1 Overview

The **sampler** in DiffusionGemma controls three things:

```
┌──────────────────────────────────────────────────────────────┐
│               ENTROPY-BOUNDED SAMPLER                         │
│                                                               │
│  1. CANVAS INITIALIZATION                                    │
│     How is the initial noisy canvas created?                  │
│                                                               │
│  2. TOKEN ACCEPTANCE                                         │
│     Which predicted tokens are kept?                          │
│                                                               │
│  3. TOKEN RE-NOISING                                         │
│     What happens to rejected tokens?                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 5.4.2 Canvas Initialization

The canvas starts as **pure random tokens**, drawn uniformly from the vocabulary:

$$
x_0^i \sim \text{Uniform}\{1, 2, \ldots, K\}, \qquad i = 1, \ldots, L
$$

This is the discrete analog of starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$ in image diffusion.

```
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─────┬───┐
  │ @ │ ∆ │the│ ! │ q │ 7 │cat│ ¥ │ m │ & │ ... │ z │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─────┴───┘
     1   2   3   4   5   6   7   8   9  10  ...  256
  
  Each token drawn independently and uniformly from
  vocabulary of K = 256,000 tokens.
  P(any specific token at any position) = 1/256,000
```

---

## 5.4.3 Token Acceptance: The Entropy Bound

After the model predicts a probability distribution at each position, the sampler decides which predictions to **accept** (keep) and which to **reject** (re-noise).

### Step 1: Compute Entropy at Each Position

For each position $i$, compute the Shannon entropy:

$$
\mathcal{H}_i = -\sum_{k=1}^{K} p_i[k] \log_2 p_i[k]
$$

**Low entropy** = confident prediction (distribution is peaked)  
**High entropy** = uncertain prediction (distribution is flat)

```
  Position 1: p = [0.95, 0.02, 0.01, ...]    → H = 0.35 bits  (confident)
  Position 2: p = [0.60, 0.20, 0.10, ...]    → H = 2.10 bits  (moderate)
  Position 3: p = [0.05, 0.04, 0.04, ...]    → H = 7.80 bits  (uncertain)
  Position 4: p = [0.90, 0.05, 0.03, ...]    → H = 0.70 bits  (confident)
  Position 5: p = [0.10, 0.08, 0.07, ...]    → H = 6.50 bits  (uncertain)
```

### Step 2: Sort by Confidence (Lowest Entropy First)

$$
\text{sorted indices: } \pi = \text{argsort}(\mathcal{H}_1, \mathcal{H}_2, \ldots, \mathcal{H}_L)
$$

```
  Sorted by entropy (ascending):
  
  Rank  Position   Entropy    Predicted Token
  ────  ────────   ───────    ───────────────
   1    pos 1      0.35       "The"      ← most confident
   2    pos 4      0.70       "on"
   3    pos 2      2.10       "cat"
   4    pos 5      6.50       "and"      ← less confident
   5    pos 3      7.80       "xyz"      ← least confident
```

### Step 3: Cumulative Entropy Check

Starting from the most confident prediction, accumulate entropies and check against a budget:

$$
\boxed{\text{Accept position } \pi_j \text{ if: } \sum_{m=1}^{j} \left(\mathcal{H}_{\pi_m} - \mathcal{H}_{\max}\right) \leq B}
$$

where:
- $\mathcal{H}_{\max} = \log_2 K$ is the maximum possible entropy (uniform distribution over vocabulary)
- $B$ is the **entropy budget** (a hyperparameter)
- The sum $\sum_{m=1}^{j}(\mathcal{H}_{\pi_m} - \mathcal{H}_{\max})$ is always **negative** (since $\mathcal{H}_{\pi_m} < \mathcal{H}_{\max}$)

### Intuitive Explanation

Since $\mathcal{H}_{\pi_m} - \mathcal{H}_{\max} < 0$, the sum becomes **more negative** with each confident token added. The criterion says: "keep accepting tokens as long as the cumulative information gain is large enough."

A very confident token (low entropy) contributes a large negative term ($\mathcal{H}_i - \mathcal{H}_{\max} \ll 0$), allowing more tokens to be accepted. An uncertain token contributes a small negative term ($\mathcal{H}_i \approx \mathcal{H}_{\max}$), rapidly exhausting the budget.

### Worked Example

Let $K = 256000$, so $\mathcal{H}_{\max} = \log_2(256000) \approx 17.97$ bits. Let budget $B = -50$.

| Rank $j$ | Position | $\mathcal{H}_{\pi_j}$ | $\mathcal{H}_{\pi_j} - \mathcal{H}_{\max}$ | Cumulative Sum | $\leq B$? | Accept? |
|------|----------|------------|------------------|----------------|-----------|---------|
| 1 | pos 1 | 0.35 | -17.62 | -17.62 | Yes | **Accept** |
| 2 | pos 4 | 0.70 | -17.27 | -34.89 | Yes | **Accept** |
| 3 | pos 2 | 2.10 | -15.87 | -50.76 | Yes | **Accept** |
| 4 | pos 5 | 6.50 | -11.47 | -62.23 | Yes | **Accept** |
| 5 | pos 3 | 7.80 | -10.17 | -72.40 | Yes | **Accept** |

With a generous budget, all tokens might be accepted. With a stricter budget (e.g., $B = -30$), only the first two would be accepted.

---

## 5.4.3b Entropy Budget: Complete Derivation

### Shannon Entropy and Maximum Entropy

For a probability distribution $\mathbf{p} = [p_1, \ldots, p_K]$ over a vocabulary of size $K$, the Shannon entropy (in bits) is:

$$
\mathcal{H}(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log_2 p_k
$$

**Maximum entropy** occurs for the uniform distribution $p_k = 1/K$:

$$
\mathcal{H}_{\max} = -\sum_{k=1}^{K} \frac{1}{K} \log_2 \frac{1}{K} = \log_2 K
$$

For DiffusionGemma's vocabulary $K = 256{,}000$:

$$
\mathcal{H}_{\max} = \log_2(256000) \approx 17.97 \text{ bits}
$$

A perfectly peaked distribution ($p_{k^*} = 1$, rest zero) has $\mathcal{H} = 0$ bits. Every real prediction lies between these extremes: $0 \leq \mathcal{H}_i \leq \mathcal{H}_{\max}$.

### Information Gain at Each Position

Define the **information gain** at position $i$ as:

$$
\Delta I_i = \mathcal{H}_{\max} - \mathcal{H}_i
$$

Since $\mathcal{H}_i \leq \mathcal{H}_{\max}$, we always have $\Delta I_i \geq 0$. This measures how much the model has "learned" about position $i$:

- $\mathcal{H}_i \approx 0$ (confident) → $\Delta I_i \approx \mathcal{H}_{\max}$ (large gain — model is sure)
- $\mathcal{H}_i \approx \mathcal{H}_{\max}$ (uncertain) → $\Delta I_i \approx 0$ (no gain — model is guessing)

### Cumulative Information Gain

Sort positions by ascending entropy: $\pi = [\pi_1, \pi_2, \ldots, \pi_L]$. The cumulative information gain after accepting the first $j$ positions is:

$$
\sum_{m=1}^{j} \Delta I_{\pi_m} = \sum_{m=1}^{j} (\mathcal{H}_{\max} - \mathcal{H}_{\pi_m}) = j \cdot \mathcal{H}_{\max} - \sum_{m=1}^{j} \mathcal{H}_{\pi_m}
$$

### Budget Criterion (Equivalent Forms)

The acceptance rule from §5.4.3:

$$
\sum_{m=1}^{j} (\mathcal{H}_{\pi_m} - \mathcal{H}_{\max}) \leq B
$$

Since $\mathcal{H}_{\pi_m} - \mathcal{H}_{\max} = -\Delta I_{\pi_m}$, this is equivalent to:

$$
-\sum_{m=1}^{j} \Delta I_{\pi_m} \leq B \quad \Longleftrightarrow \quad \sum_{m=1}^{j} \Delta I_{\pi_m} \geq -B
$$

Rewriting the original inequality:

$$
\sum_{m=1}^{j} \mathcal{H}_{\pi_m} - j \cdot \mathcal{H}_{\max} \leq B \quad \Longleftrightarrow \quad \boxed{\sum_{m=1}^{j} \mathcal{H}_{\pi_m} \leq j \cdot \mathcal{H}_{\max} + B}
$$

**Interpretation**: accept the $j$ most confident tokens as long as their total entropy does not exceed $j$ times the maximum entropy plus a budget allowance $B$. With negative $B$, the budget is **tight** — you must be very confident to accept many tokens.

In practice, positions are processed in order of ascending entropy, and each is accepted while the cumulative sum $\sum_m (\mathcal{H}_{\pi_m} - \mathcal{H}_{\max})$ remains at or above the budget floor $B$ (i.e., has not become more negative than $B$).

### Full Worked Example: 6 Positions

**Entropies** at each position:

```
  Position:   1      2      3      4      5      6
  H_i (bits): 0.35   2.10   7.80   0.70   6.50   12.0
```

**Sorted** by ascending entropy (most confident first):

```
  Rank j:     1      2      3      4      5      6
  Position:   1      4      2      5      3      6
  H_πⱼ:      0.35   0.70   2.10   6.50   7.80   12.0
```

With $\mathcal{H}_{\max} = 17.97$ and budget $B = -50$:

| Rank $j$ | Pos $\pi_j$ | $\mathcal{H}_{\pi_j}$ | $\mathcal{H}_{\pi_j} - \mathcal{H}_{\max}$ | Cumulative $\sum$ | $\geq B$? | Accept? |
|-----------|--------------|------------------------|----------------------------------------------|---------------------|------------|---------|
| 1 | 1 | 0.35 | $-17.62$ | $-17.62$ | $-17.62 \geq -50$ ✓ | **Yes** |
| 2 | 4 | 0.70 | $-17.27$ | $-34.89$ | $-34.89 \geq -50$ ✓ | **Yes** |
| 3 | 2 | 2.10 | $-15.87$ | $-50.76$ | $-50.76 \geq -50$ ✗ | **No** |
| 4 | 5 | 6.50 | $-11.47$ | — | — | **No** |
| 5 | 3 | 7.80 | $-10.17$ | — | — | **No** |
| 6 | 6 | 12.0 | $-5.97$ | — | — | **No** |

**Step-by-step cumulative trace:**

```
  After rank 1 (pos 1):  cum = 0.35 - 17.97 = -17.62
  After rank 2 (pos 4):  cum = -17.62 + (0.70 - 17.97) = -17.62 + (-17.27) = -34.89
  After rank 3 (pos 2):  cum = -34.89 + (2.10 - 17.97) = -34.89 + (-15.87) = -50.76
                         -50.76 < -50  →  BUDGET EXCEEDED  →  reject pos 2 and all remaining
```

**Result**: positions **1** and **4** accepted; positions **2, 3, 5, 6** rejected.

**Equivalent check** via the rewritten criterion $\sum_{m=1}^{j} \mathcal{H}_{\pi_m} \leq j \cdot \mathcal{H}_{\max} + B$:

| $j$ | $\sum \mathcal{H}_{\pi_m}$ | $j \cdot \mathcal{H}_{\max} + B$ | Satisfied? |
|------|---------------------------|-----------------------------------|------------|
| 1 | 0.35 | $17.97 - 50 = -32.03$ | $0.35 \leq -32.03$? No |
| 2 | 1.05 | $35.94 - 50 = -14.06$ | $1.05 \leq -14.06$? No |
| 3 | 3.15 | $53.91 - 50 = 3.91$ | $3.15 \leq 3.91$? **Yes** |

The rewritten form confirms that 3 tokens *could* fit within the entropy allowance, but the operational rule processes tokens in confidence order and stops at the first violation of the cumulative deficit floor. Position 2 (rank 3, $H = 2.10$) is the first token whose addition pushes $\sum (\mathcal{H}_{\pi_m} - \mathcal{H}_{\max})$ past $B = -50$, so it and all lower-confidence positions are rejected.

### Re-noising: Mathematical Description

For each rejected position $i$, the canvas token is replaced by a fresh draw from the uniform distribution over the vocabulary:

$$
x_{s+1}^i \sim \text{Uniform}\{1, 2, \ldots, K\}, \qquad P(x_{s+1}^i = k) = \frac{1}{K} \quad \forall\, k
$$

For the example above:

```
  BEFORE:
  Pos:  [1]    [2]    [3]    [4]    [5]    [6]
  Token: The   cat    xyz    on     and    ???
  H:     0.35  2.10   7.80   0.70   6.50   12.0
  Decision: ✓    ✗      ✗      ✓      ✗      ✗

  AFTER re-noising:
  Pos:  [1]    [2]    [3]    [4]    [5]    [6]
  Token: The   @@@    §§§    on     ∆∆∆    ◊◊◊
         kept  random random kept  random random
```

Each `@@@`, `§§§`, etc. is an independent draw: $P(\text{any specific token}) = 1/256{,}000$. This restores the **training distribution** — the model was trained with uniformly random tokens at noisy positions and expects to see them at inference time.

---

## 5.4.4 Token Re-noising

**All rejected tokens** are replaced with **new uniformly random tokens**:

$$
x_{t+1}^i = \begin{cases}
\hat{x}_0^i & \text{if position } i \text{ is accepted} \\[4pt]
\text{Uniform}\{1, \ldots, K\} & \text{if position } i \text{ is rejected}
\end{cases}
$$

```
  BEFORE acceptance/rejection:
  
  Predicted:  [The]  [cat]  [xyz]  [on]   [and]
  Entropy:     0.35   2.10   7.80   0.70   6.50
  Decision:    KEEP   KEEP   REJECT KEEP   REJECT
  
  AFTER re-noising:
  
  Canvas:     [The]  [cat]  [§§§]  [on]   [∆∆∆]
                ✓      ✓      ↑      ✓      ↑
                          new random    new random
                            token        token
```

### Why Re-noise (Not Keep Old Tokens)?

As explained in Chapter 3.3.7, the model was trained with **uniformly random** tokens at noisy positions. Keeping old tokens would violate this distribution assumption, causing the model to make predictions based on a canvas distribution it hasn't been trained on.

---

## 5.4.5 Accepted Tokens Can Still Change!

A critical point: **acceptance is not permanent**. In the next denoising step:

1. The model re-predicts probabilities for **all** positions (including accepted ones)
2. A previously "accepted" token might now have low confidence (high entropy) because the context has changed
3. It could then be **rejected and re-noised** in a later step

```
  Step 3: [The] [dog] [sat] [on] [rand]     "dog" accepted with H=1.5
  Step 4: [The] [dog] [sat] [on] [the]      Context improves...
  Step 5: [The] [cat] [sat] [on] [the]      "dog" → "cat"! Self-correction!
                  ↑
           "dog" had higher entropy now (H=3.2) because 
           "the mat" context makes "cat" more likely
```

This is the fundamental advantage of uniform state diffusion over masked diffusion — **every token remains revisable**.

---

## 5.4.6 Full Sampler Pipeline (One Step)

```
  ┌───────────────────────────────────────────────────────────────┐
  │              ONE DENOISING STEP (FULL PIPELINE)                │
  │                                                                │
  │  Input: canvas x_t = [tok₁, tok₂, ..., tok_L]                │
  │         + encoder KV cache                                     │
  │         + self-conditioning from previous step                 │
  │                                                                │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 1. FORWARD PASS                                      │      │
  │  │    Run Gemma 4 (bidirectional) → logits for all pos  │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 2. SCHEDULER: Temperature                            │      │
  │  │    logits ← logits / τ_s                             │      │
  │  │    p_i = softmax(logits_i / τ_s)                     │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 3. SAMPLE candidate tokens                           │      │
  │  │    x̂₀ⁱ ~ Cat(p_i) for each position i               │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 4. SAMPLER: Compute entropy per position             │      │
  │  │    H_i = -Σ p_i[k] log p_i[k]                       │      │
  │  │    Sort by H (ascending)                             │      │
  │  │    Accept tokens within entropy budget                │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 5. RE-NOISE rejected positions                       │      │
  │  │    Rejected positions ← Uniform(1,...,K)             │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 6. SELF-CONDITIONING: Compute soft embeddings        │      │
  │  │    ẽ_i = p_i^T · E → FFNN → c_i                     │      │
  │  │    Save for next step                                │      │
  │  └──────────────────────┬────────────────────────────────┘     │
  │                         ↓                                      │
  │  ┌─────────────────────────────────────────────────────┐      │
  │  │ 7. SCHEDULER: Adaptive stopping check                │      │
  │  │    IF stable AND confident → STOP                    │      │
  │  │    ELSE → continue to step s+1                       │      │
  │  └─────────────────────────────────────────────────────┘      │
  │                                                                │
  │  Output: updated canvas x_{t+1}                                │
  └───────────────────────────────────────────────────────────────┘
```

---

**Next**: [../06_Training/01_training_objective.md](../../06_Training/01_training_objective/) — How DiffusionGemma is trained.
