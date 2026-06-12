# Chapter 4.2: The Encoder-Denoiser Patch — One Model, Two Roles (Step by Step)

> *"Same brain, different glasses."*

![Encoder-Denoiser Patch](../../diagrams/09_encoder_denoiser.png)

![Attention Masks](../../diagrams/08_attention_masks.png)

---

## 4.2.1 The Key Question

We have a pre-trained Gemma 4 that's excellent at next-token prediction. But diffusion needs:
1. An **Encoder** — to understand the query
2. A **Denoiser** — to clean up a noisy canvas

How do we get both from a single model?

---

## 4.2.2 The Answer: Change the Attention Mask

A transformer's behavior is fundamentally controlled by its **attention mask** — which tokens can "see" which other tokens. By switching this mask, the same model can behave in two completely different ways.

Let's trace through a **tiny example** to see exactly what changes.

---

## 4.2.3 Concrete Example: Encoder Mode

**Query**: "Tell me a joke" (4 tokens)

### Step 1: Token Embedding

```
  "Tell"  "me"   "a"   "joke"
    │       │      │      │
    ▼       ▼      ▼      ▼
  ┌─────┐┌─────┐┌─────┐┌─────┐
  │ e₁  ││ e₂  ││ e₃  ││ e₄  │    e_i ∈ ℝ^d  (d=4096 in real model)
  │[0.3]││[-0.1││[0.5]││[0.8]│    (showing just 1 dimension for clarity)
  └─────┘└─────┘└─────┘└─────┘
```

### Step 2: Attention with CAUSAL Mask

Each token can only attend to itself and tokens **before** it:

```
  CAUSAL ATTENTION MASK:
  
  "Tell" can see:  [Tell]              → 1 token
  "me"   can see:  [Tell, me]          → 2 tokens
  "a"    can see:  [Tell, me, a]       → 3 tokens
  "joke" can see:  [Tell, me, a, joke] → 4 tokens (ALL)
  
       Tell  me    a   joke
  Tell [ ✓    ·    ·    · ]     ✓ = can attend
  me   [ ✓    ✓    ·    · ]     · = blocked (masked out)
  a    [ ✓    ✓    ✓    · ]
  joke [ ✓    ✓    ✓    ✓ ]
```

### Step 3: What Comes Out at Each Position

```
  After processing through ALL transformer layers:
  
  Position 1 ("Tell"):
  h₁ = representation of just "Tell" alone
       (limited context — only knows one word)
  
  Position 2 ("me"):
  h₂ = representation of "Tell me"
       (knows 2 words of context)
  
  Position 3 ("a"):
  h₃ = representation of "Tell me a"
       (knows 3 words of context)
  
  Position 4 ("joke"):
  h₄ = representation of "Tell me a joke"
       (knows the FULL query — richest representation)
```

### Step 4: Store the KV Cache (THE KEY OUTPUT)

At every attention layer, the model computes **Key** and **Value** vectors for each token. These are stored:

```
  LAYER 1:
  ┌──────────────────────────────────────────────────────┐
  │  Token "Tell":  K₁,₁ = W_K · h₁  ,  V₁,₁ = W_V · h₁  │
  │  Token "me":    K₁,₂ = W_K · h₂  ,  V₁,₂ = W_V · h₂  │
  │  Token "a":     K₁,₃ = W_K · h₃  ,  V₁,₃ = W_V · h₃  │
  │  Token "joke":  K₁,₄ = W_K · h₄  ,  V₁,₄ = W_V · h₄  │
  └──────────────────────────────────────────────────────┘
  
  LAYER 2:
  ┌──────────────────────────────────────────────────────┐
  │  Token "Tell":  K₂,₁ , V₂,₁                             │
  │  Token "me":    K₂,₂ , V₂,₂                             │
  │  Token "a":     K₂,₃ , V₂,₃                             │
  │  Token "joke":  K₂,₄ , V₂,₄                             │
  └──────────────────────────────────────────────────────┘
  
  ... (for all layers)
  
  This is the KV CACHE — it captures the encoder's understanding.
  It will be SHARED with the denoiser.
```

### What We Do NOT Use

In autoregressive mode, only the **last position's hidden state** ($h_4$) would be used to predict the next token. In encoder mode, we don't predict any token at all — we just save the KV cache.

```
  AUTOREGRESSIVE:
  h₁ → (thrown away)
  h₂ → (thrown away)
  h₃ → (thrown away)
  h₄ → LM Head → predict next token "Why"
  
  ENCODER MODE (DiffusionGemma):
  h₁ → KV cache stored ✓
  h₂ → KV cache stored ✓
  h₃ → KV cache stored ✓
  h₄ → KV cache stored ✓
  No LM Head used. We just want the KV cache.
```

---

## 4.2.4 Concrete Example: Denoiser Mode

Now the denoiser takes a **noisy canvas** and the **encoder's KV cache**, and tries to predict clean tokens.

### Setup

```
  Encoder KV cache: captures "Tell me a joke" (4 positions)
  
  Noisy canvas (6 tokens):  ["Why", "rand₁", "rand₂", "chicken", "rand₃", "road"]
  (The correct answer might be: "Why did the chicken cross the road")
```

### Step 1: Canvas Token Embedding

```
  "Why"     "rand₁"  "rand₂"  "chicken"  "rand₃"  "road"
    │          │        │         │          │        │
    ▼          ▼        ▼         ▼          ▼        ▼
  ┌─────┐ ┌─────┐ ┌─────┐  ┌─────┐   ┌─────┐ ┌─────┐
  │ e₁  │ │ e₂  │ │ e₃  │  │ e₄  │   │ e₅  │ │ e₆  │
  └─────┘ └─────┘ └─────┘  └─────┘   └─────┘ └─────┘
```

### Step 2: Attention with BIDIRECTIONAL Mask + Encoder KV Cache

This is where the magic happens. Each canvas token attends to:
- **ALL encoder positions** (the query "Tell me a joke")
- **ALL other canvas positions** (the entire noisy canvas)

```
  COMBINED ATTENTION:
  
  Canvas token "Why" (position 1) attends to:
  
  FROM ENCODER KV CACHE (causal within encoder):
  ┌──────────────────────────────────────────────────────┐
  │  "Tell" → attend ✓  (sees what "Tell" encoded)      │
  │  "me"   → attend ✓  (sees what "Tell me" encoded)   │
  │  "a"    → attend ✓  (sees what "Tell me a" encoded) │
  │  "joke" → attend ✓  (sees FULL query understanding) │
  └──────────────────────────────────────────────────────┘
  
  FROM CANVAS (bidirectional):
  ┌──────────────────────────────────────────────────────┐
  │  "Why"     → attend ✓  (itself)                      │
  │  "rand₁"  → attend ✓  (can see FUTURE tokens!)     │
  │  "rand₂"  → attend ✓  (can see FUTURE tokens!)     │
  │  "chicken" → attend ✓ (can see FUTURE tokens!)     │
  │  "rand₃"  → attend ✓                                │
  │  "road"   → attend ✓                                │
  └──────────────────────────────────────────────────────┘
  
  Total: "Why" attends to 4 (encoder) + 6 (canvas) = 10 positions
```

### The Full Attention Mask (Visualized)

```
                        KEYS
              Encoder tokens        Canvas tokens
              Tell  me  a  joke    Why r₁ r₂ chk r₃ road
           ┌─────────────────────┬────────────────────────┐
  Q  Tell  │  ✓   ·   ·   ·    │   ·   ·   ·   ·  ·  · │ ← Encoder
  U  me    │  ✓   ✓   ·   ·    │   ·   ·   ·   ·  ·  · │   tokens
  E  a     │  ✓   ✓   ✓   ·    │   ·   ·   ·   ·  ·  · │   don't
  R  joke  │  ✓   ✓   ✓   ✓    │   ·   ·   ·   ·  ·  · │   query
  I  ──────┼─────────────────────┼────────────────────────┤
  E  Why   │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │ ← Canvas
  S  r₁    │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │   tokens
     r₂    │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │   attend
     chk   │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │   to
     r₃    │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │   EVERYTHING
     road  │  ✓   ✓   ✓   ✓    │   ✓   ✓   ✓   ✓  ✓  ✓ │
           └─────────────────────┴────────────────────────┘
  
  TOP-LEFT:     Encoder self-attention (CAUSAL) — already computed
  TOP-RIGHT:    Zeros — encoder doesn't see canvas (it ran first)
  BOTTOM-LEFT:  Canvas sees encoder (ALL encoder positions)
  BOTTOM-RIGHT: Canvas self-attention (BIDIRECTIONAL)
  
  BUT: Only the canvas tokens generate Queries.
  The encoder tokens don't run again — we just reuse their K,V.
```

### Step 3: The Model Processes Each Canvas Position

```
  For canvas position 1 ("Why"):
  
  Attention scores = softmax( Q₁ · [K_enc₁, K_enc₂, K_enc₃, K_enc₄,
                                      K_can₁, K_can₂, K_can₃, K_can₄, K_can₅, K_can₆]ᵀ / √d )
  
  The model "reads" the query ("Tell me a joke") through the encoder KVs
  AND "reads" the entire canvas simultaneously.
  
  It then produces a hidden state that knows:
  - The user wants a joke ← from encoder
  - Position 1 currently has "Why" ← from canvas
  - The canvas has "chicken" at pos 4 and "road" at pos 6 ← from canvas
  
  This rich context lets it predict: "Why" is probably correct!
```

### Step 4: LM Head at EVERY Position

```
  After all transformer layers, EVERY canvas position gets logits:
  
  Position 1 ("Why"):
  logits₁ → softmax → p₁ = [P("Why")=0.85, P("How")=0.05, ...]
  → Best prediction: "Why" (confident ✓)
  
  Position 2 ("rand₁"):
  logits₂ → softmax → p₂ = [P("did")=0.55, P("does")=0.15, P("would")=0.10, ...]
  → Best prediction: "did" (moderate confidence)
  
  Position 3 ("rand₂"):
  logits₃ → softmax → p₃ = [P("the")=0.45, P("a")=0.20, ...]
  → Best prediction: "the" (moderate confidence)
  
  Position 4 ("chicken"):
  logits₄ → softmax → p₄ = [P("chicken")=0.80, P("duck")=0.05, ...]
  → Best prediction: "chicken" (confident ✓)
  
  Position 5 ("rand₃"):
  logits₅ → softmax → p₅ = [P("cross")=0.30, P("eat")=0.10, P("see")=0.08, ...]
  → Best prediction: "cross" (low confidence)
  
  Position 6 ("road"):
  logits₆ → softmax → p₆ = [P("road")=0.70, P("street")=0.10, ...]
  → Best prediction: "road" (confident ✓)
```

### Step 5: Accept/Reject/Re-noise

```
  BEFORE:  ["Why"]  ["rand₁"]  ["rand₂"]  ["chicken"]  ["rand₃"]  ["road"]
  
  Model predicts:
  Pos 1: "Why"     → H=0.5   → ACCEPT ✓
  Pos 2: "did"     → H=1.8   → ACCEPT ✓
  Pos 3: "the"     → H=2.5   → ACCEPT ✓
  Pos 4: "chicken" → H=0.8   → ACCEPT ✓
  Pos 5: "cross"   → H=3.2   → REJECT ✗ (not confident enough)
  Pos 6: "road"    → H=1.0   → ACCEPT ✓
  
  AFTER:   ["Why"]  ["did"]  ["the"]  ["chicken"]  [NEW RAND]  ["road"]
                                                      ↑
                                                 Re-noised!
```

### Step 6: Next Denoising Step

```
  The updated canvas goes back in:
  
  Canvas: ["Why", "did", "the", "chicken", "rand₃'", "road"]
  
  NOW position 5 has much better context:
  - "Why did the chicken ??? road"
  - The model can now confidently predict "cross" → H=0.8 → ACCEPT ✓
  
  Final canvas: ["Why", "did", "the", "chicken", "cross", "road"]
  
  → "Why did the chicken cross the road" ← But wait, "the" is missing!
  (In reality, with 256 tokens, this would work out fine with more positions)
```

---

## 4.2.4b Mathematical Formulation of the Two Modes

This section formalizes everything above in precise linear-algebra notation, then walks through a **complete numerical trace** with $d = d_k = d_v = 4$.

### Encoder Mode: Causal Self-Attention

Given query tokens $q_1, \ldots, q_n$ (e.g., $n = 4$ for "Tell me a joke"), the encoder runs a standard transformer forward pass with a **causal** attention mask:

$$
\mathbf{H}^{(\ell)} = \text{Transformer}^{(\ell)}\!\left(\mathbf{H}^{(\ell-1)},\; \text{mask}=\text{causal}\right), \quad \mathbf{H}^{(0)} = \text{Embed}(q_1, \ldots, q_n)
$$

At each layer $\ell \in \{1, \ldots, L_{\text{layers}}\}$, the model projects hidden states into keys and values and **stores them in the KV cache**:

$$
\mathbf{K}_{\text{enc}}^{(\ell)} = \mathbf{W}_K^{(\ell)} \cdot \mathbf{H}^{(\ell)} \in \mathbb{R}^{d_k \times n}, \qquad
\mathbf{V}_{\text{enc}}^{(\ell)} = \mathbf{W}_V^{(\ell)} \cdot \mathbf{H}^{(\ell)} \in \mathbb{R}^{d_v \times n}
$$

Here each **column** $j$ of $\mathbf{K}_{\text{enc}}^{(\ell)}$ is the key vector for token $q_j$, and column $j$ of $\mathbf{V}_{\text{enc}}^{(\ell)}$ is the corresponding value vector. The causal mask ensures token $j$'s hidden state only aggregates information from positions $1, \ldots, j$.

**Encoder output**: the KV cache $\{\mathbf{K}_{\text{enc}}^{(\ell)}, \mathbf{V}_{\text{enc}}^{(\ell)}\}_{\ell=1}^{L_{\text{layers}}}$. No LM head is applied.

### Denoiser Mode: Concatenated Cross-Context Attention

Given a noisy canvas $x_t$ of length $L$ (e.g., $L = 6$), at each layer $\ell$ the denoiser:

1. Computes canvas queries, keys, and values from canvas hidden states $\mathbf{H}_{\text{canvas}}^{(\ell)}$:

$$
\mathbf{Q}^{(\ell)} = \mathbf{W}_Q^{(\ell)} \cdot \mathbf{H}_{\text{canvas}}^{(\ell)} \in \mathbb{R}^{d_k \times L}
$$

$$
\mathbf{K}_{\text{canvas}}^{(\ell)} = \mathbf{W}_K^{(\ell)} \cdot \mathbf{H}_{\text{canvas}}^{(\ell)} \in \mathbb{R}^{d_k \times L}, \qquad
\mathbf{V}_{\text{canvas}}^{(\ell)} = \mathbf{W}_V^{(\ell)} \cdot \mathbf{H}_{\text{canvas}}^{(\ell)} \in \mathbb{R}^{d_v \times L}
$$

2. **Concatenates** the frozen encoder cache with the fresh canvas cache (horizontal concatenation of columns):

$$
\mathbf{K}_{\text{full}}^{(\ell)} = \left[\mathbf{K}_{\text{enc}}^{(\ell)} \;\middle|\; \mathbf{K}_{\text{canvas}}^{(\ell)}\right] \in \mathbb{R}^{d_k \times (n + L)}
$$

$$
\mathbf{V}_{\text{full}}^{(\ell)} = \left[\mathbf{V}_{\text{enc}}^{(\ell)} \;\middle|\; \mathbf{V}_{\text{canvas}}^{(\ell)}\right] \in \mathbb{R}^{d_v \times (n + L)}
$$

3. Computes scaled dot-product attention:

$$
\mathbf{A}^{(\ell)} = \text{softmax}\!\left(\frac{\mathbf{Q}^{(\ell)\top} \mathbf{K}_{\text{full}}^{(\ell)}}{\sqrt{d_k}} + \mathbf{M}_{\text{log}}\right)^\top \mathbf{V}_{\text{full}}^{(\ell)} \in \mathbb{R}^{d_v \times L}
$$

Equivalently, for each canvas position $i \in \{1, \ldots, L\}$, the output row is:

$$
\mathbf{a}_i = \sum_{j=1}^{n+L} \alpha_{ij}\, \mathbf{v}_j, \quad \text{where } \alpha_{ij} = \frac{\exp\!\left(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k} + M_{\text{log}}[i,j]\right)}{\sum_{j'=1}^{n+L} \exp\!\left(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_k} + M_{\text{log}}[i,j']\right)}
$$

### The Attention Mask

The mask $\mathbf{M} \in \{0, 1\}^{L \times (n + L)}$ governs which key positions each canvas query may attend to:

$$
M[i, j] = \begin{cases}
1 & \text{if canvas query } i \text{ may attend to key } j \\
0 & \text{otherwise (masked out → score set to } -\infty \text{ before softmax)}
\end{cases}
$$

In denoiser mode, **every canvas token sees everything**:

$$
M[i, j] = 1 \quad \forall\, i \in \{1,\ldots,L\},\; j \in \{1,\ldots,n+L\}
$$

The encoder-to-encoder block (top-left $n \times n$) was already computed causally during encoding and is **not re-queried** — only its stored $\mathbf{K}, \mathbf{V}$ columns are reused. The encoder never sees canvas keys (top-right block is zero during encoding; canvas never queries encoder positions as queries).

### LM Head: Logits at Every Canvas Position

After all $L_{\text{layers}}$ denoiser layers, each canvas position $i$ has a final hidden state $\mathbf{h}_i^{(\text{final})} \in \mathbb{R}^d$. The shared LM head maps this to a vocabulary-sized logit vector:

$$
\text{logits}_i = \mathbf{W}_{\text{head}} \cdot \mathbf{h}_i^{(\text{final})} + \mathbf{b}_{\text{head}} \in \mathbb{R}^{K}
$$

where $K$ is vocabulary size (e.g., $K = 256{,}000$ for Gemma). Token probabilities follow:

$$
p_i(k) = \text{softmax}(\text{logits}_i)[k] = \frac{e^{\text{logits}_i[k]}}{\sum_{j=1}^{K} e^{\text{logits}_i[j]}}, \quad k \in \{1, \ldots, K\}
$$

Unlike autoregressive decoding (logits only at the last position), diffusion denoising uses **all** $L$ logit vectors simultaneously to predict clean tokens at every position.

---

### Full Numerical Trace: "Tell me a joke" ($d = 4$)

We trace **one attention layer** with $n = 4$ encoder tokens and $L = 6$ canvas tokens. Weight matrices are shared between modes (shown once):

$$
\mathbf{W}_Q = \begin{bmatrix} 1.0 & 0.2 & 0.0 & 0.1 \\ 0.1 & 1.0 & 0.2 & 0.0 \\ 0.0 & 0.1 & 1.0 & 0.2 \\ 0.2 & 0.0 & 0.1 & 1.0 \end{bmatrix}, \quad
\mathbf{W}_K = \begin{bmatrix} 0.8 & 0.1 & 0.0 & 0.2 \\ 0.0 & 0.9 & 0.1 & 0.0 \\ 0.1 & 0.0 & 0.8 & 0.1 \\ 0.2 & 0.1 & 0.0 & 0.9 \end{bmatrix}
$$

$$
\mathbf{W}_V = \begin{bmatrix} 0.9 & 0.0 & 0.1 & 0.2 \\ 0.1 & 0.8 & 0.0 & 0.1 \\ 0.0 & 0.2 & 0.9 & 0.0 \\ 0.1 & 0.0 & 0.1 & 0.8 \end{bmatrix}
$$

#### Phase A: Encoder Pass (Causal)

**Token embeddings** (after lookup):

| Token | $\mathbf{h}$ |
|-------|------------------|
| Tell  | $[1.0,\; 0.0,\; 0.0,\; 0.0]^\top$ |
| me    | $[0.0,\; 1.0,\; 0.0,\; 0.0]^\top$ |
| a     | $[0.0,\; 0.0,\; 1.0,\; 0.0]^\top$ |
| joke  | $[0.0,\; 0.0,\; 0.0,\; 1.0]^\top$ |

**Compute encoder K and V** (stored in cache; $\mathbf{K} = \mathbf{W}_K \mathbf{h}$):

```
  K_enc(Tell) = W_K · h_Tell = [0.80, 0.00, 0.10, 0.20]ᵀ
  K_enc(me)   = W_K · h_me   = [0.10, 0.90, 0.00, 0.10]ᵀ
  K_enc(a)    = W_K · h_a    = [0.00, 0.10, 0.80, 0.10]ᵀ
  K_enc(joke) = W_K · h_joke = [0.20, 0.10, 0.10, 0.90]ᵀ

  V_enc(Tell) = W_V · h_Tell = [0.90, 0.10, 0.00, 0.10]ᵀ
  V_enc(me)   = W_V · h_me   = [0.00, 0.80, 0.20, 0.00]ᵀ
  V_enc(a)    = W_V · h_a    = [0.00, 0.00, 0.90, 0.10]ᵀ
  V_enc(joke) = W_V · h_joke = [0.00, 0.00, 0.10, 0.80]ᵀ
```

These 4 key-value pairs per layer are **frozen** and passed to the denoiser.

#### Phase B: Denoiser Pass (Bidirectional + Encoder KV)

**Canvas embeddings**:

| Position | Token    | $\mathbf{c}$ |
|----------|----------|------------------|
| 1        | Why      | $[0.8,\; 0.1,\; 0.2,\; 0.0]^\top$ |
| 2        | rand₁    | $[0.2,\; 0.7,\; 0.1,\; 0.3]^\top$ |
| 3        | rand₂    | $[0.5,\; 0.3,\; 0.6,\; 0.1]^\top$ |
| 4        | chicken  | $[0.3,\; 0.4,\; 0.2,\; 0.8]^\top$ |
| 5        | rand₃    | $[0.1,\; 0.6,\; 0.3,\; 0.2]^\top$ |
| 6        | road     | $[0.6,\; 0.2,\; 0.4,\; 0.5]^\top$ |

**Canvas Q, K, V** (position 1 shown in detail):

```
  Q₁ = W_Q · c_Why = [0.82, 0.12, 0.24, 0.18]ᵀ

  K_can₁(Why)     = [0.66, 0.11, 0.18, 0.22]ᵀ
  K_can₂(rand₁)   = [0.22, 0.71, 0.11, 0.35]ᵀ
  K_can₃(rand₂)   = [0.43, 0.39, 0.52, 0.23]ᵀ
  K_can₄(chicken) = [0.28, 0.46, 0.18, 0.82]ᵀ
  K_can₅(rand₃)   = [0.14, 0.64, 0.27, 0.30]ᵀ
  K_can₆(road)    = [0.50, 0.28, 0.42, 0.57]ᵀ

  V_can₁(Why)     = [0.74, 0.09, 0.20, 0.18]ᵀ
  V_can₂(rand₁)   = [0.20, 0.66, 0.11, 0.34]ᵀ
  ... (analogous for positions 3–6)
```

**Concatenate** into full key/value sets ($n + L = 10$ columns):

$$
\mathbf{K}_{\text{full}} = \big[\underbrace{K_{\text{enc}}}_{4}\;\big|\;\underbrace{K_{\text{canvas}}}_{6}\big], \qquad
\mathbf{V}_{\text{full}} = \big[\underbrace{V_{\text{enc}}}_{4}\;\big|\;\underbrace{V_{\text{canvas}}}_{6}\big]
$$

#### Position 1 ("Why"): Attention Step-by-Step

**Step 1 — Raw scores** $s_j = \mathbf{q}_1^\top \mathbf{k}_j / \sqrt{4}$:

```
  j=1  Tell:     q₁·K_enc(Tell)     / 2 = 0.355
  j=2  me:       q₁·K_enc(me)       / 2 = 0.285
  j=3  a:        q₁·K_enc(a)        / 2 = 0.310
  j=4  joke:     q₁·K_enc(joke)    / 2 = 0.410
  j=5  Why:      q₁·K_can₁(Why)     / 2 = 0.500
  j=6  rand₁:    q₁·K_can₂(rand₁)   / 2 = 0.303
  j=7  rand₂:    q₁·K_can₃(rand₂)   / 2 = 0.430
  j=8  chicken:  q₁·K_can₄(chicken) / 2 = 0.490
  j=9  rand₃:    q₁·K_can₅(rand₃)   / 2 = 0.275
  j=10 road:     q₁·K_can₆(road)    / 2 = 0.500

  s = [0.355, 0.285, 0.310, 0.410, 0.500, 0.303, 0.430, 0.490, 0.275, 0.500]
```

**Step 2 — Softmax** (no masking; $M[i,j] = 1$ everywhere):

```
  exp(s) ≈ [1.426, 1.330, 1.364, 1.507, 1.649, 1.354, 1.537, 1.632, 1.317, 1.649]
  Z = Σ exp(s) = 14.765

  α₁ = [0.097, 0.090, 0.092, 0.102, 0.112, 0.092, 0.104, 0.111, 0.089, 0.112]
        ↑enc Tell  me    a    joke   ↑canvas Why  r₁   r₂   chk   r₃   road
```

**Step 3 — Weighted value sum** $\mathbf{a}_1 = \sum_j \alpha_{1j}\, \mathbf{v}_j$:

```
  a₁ ≈ 0.097·V(Tell) + 0.090·V(me) + 0.092·V(a) + 0.102·V(joke)    ← encoder context
       + 0.112·V(Why) + 0.092·V(rand₁) + 0.104·V(rand₂)              ← canvas context
       + 0.111·V(chicken) + 0.089·V(rand₃) + 0.112·V(road)

  a₁ ≈ [0.31, 0.28, 0.35, 0.42]ᵀ
```

The model blends "user wants a joke" (encoder, ~38% total weight) with canvas context including "chicken" (11%) and "road" (11%).

#### Position 2 (rand₁): Attention (Abbreviated)

```
  Q₂ = W_Q · c_rand₁ = [0.37, 0.78, 0.19, 0.41]ᵀ

  Top scores after softmax:
    joke (enc): 0.118    chicken (canvas): 0.125    rand₁ (self): 0.108

  a₂ ≈ [0.22, 0.41, 0.38, 0.39]ᵀ
```

Position 2 attends strongly to encoder "joke" (full-query understanding) and canvas "chicken" (future context visible only in bidirectional mode).

#### LM Head: Logits and Probabilities (2 Positions)

After all layers, apply $\mathbf{W}_{\text{head}} \in \mathbb{R}^{5 \times 4}$ (toy vocab $K = 5$):

$$
\mathbf{W}_{\text{head}} = \begin{bmatrix}
2.0 & 0.5 & 0.1 & 0.0 \\
0.1 & 1.8 & 0.3 & 0.2 \\
0.0 & 0.2 & 2.2 & 0.1 \\
0.3 & 0.1 & 0.4 & 1.5 \\
0.1 & 0.0 & 0.2 & 2.0
\end{bmatrix}, \quad \text{vocab} = \{\text{Why},\; \text{did},\; \text{the},\; \text{chicken},\; \text{road}\}
$$

**Position 1** ($\mathbf{h}_1^{(\text{final})} = \mathbf{a}_1$):

```
  logits₁ = W_head · a₁
          = [2.0·0.31 + 0.5·0.28 + ..., ...]
          = [1.42, 0.38, 0.95, 0.89, 1.05]

  softmax(logits₁):
    P(Why)     = e^1.42 / Z  = 0.42
    P(did)     = e^0.38 / Z  = 0.10
    P(the)     = e^0.95 / Z  = 0.18
    P(chicken) = e^0.89 / Z  = 0.17
    P(road)    = e^1.05 / Z  = 0.13

  → Best prediction: "Why" (42% confidence) ✓
```

**Position 2** ($\mathbf{h}_2^{(\text{final})} = \mathbf{a}_2$):

```
  logits₂ = W_head · a₂
          = [0.55, 1.62, 1.18, 0.72, 0.48]

  softmax(logits₂):
    P(Why)     = 0.08
    P(did)     = 0.38   ← highest
    P(the)     = 0.24
    P(chicken) = 0.15
    P(road)    = 0.11

  → Best prediction: "did" (38% confidence) — moderate, as expected for a noisy position
```

This trace shows the full pipeline: **encoder KV cache → concatenated attention → per-position LM head → independent token predictions** across all canvas positions.

---

## 4.2.5 Summary: The Two Modes Side by Side

```
  ┌─────────────────────────────┬─────────────────────────────────┐
  │      ENCODER MODE            │       DENOISER MODE              │
  ├─────────────────────────────┼─────────────────────────────────┤
  │                              │                                  │
  │  INPUT                       │  INPUT                           │
  │  User query tokens           │  Noisy canvas tokens             │
  │  "Tell me a joke"            │  "Why rand₁ rand₂ chicken..."   │
  │                              │                                  │
  │  ATTENTION                   │  ATTENTION                       │
  │  Causal (lower triangular)   │  Bidirectional (full matrix)    │
  │  Token i sees tokens 1..i    │  Token i sees ALL tokens        │
  │                              │  + encoder KV cache              │
  │                              │                                  │
  │  OUTPUT                      │  OUTPUT                          │
  │  KV cache at all layers      │  Logits at EVERY position       │
  │  (K and V for each token)    │  (probability distribution      │
  │                              │   over vocabulary for each pos)  │
  │  NO LM head used             │  LM head used at ALL positions  │
  │                              │                                  │
  │  RUNS                        │  RUNS                            │
  │  ONCE                        │  S times (denoising steps)       │
  │                              │  Reuses encoder KV cache         │
  │                              │                                  │
  │  ANALOGY                     │  ANALOGY                         │
  │  "Reading and understanding  │  "Writing a draft, reviewing,   │
  │   the assignment"            │   and rewriting it"              │
  │                              │                                  │
  └─────────────────────────────┴─────────────────────────────────┘
```

---

## 4.2.6 Why This Is Brilliant

```
  1. WEIGHT REUSE
     The same 26B parameters serve both roles.
     No extra model needed.
     
  2. PRE-TRAINED KNOWLEDGE
     Gemma 4 already understands language deeply.
     The encoder mode benefits from this immediately.
     The denoiser mode just needs fine-tuning to adapt.
     
  3. SHARED VOCABULARY
     Both modes use the same token embeddings and LM head.
     No mismatched representations.
     
  4. MINIMAL CHANGE
     The ONLY architectural change is the attention mask.
     Everything else — weights, embeddings, MoE routing — stays.
     
  5. KV CACHE AS BRIDGE
     The KV cache is a natural output of any transformer.
     No cross-attention, no adapter layers, no new parameters.
```

---

**Next**: [03_bidirectional_attention.md](../../03_bidirectional_attention/03_bidirectional_attention/) — What happens INSIDE a single transformer layer.
