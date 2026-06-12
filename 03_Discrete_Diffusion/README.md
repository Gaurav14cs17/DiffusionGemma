# Chapter 3 — Discrete Diffusion

You can't add Gaussian noise to a word. Text lives in a **discrete** vocabulary space — "cat" doesn't become "slightly noisy cat." This chapter shows how diffusion is reinvented for discrete tokens using transition matrices, Continuous-Time Markov Chains (CTMCs), and two key strategies: masking tokens and replacing them with random ones.

![Discrete Diffusion — Masked vs Uniform](../diagrams/05_masked_vs_uniform.png)

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 3.1 | [From Continuous to Discrete](01_from_continuous_to_discrete/) | Transition matrices $\mathbf{Q}_t$, CTMCs, matrix exponentials, how $q(x_t \mid x_0)$ works for tokens |
| 3.2 | [Masked Diffusion (MDLM)](02_masked_diffusion/) | Absorbing-state noise — tokens are replaced with `[MASK]`, connection to BERT, worked examples |
| 3.3 | [Uniform State Diffusion (UDLM)](03_uniform_state_diffusion/) | Random-token corruption — tokens are replaced with any vocabulary token, self-correction ability |
| 3.4 | [Comparison Table](04_comparison_table/) | AR vs. Masked vs. Uniform — side-by-side with numerical traces, strengths, weaknesses |

---

**Previous**: [Chapter 2 — Diffusion Fundamentals](../02_Diffusion_Fundamentals/) · **Next**: [Chapter 4 — The Architecture](../04_Architecture/)
