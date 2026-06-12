# Chapter 1 — Why Diffusion for Text?

Traditional autoregressive LLMs generate text one token at a time — each token requires a full forward pass through billions of parameters. For single-user inference, the GPU spends most of its time *loading weights from memory*, not actually computing. This is the **memory-bound bottleneck**.

DiffusionGemma takes a completely different approach: generate **256 tokens at once** on a canvas and iteratively denoise them — turning a sequential 256-step problem into a parallel ~16-step one.

![AR vs Diffusion](../diagrams/01_why_diffusion.png)

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 1.1 | [Autoregressive vs. Diffusion](01_autoregressive_vs_diffusion/) | The roofline model, latency math, arithmetic intensity, self-correction — why diffusion changes the compute equation |

---

**Next**: [Chapter 2 — Diffusion Fundamentals](../02_Diffusion_Fundamentals/)
