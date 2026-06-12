# Chapter 5 — Inference

This is where all the pieces come together. At generation time, DiffusionGemma takes a prompt, initializes a 256-token canvas with noise, and iteratively denoises it — using self-conditioning, temperature scheduling, and entropy-based acceptance to produce coherent text in ~16 steps.

![Denoising trace — canvas evolving from noise to text](../diagrams/15_denoising_trace.png)

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 5.0 | [Connecting the Components](00_connecting_the_components/) | Full anatomy of one denoising step — encoder, denoiser, sampler — with numbers at every stage |
| 5.1 | [Self-Conditioning](01_self_conditioning/) | The model's memory — how previous predictions become soft embeddings for the next step |
| 5.2 | [Multi-Canvas Sampling](02_multi_canvas_sampling/) | Block diffusion for long text — generating beyond 256 tokens by stitching canvases with overlap |
| 5.3 | [The Scheduler](03_scheduler/) | Step count, temperature schedule ($\tau$ from 1.0 → 0.1), adaptive stopping criteria |
| 5.4 | [Entropy-Bounded Sampler](04_entropy_bounded_sampler/) | Canvas initialization, acceptance criterion, re-noising — how confident tokens are kept and uncertain ones are retried |

---

**Previous**: [Chapter 4 — The Architecture](../04_Architecture/) · **Next**: [Chapter 6 — Training](../06_Training/)
