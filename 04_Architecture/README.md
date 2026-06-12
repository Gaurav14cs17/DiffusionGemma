# Chapter 4 — The Architecture

DiffusionGemma doesn't train a new model from scratch. It takes an existing **Gemma 4 26B** autoregressive model and patches it to perform diffusion — using a single model that plays *two roles* (encoder and denoiser) by switching its attention mask between causal and bidirectional.

![The full DiffusionGemma Architecture](../diagrams/11_full_architecture.png)

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 4.0 | [Bridge — Theory to Architecture](00_bridge_from_theory_to_architecture/) | Connecting discrete diffusion theory to DiffusionGemma's design, the attention mask trick, cost analysis |
| 4.1 | [Gemma 4 Base Model](01_gemma4_base_model/) | MoE router math, Grouped Query Attention (GQA), RoPE, layer anatomy — what the base model provides |
| 4.2 | [Encoder-Denoiser Patch](02_encoder_denoiser_patch/) | One model, two roles — how the encoder/denoiser split works with full numerical trace |
| 4.3 | [Bidirectional Attention](03_bidirectional_attention/) | Inside a single layer — causal vs. bidirectional walkthrough, attention mask matrices |
| 4.4 | [KV-Cache Sharing](04_kv_cache_sharing/) | Data flow traced through layers — how the encoder's cached KV pairs feed into the denoiser, memory savings |

---

**Previous**: [Chapter 3 — Discrete Diffusion](../03_Discrete_Diffusion/) · **Next**: [Chapter 5 — Inference](../05_Inference/)
