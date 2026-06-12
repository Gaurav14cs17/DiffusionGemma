# Chapter 6 — Training

How do you teach a pre-trained autoregressive model to do diffusion? DiffusionGemma starts from Gemma 4 26B (which already understands language) and fine-tunes it with a discrete diffusion objective — adding timestep conditioning, switching attention patterns, and training it to predict clean tokens from noisy inputs.

![Training pipeline](../diagrams/17_training.png)

---

## Topics

| # | Topic | Description |
|---|-------|-------------|
| 6.1 | [Training Objective](01_training_objective/) | ELBO for discrete diffusion, weighted cross-entropy loss, gradient trace through the model |
| 6.2 | [Fine-Tuning from Gemma 4](02_fine_tuning_from_gemma4/) | Timestep conditioning, what parameters change, what stays frozen, training hyperparameters |

---

**Previous**: [Chapter 5 — Inference](../05_Inference/) · **Next**: [Chapter 7 — Full Pipeline](../07_Full_Pipeline/)
