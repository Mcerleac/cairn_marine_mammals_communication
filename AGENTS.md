# AGENTS.md — AI Context & Working Conventions
This document specifies system context, technical constraints, data contracts,
and conventions for AI assistants (Claude, ChatGPT, GitHub Copilot, Cursor, etc.)
working in this repository.
---
## 1. Project Context & Objectives
- **Domain:** Bioacoustic Language Modeling for Marine Mammal Communication (Humpback
Whales, extending to Orcas).
- **Core Problem:** Underwater waveforms are continuous; we convert continuous hydrophone
audio into discrete token sequences via a 9-codebook neural audio codec and train long-context
autoregressive transformers for generative and acoustic feature modeling.
- **Data Source:** NOAA SanctSound passive bioacoustic monitoring (Hawaiian stations HI01,
HI03, HI04, HI05), publicly available at `gs://noaa-passive-bioacoustic/`.
- **Released Benchmarks:**
- **9CB 10k Large v2 (479M params):** 10,240 context (~13s), Hybrid SWA + MoE, achieves
**203.2 validation perplexity** (loss 5.314).
- **9CB 32k Medium (205M params):** 32,768 context, 8 experts (top-2), perplexity 208.5.
- **9CB 128k NSA (375M params):** 131,072 context (~2.8 minutes audio), 16 experts (top-1),
perplexity 634.6.
- **Active Goals:** Context scaling to 128k tokens, LR reduction/resume strategies, spectral loss
evaluation (analog -> tokens -> analog), dataset expansion, and interpretability via Sparse
Autoencoders (SAEs).
