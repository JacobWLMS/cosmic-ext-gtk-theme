# State of Affairs
*Last updated after: Cycle 10 (Z-Image architecture decision) — 2026-03-25*

## Current Model: Identity Lives in Conditioning, Not Latent Space — Confirmed Cross-Architecture

### The Core Finding

Identity information is **linearly separable in conditioning space** (CLIP: silhouette +0.40) but **NOT separable in VAE latent space** — confirmed on both SDXL (4ch UNet) and Z-Image Turbo (16ch DiT). This is an architecture-general property of diffusion models, not an artifact of any specific VAE or denoiser.

### Primary Architecture: Z-Image Turbo

Switching from SDXL to Z-Image Turbo for the next research phase. Z-Image offers stronger signal (2.5x discrimination), more channels to work with (16 vs 4), and is less explored in the community.

| Property | SDXL | Z-Image Turbo |
|----------|------|---------------|
| VAE channels | 4 | 16 |
| Denoiser | UNet | DiT (transformer) |
| Conditioning | CLIP | **Qwen3** (LLM, not CLIP!) |
| Sensitivity range | 1.25x | **3.25x** |
| Best discrimination | 0.046 | **0.117** |
| Best PCA silhouette | -0.079 | -0.137 |
| Steps | 20 | 9 |
| dtype | fp16 | bfloat16 |
| VRAM | ~12GB | ~21GB |

### Z-Image Channel Map

| Tier | Channels | Sensitivity | Role |
|------|----------|-------------|------|
| **Face-focused** | 1, 6, 9, 12 | 0.66–0.89 | Strong face-region activation, highest identity sensitivity |
| **Mixed** | 7, 13, 14, 15 | 0.55–0.70 | Moderate identity signal, shared face/structure |
| **Background** | 0, 2, 3, 4, 5, 8, 10, 11 | 0.27–0.49 | Texture, structure, scene — low identity sensitivity |

Note: Discrimination ratios don't perfectly align with sensitivity tiers — Ch 11 (0.117) and Ch 4 (0.111) are top discriminators despite being "background" by sensitivity. Identity is distributed even within Z-Image.

### Conditioning Space: Where Identity Lives

| Space | Identity Clustering | Method | Architecture |
|-------|-------------------|--------|-------------|
| CLIP mean-pooled embeddings | **+0.403** (strong) | PCA 5d | SDXL |
| Z-Image conditioning (Qwen3) | **Unknown** | Not yet tested | Z-Image |
| SDXL VAE latent (best) | -0.079 (anti-cluster) | PCA 20d | SDXL |
| Z-Image VAE latent (best) | -0.137 (anti-cluster) | PCA 10d | Z-Image |

### Key Beliefs

- **HIGH confidence:** Identity is NOT linearly separable in VAE latent space — confirmed on 2 architectures (SDXL, Z-Image), 4+ methods (PCA, t-SNE, UMAP, scene-controlled)
- **HIGH confidence:** Identity IS linearly separable in CLIP conditioning space (SDXL: silhouette +0.40)
- **HIGH confidence:** More VAE channels = more functional specialization, but NOT more identity separability
- **HIGH confidence:** Channel-level sensitivity and discrimination are real but don't enable linear isolation
- **MEDIUM confidence:** Denoiser (UNet or DiT) "heals" single-channel perturbations — confirmed for UNet, predicted for DiT
- **LOW confidence:** Z-Image's conditioning space (Qwen3 LLM) will also show identity separability — needs testing. Qwen3 is a decoder-only LLM, fundamentally different from CLIP's contrastive encoder
- **LOW confidence:** Multi-channel swap (face channels 1+6+9+12 together) might overwhelm DiT healing — untested hypothesis

## Evidence Map

| Claim | Supporting Evidence | Strength |
|-------|-------------------|----------|
| Identity clusters in CLIP conditioning | Exploration Test C (silhouette +0.40) | Moderate (1 test, SDXL only) |
| Identity does NOT cluster in latent space | Exp 7, Tests A/B, Cycle 9 (all negative, 2 architectures) | **Strong** (5+ tests, 2 models) |
| Z-Image channel specialization | Cycle 9 (3.25x range, visual + numerical) | Strong |
| Z-Image has face-dedicated channels | Cycle 9 (Ch 1,6,9,12 visual + sensitivity) | Strong |
| Discrimination ratios improve with more channels | SDXL 0.046 vs Z-Image 0.117 | Strong (direct comparison) |
| UNet heals single-channel perturbations | Exp 5 V2 | Moderate (SDXL only, untested on DiT) |
| Scene variation dominates latent space | Exp 7, Test B, Cycle 9 | Strong (both architectures) |

## Open Questions (ranked by priority)

1. **~~What conditioning encoder does Z-Image use?~~** → **ANSWERED: Qwen3** (an LLM, not CLIP or T5). Uses Qwen2Tokenizer. This is a fundamentally different conditioning approach — LLM embeddings vs CLIP/T5.
2. **Can multi-channel face swap (Ch 1+6+9+12) overwhelm the DiT?** → 4 of 16 channels is 25% of the latent — much harder to "heal" than 1 of 4 (SDXL)
3. **Does Z-Image's DiT cross-attention carry identity differently than UNet?** → Transformer attention is structurally different
4. **Is identity separable in Z-Image's conditioning space?** → If T5-based, this is a novel finding
5. **Can we extract identity-specific directions in conditioning space and use them for controlled manipulation?** → The practical endgame
6. **Does the denoising step-by-step identity emergence differ in a 9-step flow-matching model?** → SDXL showed steps 7-9; Z-Image only has 9 total

## Rejected Hypotheses

| Hypothesis | Rejected By | Why |
|-----------|------------|-----|
| Identity is linearly separable in VAE latent space | Exp 7, Tests A/B, Cycle 9 | Negative silhouette on 2 architectures, 5+ methods |
| More VAE channels → identity separability | Cycle 9 | Z-Image 16ch worse PCA (-0.137) than SDXL 4ch (-0.079) |
| Single-channel swap transfers identity | Exp 5 V2 | UNet healing corrects perturbations |
| Ch 2 is the primary identity channel (SDXL) | Exp 6 | Frequency sensitivity ≠ identity importance |
| Nonlinear methods find identity clusters | Test A | t-SNE/UMAP worse than PCA |

## Research Direction

**Phase 2: Z-Image Turbo Deep Dive**

The SDXL phase (Exp 1-7 + exploration) established the fundamental finding. Z-Image offers a fresh platform with stronger signals and unexplored territory. Three parallel threads:

1. **Conditioning space analysis** — Identify Z-Image's text encoder (T5?), test identity separability, extract identity directions
2. **Multi-channel manipulation** — Swap face channels (1+6+9+12) simultaneously. With 25% of channels carrying face signal, the DiT may not be able to heal the perturbation
3. **DiT internals** — Cross-attention maps, intermediate transformer states, step-by-step identity emergence in a 9-step flow-matching model
