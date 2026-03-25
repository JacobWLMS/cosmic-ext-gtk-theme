# State of Affairs
*Last updated after: Cycle 8 (Exploration Tests A/B/C) — 2026-03-25*

## Current Model: Identity Lives in Conditioning, Not Latent Space

### The Core Finding

Identity information is **linearly separable in CLIP text embedding space** (silhouette +0.40) but **NOT separable in VAE latent space** (all scores negative, all methods). The UNet distributes identity across all latent channels during denoising, making it inseparable in the output latent.

### Latent Space: How Identity Gets Distributed (SDXL)

| Channel | Role | Identity Contribution | Confidence |
|---------|------|----------------------|------------|
| Ch 0 | Luminance / foundation | Carries ALL broad structure (identity + scene inseparable) | HIGH |
| Ch 3 | Pattern / structure | Best *discriminative* signal but NOT linearly separable | HIGH |
| Ch 2 | Warm-cool chrominance | Fine detail, minimal identity | HIGH |
| Ch 1 | Red-cyan chrominance | Style/texture, identity-neutral | HIGH |

### Conditioning Space: Where Identity Actually Lives

| Space | Identity Clustering | Method | Notes |
|-------|-------------------|--------|-------|
| CLIP mean-pooled embeddings | **+0.403** (strong) | PCA 5d | Identity distributed across token positions |
| CLIP CLS token only | 0.000 (none) | PCA | CLS alone doesn't capture identity |
| VAE latent (full, PCA 20d) | -0.079 (anti-cluster) | PCA | Scene/style dominates |
| VAE latent (Ch3, PCA 20d) | -0.084 (anti-cluster) | PCA | No advantage over full |
| VAE latent (any, t-SNE/UMAP) | -0.15 to -0.22 | Nonlinear | Worse than PCA |

### Key Beliefs

- **HIGH confidence:** Identity is NOT linearly separable in VAE latent space (confirmed by PCA, t-SNE, UMAP, scene-controlled experiments)
- **HIGH confidence:** Identity IS linearly separable in CLIP text embedding space (silhouette +0.40, mean-pooled)
- **HIGH confidence:** The UNet enforces cross-channel consistency during denoising, "healing" single-channel perturbations
- **HIGH confidence:** SDXL channel hierarchy (Ch 0 > Ch 3 > Ch 2 ≈ Ch 1) is real but describes HOW identity gets distributed, not WHERE to manipulate it
- **MEDIUM confidence:** These findings are architecture-general (pending Z-Image Turbo and Lumina2 validation)
- **LOW confidence:** CLIP identity directions can be extracted and used for practical identity manipulation

## Evidence Map

| Claim | Supporting Experiments | Strength |
|-------|----------------------|----------|
| Identity clusters in CLIP space | Exploration Test C (silhouette +0.40) | Moderate (1 test, 1 model) |
| Identity does NOT cluster in latent space | Exp 7, Tests A/B (all negative across methods) | Strong (4 methods tested) |
| SDXL channel hierarchy | Exp 1, 2, 3, 5, 6 | Strong (5 experiments) |
| UNet heals latent perturbations | Exp 5 V2 | Moderate (1 experiment) |
| Scene variation dominates latent space | Exp 7, Test B | Strong (controlled experiment) |

## Open Questions (ranked by priority)

1. **Can we find identity-specific directions in CLIP space?** → Next experiment: PCA/probe on CLIP embeddings for identity vs non-identity features
2. **Can manipulating CLIP identity directions produce controlled identity changes in generated images?** → Key practical test
3. **Does the latent-space finding hold for Z-Image Turbo and Lumina2?** → Cross-architecture validation running
4. **What about cross-attention maps?** → Identity may also be visible in where the UNet attends to identity tokens
5. **Can we do this at inference time without retraining?** → IP-Adapter already does this; our research explains WHY it works

## Rejected Hypotheses

| Hypothesis | Rejected By | Why |
|-----------|------------|-----|
| Ch 2 is the primary identity channel | Exp 6 (zeroing) | Frequency sensitivity ≠ identity importance |
| Single-channel swap can transfer identity | Exp 5 V2 (UNet healing) | Denoising corrects perturbations |
| Identity forms linear subspace in Ch 3 | Exp 7 (negative silhouette) | Scene/style variation dominates |
| Identity forms nonlinear clusters in latent space | Test A (t-SNE/UMAP worse than PCA) | Not a manifold issue |
| Scene confound explains latent clustering failure | Test B (scene-controlled still negative) | Identity genuinely isn't there |

## Research Direction

**Pivot from latent space → conditioning space.** The channel analysis (Exp 1-7) established HOW identity gets distributed through the VAE/UNet pipeline. The exploration tests revealed WHERE it actually lives — upstream in CLIP embeddings. The next phase focuses on:

1. Extracting identity-specific directions from CLIP space
2. Testing whether conditioning manipulation produces controlled identity changes
3. Cross-architecture validation (Z-Image, Lumina2)
4. Cross-attention analysis as a bridge between conditioning and latent space
