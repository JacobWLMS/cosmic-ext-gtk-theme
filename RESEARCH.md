# Identity Signal Analysis — Research Overview

Cross-experiment findings and synthesis. See individual experiment files for full details.

## Experiment Reports

- [EXP1_RESEARCH.md](EXP1_RESEARCH.md) — Paired Latent Frequency Analysis
- [EXP2_RESEARCH.md](EXP2_RESEARCH.md) — Within-Identity Latent Invariance
- [EXP3_RESEARCH.md](EXP3_RESEARCH.md) — Step-by-Step Identity Emergence
- [EXP5_RESEARCH.md](EXP5_RESEARCH.md) — Channel-Based Identity Transplant
- [EXP6_RESEARCH.md](EXP6_RESEARCH.md) — Channel Importance (Zeroing + Face Region)

## Key Findings So Far

### Channel Identity Model in SDXL (Revised after Exp 6)

SDXL's 4-channel VAE encodes identity information across a hierarchy, not in isolated channels:

| Role | Channel | Zeroing Impact | Discrimination Ratio | Evidence |
|------|---------|---------------|---------------------|----------|
| **Foundation** | Ch 0 | 0.607 (catastrophic) | 0.022 (worst) | Carries broadest image structure including face shape. Zeroing destroys the image. High variance in all conditions — identity is embedded but inseparable from scene. |
| **Identity fingerprint** | Ch 3 | 0.769 (significant) | 0.046 (best) | Most *discriminative* channel — stable within identity, varies between. Locks in at step 7-9 during denoising. Second-most damaging when zeroed. |
| **Identity detail** | Ch 2 | 0.894 (minor) | 0.034 | Largest *frequency* differences when identity changes (Exp 1, 67% of pairs). Carries fine identity refinements in last denoising steps. Zeroing barely affects ArcFace. |
| **Style/texture** | Ch 1 | 0.954 (negligible) | 0.042 | Not identity-critical. Zeroing has almost no perceptual identity impact. |

### Key Insight: Structural vs Discriminative Identity

There are two types of identity information in the latent space:

1. **Structural identity (Ch 0):** Basic face shape, features, proportions — embedded in the foundational image structure alongside scene/composition. Cannot be manipulated without destroying the image. This is *what makes a face look like a face*.

2. **Discriminative identity (Ch 3):** The information that distinguishes *who* someone is — separable from scene context, stable across varied prompts. This is *what makes Brad Pitt look like Brad Pitt*. This is the target for identity manipulation.

3. **Identity refinement (Ch 2):** Fine-grained details added in late denoising — subtle features that increase identity specificity but are noisy across generations.

### Implication for Identity Manipulation

To transplant or modify identity, **target Ch 3** — it's the only channel where identity is both meaningful AND separable. Ch 0 is too foundational (touching it breaks everything), and Ch 2 is too noisy (it varies per generation).

### Two-Phase Identity Emergence (Exp 3)

1. **Phase 1 (steps 0-5):** Foundation — Ch 0 saturates, basic face structure established
2. **Phase 2 (steps 6-11):** Identity refinement — Ch 3 fingerprint locks in (step 7-9), Ch 2 continues until step 10

### Signal-to-Noise

- Identity is ~2-5% of total latent variance — scene/composition dominates
- Frequency analysis adds ~25% information beyond raw spatial differences (r=0.75)
- Discrimination ratios are low (<0.05) — identity is embedded, not isolated

### Channel Swap Identity Transfer (Exp 5)

Identity transplant via channel swapping is possible but crude:
- **Ch 0 swap:** +0.367 ArcFace gain but SSIM 0.491 (scene destroyed)
- **Ch 3 swap:** +0.153 ArcFace gain with SSIM 0.543 (moderate scene preservation)
- **Ch 1/2 swap:** No identity transfer (~-0.05), confirming negative controls

The swap results perfectly mirror the zeroing results (Exp 6), fully validating the hierarchical model. For practical identity manipulation, channel-level swapping is too coarse — subspace methods within Ch 3 or identity-conditioned generation are needed.

### Priority Next Steps

1. **Exp 7 (PCA on Ch 3)** — Is discriminative identity a linear subspace within the fingerprint channel?
2. **Re-denoising experiment** — Swap Ch 3 then CONTINUE denoising instead of direct VAE decode
3. **Cross-model validation** — Test channel hierarchy on Lumina2 (16-channel VAE)

---

*Research conducted 2026-03-25. All results archived to Google Drive at identity_analysis/experiments/.*
