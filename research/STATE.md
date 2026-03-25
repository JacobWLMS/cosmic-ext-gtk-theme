# State of Affairs
*Last updated after: Cycle 6 (Exp 5 V2) — 2026-03-25*

## Current Model: Hierarchical Channel Identity in SDXL VAE

SDXL's 4-channel VAE latent space encodes identity information hierarchically:

| Channel | Role | Identity Contribution | Confidence |
|---------|------|----------------------|------------|
| Ch 0 | Luminance / foundation | Carries ALL broad structure including identity + scene (inseparable) | HIGH |
| Ch 3 | Pattern / structure | Best *discriminative* identity signal (separable from scene) | HIGH |
| Ch 2 | Warm-cool chrominance | Fine detail, minimal identity | HIGH |
| Ch 1 | Red-cyan chrominance | Style/texture, identity-neutral | HIGH |

### Key Beliefs

- **HIGH confidence:** The channel hierarchy (Ch 0 > Ch 3 > Ch 2 ≈ Ch 1) is consistent across 5 experiments using different methods (frequency analysis, discrimination ratios, emergence tracking, zeroing, channel swapping).
- **HIGH confidence:** Identity emerges at steps 7-9 during 12-step denoising, with Ch 3 diverging before Ch 0.
- **HIGH confidence:** The UNet actively heals single-channel perturbations during denoising. Raw latent manipulation cannot overpower text-conditioned denoising.
- **MEDIUM confidence:** Ch 3 carries a *separable* identity fingerprint — but we don't yet know if it's a linear subspace or something more complex.
- **LOW confidence:** Identity manipulation at the latent level is fundamentally limited by cross-channel consistency enforcement.

## Evidence Map

| Claim | Supporting Experiments | Strength |
|-------|----------------------|----------|
| Ch 0 is foundational (identity + scene) | Exp 1 (frequency), Exp 2 (discrimination), Exp 5 (swap), Exp 6 (zeroing) | Strong (4 exps) |
| Ch 3 is best identity discriminator | Exp 2 (ratio 0.046), Exp 6 (zeroing 0.769), Exp 3 (emergence) | Strong (3 exps) |
| Ch 1/2 are identity-neutral | Exp 2, Exp 5 (negative control), Exp 6 | Strong (3 exps) |
| UNet heals channel swaps | Exp 5 V2 (SSIM 0.67-0.90 to target) | Moderate (1 exp, 1 seed) |
| Identity emerges steps 7-9 | Exp 3 | Moderate (1 exp) |

## Open Questions (ranked by priority)

1. **Is identity a linear subspace within Ch 3?** → Exp 7 (PCA), running now
2. **Can multi-channel or late-step swaps overcome UNet healing?** → Future Exp 5 follow-up
3. **Does this channel hierarchy hold for other architectures?** → Lumina2 (16-ch VAE), Flux
4. **What do the PCA components actually represent?** → May reveal scene/style axes as bonus

## Rejected Hypotheses

| Hypothesis | Rejected By | Why |
|-----------|------------|-----|
| Ch 2 is the primary identity channel | Exp 6 (zeroing had minimal impact) | Exp 1 frequency shifts were misleading — Ch 2 changes most in frequency domain but isn't identity-critical |
| Channel swapping can transfer identity | Exp 5 V2 (UNet healing) | Single-channel swap at mid-denoising is corrected by remaining steps |
| Direct latent decode shows swap effects | Exp 5 V1→V2 comparison | Intermediate latents are noise, not images — ArcFace on garbled output overstated effects |
