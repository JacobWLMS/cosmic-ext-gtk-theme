# Identity Signal Analysis — Research Overview

Cross-experiment findings and synthesis. See individual experiment files for full details.

## Experiment Reports

- [EXP1_RESEARCH.md](EXP1_RESEARCH.md) — Paired Latent Frequency Analysis
- [EXP2_RESEARCH.md](EXP2_RESEARCH.md) — Within-Identity Latent Invariance
- [EXP3_RESEARCH.md](EXP3_RESEARCH.md) — Step-by-Step Identity Emergence

## Key Findings So Far

### Dual-Channel Identity Model in SDXL

| Role | Channel | Evidence |
|------|---------|----------|
| **Identity magnitude** | Ch 2 | Largest frequency difference when identity changes (Exp 1, 67% of pairs). Last to settle during denoising — step 10/12 (Exp 3). |
| **Identity fingerprint** | Ch 3 | Best discrimination ratio (Exp 2). Locks in 2-3 steps before Ch 2 (Exp 3). |
| **Scene/composition** | Ch 0 | High variance in both conditions (Exp 2). Saturates by step 5-6 (Exp 3). |
| **Neutral** | Ch 1 | Low reactivity (Exp 1), moderate discrimination (Exp 2). Late settling (Exp 3). |

### Two-Phase Identity Emergence (Exp 3)

1. **Phase 1 (steps 0-5):** Coarse structure — scene/composition settles, broad identity features encoded
2. **Phase 2 (steps 6-11):** Identity refinement — Ch 3 fingerprint locks in (step 7-9), Ch 2 magnitude refines until step 10

### Signal-to-Noise

- Identity is ~2-5% of total latent variance — scene/composition dominates
- Frequency analysis adds ~25% information beyond raw spatial differences (r=0.75)
- Simple variance-based discrimination ratios are low (<0.05) — more sophisticated methods needed

### Priority Next Steps

1. **Exp 7 (PCA on Ch 3)** — Is identity a linear subspace in the best discriminator channel?
2. **Exp 4 (ArcFace Correlation)** — Do latent-space identity differences correlate with perceptual identity metrics?
3. **Exp 5 (Frequency Injection at mid-denoising)** — Can we transplant identity by swapping Ch 3 at step 7?

---

*Research conducted 2026-03-25. All results archived to Google Drive at identity_analysis/experiments/.*
