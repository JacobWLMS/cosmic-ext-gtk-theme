# Identity Signal Analysis — Research Overview

Cross-experiment findings and synthesis. See individual experiment files for full details.

## Experiment Reports

- [EXP1_RESEARCH.md](EXP1_RESEARCH.md) — Paired Latent Frequency Analysis
- [EXP2_RESEARCH.md](EXP2_RESEARCH.md) — Within-Identity Latent Invariance

## Key Findings So Far

### Dual-Channel Identity Model in SDXL

| Role | Channel | Evidence |
|------|---------|----------|
| **Identity magnitude** | Ch 2 | Largest frequency difference when identity changes (Exp 1, 67% of pairs) |
| **Identity fingerprint** | Ch 3 | Best discrimination ratio — stable within identity, varies between (Exp 2) |
| **Scene/composition** | Ch 0 | High variance in both conditions — dominated by non-identity factors |
| **Neutral** | Ch 1 | Low reactivity (Exp 1), moderate discrimination (Exp 2) |

### Signal-to-Noise

- Identity is ~2-5% of total latent variance — scene/composition dominates
- Frequency analysis adds ~25% information beyond raw spatial differences (r=0.75)
- Simple variance-based discrimination ratios are low (<0.05) — more sophisticated methods needed

### Priority Next Steps

1. **Exp 3 (Identity Emergence)** — When during denoising do Ch 2 and Ch 3 diverge?
2. **Exp 7 (PCA on Ch 3)** — Is identity a linear subspace in the best discriminator channel?
3. **Exp 4 (ArcFace Correlation)** — Do latent-space identity differences correlate with perceptual identity metrics?

---

*Research conducted 2026-03-25. All results archived to Google Drive at identity_analysis/experiments/.*
