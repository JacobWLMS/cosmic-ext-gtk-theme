# Research Log: Identity Signal Analysis in Diffusion Model Latent Spaces

## Experiment 2: Within-Identity Latent Invariance

**Date:** 2025-03-25
**Status:** Complete
**Runtime:** T4 GPU, ~35 min total (100 generations)

### Question

What stays constant in the latent frequency domain when the same person appears in different contexts? Which channels best **discriminate** between identities (high across-identity variance, low within-identity variance)?

### Setup

| Parameter | Value |
|-----------|-------|
| Model | SDXL Base 1.0 (fp16) |
| Resolution | 1024x1024 |
| Steps | 20 |
| Guidance scale | 7.5 |
| Negative prompt | anime, cartoon, illustration, painting... |
| Celebrities | Brad Pitt, Taylor Swift, Morgan Freeman, Scarlett Johansson, Keanu Reeves |
| Prompts per celebrity | 20 (varied scenes, lighting, clothing) |
| Total generations | 100 |
| VAE | madebyollin/sdxl-vae-fp16-fix |
| VAE tiling | Enabled |

### Method

For each celebrity, generate 20 images across varied contexts (studio portrait, park, restaurant, black & white, etc.) using SDXL's strong celebrity priors. Re-encode each generated image through the VAE encoder to get clean latents. Compute per-channel variance **within** each celebrity (how much the same person varies across scenes) and **across** celebrities (how much different people differ). The discrimination ratio (across / within) measures how well each channel separates identities.

### Results

#### 1. Discrimination Ratios

The key metric: across-identity variance / within-identity variance. Higher = better identity separator.

| Channel | Within Variance | Across Variance | Discrimination Ratio | Rank |
|---------|----------------|-----------------|---------------------|------|
| Ch 0 | 0.726 | 0.016 | 0.022 | 4th (worst) |
| Ch 1 | 0.420 | 0.018 | 0.042 | **2nd** |
| Ch 2 | 0.582 | 0.020 | 0.034 | 3rd |
| **Ch 3** | **0.454** | **0.021** | **0.046** | **1st (best)** |

**Channel ranking (best discriminator to worst): Ch 3 > Ch 1 > Ch 2 > Ch 0**

#### 2. Exp 1 vs Exp 2 Contradiction

| Metric | Best Channel | Interpretation |
|--------|-------------|----------------|
| Exp 1: Largest frequency difference when identity changes | **Ch 2** (67% of pairs) | Most *reactive* to identity change |
| Exp 2: Best discrimination ratio (across/within variance) | **Ch 3** | Most *reliable* identity separator |

**This is a critical finding.** Ch 2 reacts most strongly to identity changes (Exp 1) but is noisy — it also varies a lot for the same person across scenes. Ch 3 is a more reliable identity channel because it has:
- The highest across-identity variance (0.021) — it genuinely differs between people
- Moderate within-identity variance (0.454) — it stays relatively stable for the same person

Ch 0 is the worst discriminator despite having high absolute differences (Exp 1) because its within-identity variance is massive (0.726) — it changes dramatically with scene/lighting regardless of identity.

#### 3. Frequency Domain Variance

| Channel | Within-Identity Freq Variance | Across-Identity Freq Variance |
|---------|------------------------------|------------------------------|
| Ch 0 | 954.7 | 28.5 |
| Ch 1 | 473.1 | 14.5 |
| Ch 2 | 593.3 | 22.7 |
| Ch 3 | 472.2 | 17.2 |

All channels have much higher within-identity than across-identity frequency variance, confirming that scene/composition dominates over identity in raw frequency space. The identity signal is ~2-5% of total variance.

### Conclusions

1. **Ch 3 is the best identity discriminator** in SDXL's VAE, not Ch 2. Discrimination requires both high between-identity variance AND low within-identity variance.

2. **Reactivity does not equal reliability.** Ch 2 reacts most to identity changes (Exp 1) but is noisy. Ch 3 is quieter but more consistent — a better "identity fingerprint" channel.

3. **Identity signal is ~2-5% of total latent variance.** Scene/composition dominates. Any identity extraction method must account for this signal-to-noise ratio.

4. **All discrimination ratios are low (<0.05).** Identity is not cleanly separable in raw latent space using simple variance analysis. More sophisticated methods (PCA within channels, frequency band filtering) may be needed.

### Cross-Experiment Synthesis: Dual-Channel Identity Model

The combination of Exp 1 and Exp 2 reveals a **dual-channel identity model** in SDXL:

| Role | Channel | Evidence |
|------|---------|----------|
| **Identity magnitude** | Ch 2 | Largest frequency difference when identity changes (Exp 1, 67% of pairs) |
| **Identity fingerprint** | Ch 3 | Best discrimination ratio — stable within identity, varies between (Exp 2) |
| **Scene/composition** | Ch 0 | High variance in both conditions — dominated by non-identity factors |
| **Neutral** | Ch 1 | Low reactivity (Exp 1), moderate discrimination (Exp 2) |

### Implications for Next Experiments

- **Exp 3 (Identity Emergence):** Track Ch 2 and Ch 3 separately during denoising — do they diverge at different timesteps? Ch 3 may lock in identity earlier.
- **Exp 5 (Frequency Injection):** Target Ch 3 for identity transfer, not Ch 2 — it's the most reliable identity carrier.
- **Exp 7 (PCA):** Run PCA on Ch 3 specifically — it may have the cleanest linear identity subspace given its high discrimination ratio.
- **Combined approach:** Use Ch 3 for identity discrimination, Ch 2 for identity magnitude — together they may provide both sensitivity and stability.
- **Low discrimination ratios suggest** frequency-domain analysis alone may not be sufficient. Cross-referencing with ArcFace embeddings (Exp 4) will be important.

### Open Questions

1. Is Ch 3's discrimination advantage specific to SDXL, or is it a general VAE property? (Test with Lumina2's 16-channel VAE)
2. Does the discrimination ratio improve if we analyze specific frequency bands within Ch 3 rather than the full spectrum?
3. Would using the diffusion latent directly (instead of VAE re-encoding) change the channel rankings?
4. Can we combine Ch 2 (magnitude) and Ch 3 (fingerprint) into a composite identity metric?

### Plots

Saved to Google Drive at `identity_analysis/experiments/exp2_within_identity/`.

- `plots/variance_within_vs_across.png` — Within vs across identity variance per channel
- `plots/discrimination_ratio.png` — Discrimination ratio bar chart
- `plots/freq_variance_within_vs_across.png` — Same analysis in frequency domain
- `plots/mean_freq_mag_{celebrity}.png` — Mean frequency magnitude heatmap per celebrity
- `samples/{Celebrity}_sample_{0-3}.png` — Sample generated images

---

*Experiment run on Google Colab T4, 2026-03-25. Results archived to Google Drive at identity_analysis/experiments/exp2_within_identity/.*
