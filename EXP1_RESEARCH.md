# Research Log: Identity Signal Analysis in Diffusion Model Latent Spaces

## Experiment 1: Paired Latent Frequency Analysis

**Date:** 2025-03-25
**Status:** Complete (pilot, 5 seeds)
**Runtime:** T4 GPU, ~10 min per pair (50 min total)

### Question

When only identity changes between a prompt pair (same scene, different person), what changes in the latent's frequency domain?

### Setup

| Parameter | Value |
|-----------|-------|
| Model | SDXL Base 1.0 (fp16) |
| Resolution | 1024x1024 |
| Steps | 20 |
| Guidance scale | 7.5 |
| Negative prompt | anime, cartoon, illustration, painting... |
| Seeds | 5 (0-4) |
| Prompt pairs | 12 (including 1 control) |
| VAE | madebyollin/sdxl-vae-fp16-fix |
| VAE tiling | Enabled |

### Prompt Pairs

| Pair | Prompt A | Prompt B | Type |
|------|----------|----------|------|
| 0 | young man in cafe | old man in cafe | Age (same gender) |
| 1 | young woman in park | old woman in park | Age (same gender) |
| 2 | man in studio | woman in studio | Gender |
| 3 | middle-aged man at desk | middle-aged man at beach | **Control: scene change, same identity** |
| 4 | Asian man, white bg | European man, white bg | Ethnicity |
| 5 | African woman, white bg | European woman, white bg | Ethnicity |
| 6 | teenage boy, street | elderly man, street | Age (large gap) |
| 7 | young woman, library | elderly woman, library | Age (large gap) |
| 8 | tall man in suit | short man in suit | Build/height |
| 9 | photo of a man | photo of a woman | Gender (minimal) |
| 10 | photo of a child | photo of an old person | Age (maximal) |
| 11 | man, short brown hair | man, long blond hair | Subtle features |

### Results

#### 1. Channel Identity Sensitivity

SDXL has 4 VAE latent channels. Averaged across all 12 pairs and 5 seeds:

| Channel | Freq Magnitude (mean) | Freq Mag (std) | Raw Spatial Diff | Raw Diff (std) |
|---------|----------------------|----------------|-----------------|----------------|
| **Ch 0** | 43.37 | 9.90 | **0.4696** | 0.1717 |
| Ch 1 | 35.99 | 7.04 | 0.3156 | 0.1203 |
| **Ch 2** | **44.82** | 9.11 | 0.3261 | 0.1042 |
| Ch 3 | 37.61 | 7.50 | 0.3431 | 0.1262 |

**Key finding:** Channels 0 and 2 carry the most identity-related frequency differences. Channel 2 was the most identity-sensitive channel in **8 out of 12 pairs** (67%), while Channel 0 was top in the remaining 4. Channels 1 and 3 were never the most identity-sensitive channel.

**Interesting divergence:** Channel 0 has the highest *spatial* (raw pixel) difference (0.47) but Channel 2 has the highest *frequency* magnitude difference (44.82). This suggests Ch 0 captures broad/low-frequency identity changes (overall appearance shift) while Ch 2 captures more structured frequency-domain identity signals.

#### 2. Control Pair Analysis

Pair 3 is the control — same identity description ("middle-aged man") but different scenes (office vs beach).

| Condition | Mean Freq Magnitude |
|-----------|-------------------|
| Control (scene change, same identity) | 49.22 |
| Identity change pairs (avg) | 39.65 |
| **Ratio** | **0.81x** |

**Surprising result:** The control pair (scene change only) produced *higher* frequency differences than identity-change pairs. This means scene/composition changes dominate the overall frequency signature more than identity changes at this resolution. Identity differences are present but are **not the dominant signal** — they are a subset of the total latent variation.

This is actually an important finding: to isolate identity signals, we need to **subtract out or control for scene/composition variation**, or work in a subspace where identity is separable.

#### 3. Per-Pair Rankings

Pairs ranked by average frequency magnitude of the identity difference:

| Rank | Pair | Description | Freq Mag |
|------|------|-------------|----------|
| 1 | 7 | young vs elderly woman (library) | 49.37 |
| 2 | 3 | **CONTROL** (same man, different scene) | 49.22 |
| 3 | 5 | African vs European woman | 46.54 |
| 4 | 10 | child vs old person | 44.16 |
| 5 | 0 | young vs old man (cafe) | 43.06 |

Large age gaps and ethnicity differences produce the biggest frequency signatures. Subtle changes (pair 8: tall vs short, pair 11: hair color) produce the smallest.

#### 4. Spatial-Frequency Correlation

Correlation between raw spatial difference and frequency magnitude: **r = 0.75**

This moderate-to-strong correlation means frequency analysis adds some information beyond raw pixel differences, but they're not fully redundant. ~44% of frequency variance is unexplained by spatial differences — this is where structured identity signals may live.

### Plots

All plots saved to `results/exp1/` and Google Drive at `identity_analysis/exp1/`.

- `avg_diff_frequency_magnitude.png` — Heatmap of average FFT magnitude of identity differences per channel
- `frequency_band_comparison.png` — Energy distribution across frequency bands for identity A vs B
- `pair{N}_identity_{a,b}.png` — Sample generated images for each prompt pair (seed 0)

### Conclusions

1. **Identity information is NOT uniformly distributed across VAE channels.** Channels 0 and 2 carry disproportionately more identity-related variation in frequency space. Ch 1 and Ch 3 are relatively identity-insensitive in terms of raw frequency magnitude.

2. **Channel 2 has the largest frequency-domain identity response** — it was the most reactive in 67% of prompt pairs. However, later experiments (Exp 6) showed this reactivity doesn't translate to perceptual identity importance — Ch 2's zeroing barely affects ArcFace similarity.

3. **Scene/composition variation dominates over identity variation.** The control pair (same identity, different scene) produced higher frequency differences than most identity-change pairs. Identity signals are present but embedded within larger scene variation.

4. **Large identity gaps produce stronger signals.** Age extremes (child vs elderly) and ethnicity differences produce ~1.5x the frequency difference of subtle changes (hair color, build).

5. **Frequency analysis adds value beyond spatial analysis** (r=0.75, not r=1.0), suggesting structured identity information exists in the frequency domain that isn't captured by raw pixel differences.

> **Note (post-Exp 6 revision):** The original interpretation that Ch 0 was purely "scene/composition" was revised. Ch 0 is actually the foundation channel carrying all broad image structure including face shape. Ch 2's high reactivity turned out to be noise rather than signal — see Exp 6 for the zeroing results that clarified this.

### Open Questions

1. Why does the control pair (scene change) have higher frequency magnitude than identity changes? Is this because scene/background changes affect more spatial area than face/identity changes?
2. Would the channel ranking hold at higher seed counts (n=50)?
3. ~~Can we design a "frequency-domain identity metric" that isolates the Ch 2 mid-frequency signal?~~ (Answered by Exp 6: Ch 2 is not perceptually important for identity)

---

*Experiment run on Google Colab T4, 2026-03-25. Results archived to Google Drive.*
