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

1. **Identity information is NOT uniformly distributed across VAE channels.** Channels 0 and 2 carry disproportionately more identity-related variation in frequency space. Ch 1 and Ch 3 are relatively identity-insensitive.

2. **Channel 2 is the primary identity channel in frequency domain** — it was the most sensitive in 67% of prompt pairs. Channel 0 is secondary and captures more spatial/compositional identity shifts.

3. **Scene/composition variation dominates over identity variation.** The control pair (same identity, different scene) produced higher frequency differences than most identity-change pairs. Identity signals are present but embedded within larger scene variation.

4. **Large identity gaps produce stronger signals.** Age extremes (child vs elderly) and ethnicity differences produce ~1.5x the frequency difference of subtle changes (hair color, build).

5. **Frequency analysis adds value beyond spatial analysis** (r=0.75, not r=1.0), suggesting structured identity information exists in the frequency domain that isn't captured by raw pixel differences.

### Implications for Next Experiments

- **Exp 2 (Within-Identity Invariance):** Critical to run next — we need to identify what *stays constant* across scenes for the same identity, to separate identity from scene signals.
- **Exp 3 (Identity Emergence):** The finding that Ch 2 is primary for identity suggests we should track Ch 2 specifically during denoising to see when identity "locks in."
- **Exp 6 (Channel Importance):** Our results already preview this — Ch 0 and Ch 2 are the candidates. Exp 6 should validate by zeroing out channels and measuring ArcFace similarity impact.
- **Exp 7 (PCA):** Given the channel structure we found, PCA should be run per-channel rather than on the full latent — Ch 2 may have the cleanest identity subspace.
- **Exp 5 (Frequency Injection):** The control pair result suggests we need to inject within specific frequency bands AND specific channels (Ch 2) to transfer identity without transferring scene.

### Open Questions

1. Why does the control pair (scene change) have higher frequency magnitude than identity changes? Is this because scene/background changes affect more spatial area than face/identity changes?
2. Is Channel 2's identity sensitivity specific to SDXL, or is it a general property of VAE architectures?
3. Would the channel ranking hold at higher seed counts (n=50)?
4. Can we design a "frequency-domain identity metric" that isolates the Ch 2 mid-frequency signal?

---

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
| Celebrities | Brad Pitt, Taylor Swift, Morgan Freeman, Scarlett Johansson, Keanu Reeves |
| Prompts per celebrity | 20 (varied scenes, lighting, clothing) |
| Total generations | 100 |

### Results

#### Discrimination Ratios

The discrimination ratio = across-identity variance / within-identity variance. Higher means the channel better separates different people while staying stable for the same person.

| Channel | Within Variance | Across Variance | Discrimination Ratio | Rank |
|---------|----------------|-----------------|---------------------|------|
| Ch 0 | 0.726 | 0.016 | 0.022 | 4th (worst) |
| Ch 1 | 0.420 | 0.018 | 0.042 | **2nd** |
| Ch 2 | 0.582 | 0.020 | 0.034 | 3rd |
| **Ch 3** | **0.454** | **0.021** | **0.046** | **1st (best)** |

**Channel ranking (best discriminator → worst): Ch 3 > Ch 1 > Ch 2 > Ch 0**

#### Key Insight: Exp 1 vs Exp 2 Contradiction

| Metric | Best Channel | Interpretation |
|--------|-------------|----------------|
| Exp 1: Largest frequency difference when identity changes | **Ch 2** (67% of pairs) | Most *reactive* to identity change |
| Exp 2: Best discrimination ratio (across/within variance) | **Ch 3** | Most *reliable* identity separator |

**This is a critical finding.** Ch 2 reacts most strongly to identity changes (Exp 1) but is noisy — it also varies a lot for the same person across scenes. Ch 3 is a more reliable identity channel because it has:
- The highest across-identity variance (0.021) — it differs between people
- Moderate within-identity variance (0.454) — it stays relatively stable for the same person

Ch 0 is the worst discriminator despite having high absolute differences (Exp 1) because its within-identity variance is massive (0.726) — it changes dramatically with scene/lighting regardless of identity.

#### Frequency Domain Variance

| Channel | Within-Identity Freq Variance | Across-Identity Freq Variance |
|---------|------------------------------|------------------------------|
| Ch 0 | 954.7 | 28.5 |
| Ch 1 | 473.1 | 14.5 |
| Ch 2 | 593.3 | 22.7 |
| Ch 3 | 472.2 | 17.2 |

All channels have much higher within-identity than across-identity frequency variance, confirming that scene/composition dominates over identity in raw frequency space. The identity signal is ~2-5% of total variance.

### Conclusions

1. **Ch 3 is the best identity discriminator** in SDXL's VAE, not Ch 2. Discrimination requires both high between-identity variance AND low within-identity variance.

2. **Reactivity ≠ reliability.** Ch 2 reacts most to identity changes (Exp 1) but is noisy. Ch 3 is quieter but more consistent — a better "identity fingerprint" channel.

3. **Identity signal is ~2-5% of total latent variance.** Scene/composition dominates. Any identity extraction method must account for this signal-to-noise ratio.

4. **All discrimination ratios are low (<0.05).** Identity is not cleanly separable in raw latent space using simple variance analysis. More sophisticated methods (PCA within channels, frequency band filtering) may be needed.

### Implications

- **Exp 5 (Frequency Injection):** Should target Ch 3 for identity transfer, not Ch 2 — it's the most reliable identity carrier.
- **Exp 7 (PCA):** Run PCA on Ch 3 specifically — it may have the cleanest linear identity subspace.
- **Combined approach:** Use Ch 3 for identity discrimination, Ch 2 for identity magnitude — together they may provide both sensitivity and stability.
- **The low discrimination ratios suggest** that frequency-domain analysis alone may not be sufficient for identity isolation. Cross-referencing with ArcFace embeddings (Exp 4) will be important.

### Drive Structure

```
identity_analysis/experiments/exp2_within_identity/
  data/
    discrimination_ratios.csv
    results.csv
  plots/
    variance_within_vs_across.png
    discrimination_ratio.png
    freq_variance_within_vs_across.png
    mean_freq_mag_{celebrity}.png
  samples/
    {Celebrity}_sample_{0-3}.png
```

---

## Cross-Experiment Analysis: Exp 1 + Exp 2

The combination of Exp 1 and Exp 2 reveals a **dual-channel identity model** in SDXL:

| Role | Channel | Evidence |
|------|---------|----------|
| **Identity magnitude** | Ch 2 | Largest frequency difference when identity changes (Exp 1, 67% of pairs) |
| **Identity fingerprint** | Ch 3 | Best discrimination ratio — stable within identity, varies between (Exp 2) |
| **Scene/composition** | Ch 0 | High variance in both conditions — dominated by non-identity factors |
| **Neutral** | Ch 1 | Low reactivity (Exp 1), moderate discrimination (Exp 2) |

**Next priority:** Exp 3 (Identity Emergence) to understand *when* during denoising Channels 2 and 3 diverge, and Exp 7 (PCA) on Ch 3 specifically to test if identity forms a linear subspace in the most discriminative channel.

---

*All experiments run on Google Colab T4, 2026-03-25. Results archived to Google Drive at identity_analysis/experiments/.*
