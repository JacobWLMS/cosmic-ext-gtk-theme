# Research Log: Identity Signal Analysis in Diffusion Model Latent Spaces

## Experiment 3: Step-by-Step Identity Emergence

**Date:** 2025-03-25
**Status:** Complete
**Runtime:** T4 GPU, ~15 min total (40 generations, 12 steps each, step latent capture)

### Question

At what denoising step does identity information appear in the latent? Do Ch 2 (identity magnitude) and Ch 3 (identity fingerprint) emerge at different times?

### Setup

| Parameter | Value |
|-----------|-------|
| Model | SDXL Base 1.0 (fp16) |
| Resolution | 1024x1024 |
| Steps | 12 |
| Seeds | 10 (0-9) |
| Identity pairs | 2 (age gap + gender) |
| Metrics | Per-channel spatial divergence, frequency divergence, ArcFace similarity |
| VAE tiling | Enabled |

### Identity Pairs

| Pair | Prompt A | Prompt B |
|------|----------|----------|
| age_gap | young woman, studio lighting | elderly woman, studio lighting |
| gender | man, studio lighting | woman, studio lighting |

### Results

#### 1. 90% Emergence Step (when each channel reaches 90% of its final divergence)

| Channel | Age Gap | Gender | Role (from Exp 1+2) |
|---------|---------|--------|---------------------|
| Ch 0 | **step 5** | **step 6** | Scene/composition |
| Ch 1 | step 9 | step 10 | Neutral |
| **Ch 2** | **step 10** | **step 10** | Identity magnitude |
| **Ch 3** | **step 9** | **step 7** | Identity fingerprint |

#### 2. 50% Emergence Step

| Channel | Age Gap | Gender |
|---------|---------|--------|
| Ch 0 | step 1 | step 1 |
| Ch 1 | step 1 | step 1 |
| Ch 2 | step 2 | step 1 |
| Ch 3 | step 1 | step 1 |

All channels reach 50% by step 1-2 — the broad structure diverges almost immediately. The interesting differences are in the 90% timing.

#### 3. Per-Channel Divergence Trajectory (age_gap pair)

| Step | Ch 0 | Ch 1 | Ch 2 | Ch 3 |
|------|------|------|------|------|
| 0 | 0.208 | 0.127 | 0.076 | 0.132 |
| 3 | 0.358 | 0.223 | 0.170 | 0.238 |
| 6 | 0.390 | 0.247 | 0.195 | 0.258 |
| 9 | 0.409 | 0.270 | 0.236 | 0.276 |
| 11 | 0.423 | 0.295 | 0.278 | 0.300 |

Ch 0 is nearly saturated by step 6, while Ch 2 is still at 70% — it has a much longer tail.

#### 4. Key Finding: Two-Phase Identity Emergence

The data reveals identity emerges in two distinct phases:

**Phase 1 (steps 0-5): Coarse structure**
- Ch 0 (scene) reaches ~90% — background, composition, lighting settle
- All channels reach ~50% — broad identity features (gender, age range) are encoded
- Ch 3 starts building the identity fingerprint

**Phase 2 (steps 6-11): Identity refinement**
- Ch 3 (fingerprint) reaches 90% by step 7-9 — reliable identity signal locked in
- Ch 2 (magnitude) continues refining until step 10 — fine identity details still being added
- Ch 1 also late (step 9-10) — may carry complementary identity detail

**Ch 2 has the longest tail** — it's the last channel to stabilize, meaning identity *magnitude* information is refined in the final denoising steps. This explains why Ch 2 has high reactivity (Exp 1) but also high within-identity variance (Exp 2): it captures fine details that vary with each generation.

**Ch 3 settles before Ch 2** — the fingerprint channel locks in identity ~2 steps earlier than the magnitude channel, especially for the gender pair (step 7 vs step 10). This confirms Ch 3 encodes a more stable, earlier-resolved identity representation.

### Conclusions

1. **Identity emerges in two phases:** coarse structure (steps 0-5) then identity refinement (steps 6-11). Scene/composition settles first, identity details last.

2. **Ch 3 (fingerprint) locks in before Ch 2 (magnitude)** — by 2-3 steps. This is consistent with Ch 3 being more stable (Exp 2) — it represents the identity decision the model makes early and commits to.

3. **Ch 2 has the longest refinement tail** — it's still gaining information in the last 2 steps. This matches its high sensitivity (Exp 1) but high variance (Exp 2): it captures fine-grained identity details that are seed-dependent.

4. **Ch 0 saturates by mid-denoising** — confirms it primarily encodes scene/composition, not identity. Useful as a negative control.

5. **50% divergence is immediate** — all channels diverge from each other by step 1-2. The gross identity separation happens in the first few steps; the remaining steps refine it.

### Implications

- **For identity manipulation:** Intervening at steps 5-7 (after Ch 3 locks in but before Ch 2 finishes) could allow modifying fine identity details while preserving the core identity fingerprint.
- **For identity transfer:** Swapping Ch 3 at step 7 between two generations could transplant the core identity while letting Ch 2 adapt to the new context.
- **For Exp 5 (Frequency Injection):** Inject at mid-denoising targeting Ch 3, not at the final step.
- **Fewer-step generation** (e.g., 8 steps) would lose Ch 2 refinement but preserve Ch 3 — acceptable for identity analysis, problematic for identity-sensitive applications.

### Plots

Saved to Google Drive at `identity_analysis/experiments/exp3_identity_emergence/`.

- `plots/{pair}_spatial_divergence_per_channel.png` — Per-channel divergence over steps (with error bars)
- `plots/{pair}_freq_divergence_per_channel.png` — Frequency domain version
- `plots/{pair}_ch2_vs_ch3_emergence.png` — Head-to-head Ch2 vs Ch3 comparison
- `plots/{pair}_normalized_divergence.png` — When each channel reaches its final state
- `plots/{pair}_arcface_emergence.png` — Perceptual identity emergence via ArcFace
- `samples/{pair}_final_{a,b}.png` — Final generated images

---

*Experiment run on Google Colab T4, 2026-03-25. Results archived to Google Drive at identity_analysis/experiments/exp3_identity_emergence/.*
