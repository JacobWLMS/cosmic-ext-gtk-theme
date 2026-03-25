# Research Log: Identity Signal Analysis in Diffusion Model Latent Spaces

## Experiment 6: Channel Importance Analysis

**Date:** 2025-03-25
**Status:** Complete
**Runtime:** T4 GPU, ~20 min total

### Question

Which VAE channels carry the most identity information? What happens to perceived identity when individual channels are zeroed out?

### Setup

| Parameter | Value |
|-----------|-------|
| Model | SDXL Base 1.0 (fp16) |
| Resolution | 1024x1024 |
| Steps | 12 |

**Part 1: Face vs Non-Face Region**
| Parameter | Value |
|-----------|-------|
| Prompt pairs | 6 (from standard set) |
| Seeds | 5 per pair |
| Face detection | InsightFace buffalo_l (CPU) |

**Part 2: Channel Zeroing**
| Parameter | Value |
|-----------|-------|
| Celebrities | Brad Pitt, Taylor Swift, Morgan Freeman |
| Seeds | 3 per celebrity |
| Metric | ArcFace cosine similarity to original |

### Results

#### Part 1: Face vs Non-Face Region Differences

Face detection rate: 80% (24/30 pairs).

The face/non-face analysis produced NaN values for face region data due to a batching issue with the face region mask computation. The non-face region data was captured correctly. This part needs a rerun, but Part 2 (channel zeroing) produced the key findings.

#### Part 2: Channel Zeroing — The Key Result

Each channel was individually zeroed out, the latent decoded back to an image, and ArcFace similarity measured against the original image:

| Channel Zeroed | ArcFace Similarity | Identity Preserved | Interpretation |
|---------------|-------------------|-------------------|----------------|
| **Ch 0** | **0.607** | Yes (barely) | **Most damaging** — foundational image structure destroyed |
| **Ch 3** | **0.769** | Yes | **Second most damaging** — discriminative identity impacted |
| Ch 2 | 0.894 | Yes | Minor impact — fine details lost |
| Ch 1 | 0.954 | Yes | Negligible impact — style/texture channel |

#### Critical Revision: Ch 0 is NOT "Scene Only"

Previous experiments (Exp 1-3) characterized Ch 0 as the "scene/composition" channel because:
- It had the highest within-identity variance (Exp 2: 0.726)
- It saturated earliest during denoising (Exp 3: step 5-6)
- It had the worst discrimination ratio (Exp 2: 0.022)

**Exp 6 reveals this was incomplete.** Ch 0 is the **foundation channel** — it carries the broadest, most fundamental image structure including:
- Basic face shape and proportions
- Overall composition and layout
- Scene/background structure
- Low-frequency lighting and color

When zeroed, Ch 0 produces the most identity damage (ArcFace drops to 0.607) because the entire image foundation collapses. It has a poor discrimination ratio not because it lacks identity information, but because it carries *everything* — identity is inseparable from the rest.

#### Two Types of Identity Information

| Type | Channel | Can be isolated? | Manipulation target? |
|------|---------|-----------------|---------------------|
| **Structural identity** | Ch 0 | No — embedded in foundation | No — zeroing destroys everything |
| **Discriminative identity** | Ch 3 | Yes — best discrimination ratio | **Yes — surgical target** |
| **Identity detail** | Ch 2 | Partially — high variance | Maybe — noisy but informative |
| **Non-identity** | Ch 1 | N/A | N/A — zeroing has negligible effect |

### Conclusions

1. **Ch 0 is the most identity-critical channel by raw impact** — zeroing it drops ArcFace similarity to 0.607 (worst). But this identity information is inseparable from scene/composition.

2. **Ch 3 is the best target for identity manipulation** — it's the second-most damaging when zeroed (0.769) AND has the best discrimination ratio (0.046). This means it carries identity information that can be separated from context.

3. **Ch 2 is surprisingly unimportant for perceptual identity** — zeroing it barely affects ArcFace (0.894). The fine frequency-domain details it carries (Exp 1) don't significantly impact how a face is recognized. This resolves the Exp 1 vs Exp 2 tension: Ch 2's high reactivity is noise, not signal.

4. **Ch 1 is identity-neutral** — zeroing it has almost no effect (0.954). Safe to modify for style transfer without affecting identity.

5. **The previous "dual-channel" model was an oversimplification.** Identity is hierarchical: Ch 0 (foundation) > Ch 3 (discriminative) > Ch 2 (detail) > Ch 1 (none). For practical manipulation, Ch 3 is the only viable target.

### Visual Evidence

The zeroed channel images (saved as samples) visually confirm:
- Ch 0 zeroed: image severely degraded, face barely recognizable
- Ch 3 zeroed: face structure intact but identity noticeably altered
- Ch 2 zeroed: face looks almost the same, subtle differences
- Ch 1 zeroed: virtually identical to original

### Implications

- **Exp 5 (Frequency Injection):** Should exclusively target Ch 3 for identity transfer. Ch 0 is too dangerous, Ch 2 is irrelevant.
- **Exp 7 (PCA):** Run PCA on Ch 3 only — it's the channel where identity forms a potentially separable subspace.
- **For identity-preserving style transfer:** Modify Ch 1 and Ch 2 freely while keeping Ch 0 and Ch 3 fixed.
- **For identity swap:** Replace Ch 3 at step 7-9 (when it locks in, per Exp 3) from a different generation.

### Open Questions

1. What exactly does Ch 0 encode that makes it so identity-critical? Is it literally face geometry in low-frequency latent space?
2. Can we do partial Ch 3 modification (scale rather than zero) to achieve subtle identity shifts?
3. Does the channel hierarchy hold for Lumina2's 16-channel VAE? More channels might mean cleaner separation.
4. The face/non-face region analysis needs a rerun — the face region mask had NaN issues.

### Plots

Saved to Google Drive at `identity_analysis/experiments/exp6_channel_importance/`.

- `plots/face_vs_nonface_importance.png` — Face vs non-face region difference per channel
- `plots/zeroing_arcface_impact.png` — ArcFace similarity after zeroing each channel
- `plots/pair{N}_diff_heatmap.png` — Spatial difference heatmaps per prompt pair
- `samples/{Celebrity}_original.png` — Original generated images
- `samples/{Celebrity}_zeroed_ch{N}.png` — Images with each channel zeroed

---

*Experiment run on Google Colab T4, 2026-03-25. Results archived to Google Drive at identity_analysis/experiments/exp6_channel_importance/.*
