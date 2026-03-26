# Research Cycles

Append-only log. Each cycle follows: OBSERVE → EXPLAIN → PREDICT → TEST → REFLECT → DECIDE

---

## Cycle 1: Paired Frequency Analysis (Exp 1)
**Date:** 2026-03-25 | **GPU:** T4 | **Status:** Complete

**Question:** When only identity changes between two prompts, what changes in the frequency domain?
**Prediction:** Identity-related frequency changes should be concentrated in specific channels.
**Result:** Ch 2 showed largest frequency shifts. Ch 0 showed broad changes.
**Surprise:** Ch 2 dominance was later revised — frequency sensitivity ≠ identity importance.
**Decision:** → Run Exp 2 to measure within-identity invariance (different method, same question).

---

## Cycle 2: Within-Identity Invariance (Exp 2)
**Date:** 2026-03-25 | **GPU:** T4 | **Status:** Complete

**Question:** What stays constant when the same person appears in different contexts?
**Prediction:** Some channels should have low within-identity variance and high across-identity variance.
**Result:** Ch 3 had best discrimination ratio (0.046). Ch 0 at 0.022. Ch 1/2 near zero.
**Surprise:** Ch 3, not Ch 2, emerged as the identity fingerprint — contradicting Exp 1's frequency analysis.
**Decision:** → Ch 2 frequency shifts may be noise/style, not identity. Need destructive test (Exp 6) to confirm.

---

## Cycle 3: Identity Emergence (Exp 3)
**Date:** 2026-03-25 | **GPU:** T4 | **Status:** Complete

**Question:** At which denoising step does identity lock in?
**Prediction:** Identity should emerge in mid-to-late steps as the UNet resolves fine details.
**Result:** Ch 3 diverges at steps 7-9, slightly before Ch 0. Identity is a mid-denoising phenomenon.
**Surprise:** Ch 3 leads Ch 0 in emergence — identity fingerprint forms before the foundation settles.
**Decision:** → This timing window (steps 7-9) is where any identity manipulation should target.

---

## Cycle 4: Channel Importance via Zeroing (Exp 6)
**Date:** 2026-03-25 | **GPU:** L4 | **Status:** Complete

**Question:** What happens when each channel is zeroed out?
**Prediction:** If Ch 3 carries identity, zeroing it should damage ArcFace scores significantly.
**Result:** Ch 0 zeroing: catastrophic (0.607). Ch 3: significant (0.769). Ch 1: negligible (0.954). Ch 2: minor (0.894).
**Surprise:** Confirmed Ch 3 > Ch 2 for identity, resolving the Exp 1 vs Exp 2 disagreement.
**Decision:** → Channel hierarchy validated. Now test if we can *transfer* identity via Ch 3 (Exp 5).

---

## Cycle 5: Channel Swap — Direct Decode (Exp 5 V1)
**Date:** 2026-03-25 | **GPU:** L4 | **Status:** Complete (flawed)

**Question:** Can swapping Ch 3 between latents transfer identity?
**Prediction:** Ch 3 swap should show positive ArcFace transfer. Ch 1 swap should show none.
**Result:** Ch 0: +0.367 transfer. Ch 3: +0.153. Ch 1/2: ~0 (negative control works).
**Surprise:** Results looked promising but images were garbled — we decoded intermediate latents directly.
**Decision:** → Method is flawed. Rewrite with callback re-denoising (V2).

---

## Cycle 6: Channel Swap — Re-Denoising (Exp 5 V2)
**Date:** 2026-03-25 | **GPU:** L4 | **Status:** Complete

**Question:** Same as Cycle 5, but with proper re-denoising after swap.
**Prediction:** Expect visible identity shift in swapped images, especially Ch 3.
**Result:** SSIM to target 0.67-0.90 — swapped images look almost identical to unmodified targets. ArcFace failed (all NaN).
**Surprise:** **UNet healing effect** — remaining denoising steps correct the swap. Single-channel manipulation can't overpower text conditioning.
**Decision:** → Channel swapping alone is too crude. Need to find the identity *subspace* within channels. → Run Exp 7 (PCA).

---

## Cycle 7: PCA on Identity (Exp 7)
**Date:** 2026-03-25 | **GPU:** L4 | **Status:** Running

**Question:** Is identity a linear subspace in latent space? Does Ch 3 show better identity clustering than other channels?
**Prediction:** Ch 3 PCA should show highest silhouette score for identity clustering (>0.3). Full latent PCA should also cluster but with more noise from scene/style variation. Ch 1 should show worst clustering (identity-neutral).
**If confirmed:** → Identity is linearly separable in Ch 3. Next: targeted subspace manipulation, project identity onto/off the subspace.
**If rejected:** → Identity is nonlinear or distributed. Try: t-SNE/UMAP, cross-channel PCA, or accept that channel-level manipulation has hit its ceiling.
**Serendipity watch:** PCA components might reveal interpretable axes (e.g., age, gender, expression) as a bonus. Also watch for scene/style clustering — could be useful for style transfer.
**Result:** ALL silhouette scores negative (-0.08 to -0.14). No identity clustering in any channel or full latent.
**Surprise:** Ch 3 showed NO advantage over other channels for PCA clustering. Full latent marginally best (-0.079).
**Decision:** → Linear PCA rejected. Run exploration tests: nonlinear methods, scene-controlled, CLIP embeddings.

---

## Cycle 8: Post-Exp7 Exploration (Tests A/B/C)
**Date:** 2026-03-25 | **GPU:** L4 (RunPod) | **Status:** Complete

**Question:** Where does identity information actually live if not in the VAE latent space?
**Tests run:**
- A: t-SNE/UMAP on latents (nonlinear methods)
- B: Scene-controlled PCA (same prompt, different seeds)
- C: CLIP text embedding analysis

**Results:**
- **Test A:** All negative. UMAP best at -0.15 — worse than PCA. Nonlinear doesn't help.
- **Test B:** Ch3 PCA 20d = -0.071 (marginal improvement over Exp 7's -0.084). Still negative.
- **Test C:** **CLIP mean-pooled embeddings: silhouette +0.403.** Strong positive identity clustering.

**Surprise:** **Identity is linearly separable in CLIP conditioning space but NOT in VAE latent space.** The signal we've been hunting in latent channels was upstream all along — in the text embeddings that drive denoising.

**Serendipity:** CLS token pooling showed 0.0 silhouette (no clustering) while mean pooling showed +0.40. Identity information in CLIP is distributed across token positions, not concentrated in CLS.

**Decision:** → Pivot research direction. The latent space channel analysis taught us HOW identity gets distributed (Ch 0 foundation, Ch 3 fingerprint, UNet healing) but the ACTIONABLE identity signal lives in the conditioning space. Next:
1. Z-Image Turbo cross-architecture validation (running now)
2. CLIP identity direction finding — extract the identity-specific subspace in CLIP embeddings
3. Conditioning manipulation — can we shift identity by moving along CLIP identity directions?

---

## Cycle 9: Z-Image Turbo Cross-Architecture Validation
**Date:** 2026-03-25 | **GPU:** L4 (RunPod) | **Status:** Complete

**Question:** Does Z-Image Turbo (flow-matching DiT, 16ch VAE) show the same patterns as SDXL?
**Prediction:** If VAE latent identity clustering fails on Z-Image too, it confirms the finding is architecture-general, not SDXL-specific. CLIP clustering should still work since both use CLIP conditioning.
**If confirmed:** → Architecture-general finding. Strengthens the case for conditioning-space manipulation.
**If different:** → Z-Image may encode identity differently (different VAE, different denoiser). Could reveal what architectural choices matter for identity encoding.
**Serendipity watch:** Z-Image uses fewer denoising steps (9 vs SDXL 20). Does identity emerge differently in a turbo model?

**Results:**
- **Channel sensitivity:** 3.25x range (0.27→0.89) vs SDXL 1.25x. Channels 1, 6, 9, 12 visually and numerically confirmed as face-focused. Clear tier structure (high/medium/low).
- **Discrimination ratios:** Best ch11=0.117, ch9=0.116 — **2.5x better than SDXL's best** (ch3=0.046). All Z-Image channels outperform SDXL's best.
- **PCA clustering:** ALL negative. Best ch13=-0.137 (10d). **Worse than SDXL** (-0.079). More channels doesn't help linear separability.

**Surprise:** Z-Image's 16 channels show dramatically better sensitivity and discrimination than SDXL — the channels genuinely specialize. But this specialization does NOT translate to linear separability. The discrimination ratios being uniformly distributed (not concentrated in face channels) suggests identity information is still entangled across the full latent. Also: Z-Image's PCA is worse than SDXL's, possibly because 16 channels = higher dimensionality = more noise axes for PCA to get lost in.

**Decision:** → Architecture-general finding CONFIRMED. Identity lives in conditioning, not latent space. However, Z-Image's stronger channel specialization (3.25x vs 1.25x) and 2.5x better discrimination ratios make it a more promising platform for future experiments — there's more signal to work with, even if it's not linearly separable. **Committing to Z-Image Turbo as primary architecture going forward.**

Next directions:
1. Z-Image conditioning space analysis (does it use CLIP? T5? Both?)
2. Multi-channel swap experiments (swap face channels 1+6+9+12 together — may overwhelm DiT healing)
3. Cross-attention analysis in the DiT (different mechanism than UNet cross-attention)

---

## Cycle 11: Qwen3 Conditioning Space Analysis
**Date:** 2026-03-25 | **GPU:** L4 (RunPod) | **Status:** COMPLETE — MAJOR FINDING

**Question:** Is identity linearly separable in Qwen3 LLM embeddings? Z-Image uses Qwen3ForCausalLM (decoder-only, hidden_size=2560, 36 layers) as its text encoder — not CLIP or T5.

**Prediction:** Mean-pooled Qwen3 hidden states will show positive identity silhouette (>0.2). Language models encode celebrity identity regardless of training objective. Last-token pooling may outperform mean pooling (decoder-only models concentrate information later in the sequence).

**Result:** **Name-token pooling: +0.792 silhouette** (layer -6, 5D). Nearly 2x CLIP's +0.403. Mean pooling: +0.024. Last-token: -0.100. Identity is HYPERLOCALIZED in the literal name tokens — not distributed like CLIP.

**Prediction assessment:** Partially correct, partially wrong.
- CORRECT: Identity is separable in Qwen3 — stronger than CLIP
- WRONG: Mean pooling does NOT work (+0.024 vs predicted >0.2) — non-name tokens drown out signal
- WRONG: Last-token pooling is worst, not best — generation prompt token carries no identity
- SURPRISE: Name-token extraction is the key insight. Identity is in the name itself, not the sentence.

**Surprise:** Deeper layers carry MORE identity signal (layer -6 > -5 > ... > -1). The pipeline uses layer -2, which is suboptimal for identity. This depth inversion is unexpected.

**Serendipity:** The Qwen3 chat template with `enable_thinking=True` adds system tokens that are NOT useful for identity. The signal-to-noise ratio is dramatically better when you extract only name tokens.

**Decision:** → This is our headline finding. Priority: token embedding swap experiment to demonstrate practical identity manipulation. See research/FINDINGS.md for full writeup.

---

## Cycle 12: Identity Emergence Timing (Z-Image)
**Date:** 2026-03-25 | **GPU:** L4 (RunPod) | **Status:** Running

**Question:** At which of Z-Image's 9 flow-matching steps does identity lock in?
**Prediction:** Identity emerges at steps 3-5 of 9 (~40-55%), proportional to SDXL's steps 7-9 of 20 (~40%). Face channels (1,6,9,12) emerge before background channels.
**If confirmed:** → Flow-matching and DDPM have similar identity emergence dynamics. Swap timing at step 5 is correct.
**If different:** → Flow-matching has distinct dynamics. May need to adjust swap step.

---

## Cycle 13: Multi-Channel Face Swap (Z-Image)
**Date:** 2026-03-25 | **GPU:** L4 (RunPod) | **Status:** Running

**Question:** Can swapping face channels (1,6,9,12) simultaneously transfer identity, overwhelming the DiT's healing?
**Prediction:** Swapping all 4 face channels (25% of latent) will produce visible identity blending. Progressive: more channels = more transfer. Background channel control (ch 3,4,8,10) should show significantly less identity effect.
**If confirmed:** → Z-Image's channel specialization enables latent-level identity manipulation that wasn't possible on SDXL. Major finding.
**If rejected:** → DiT healing is as aggressive as UNet. Confirms conditioning is the only viable manipulation point.

---

## Cycle 14: Token Embedding Swap
**Date:** 2026-03-26 | **GPU:** L4 (RunPod, via research-lab) | **Status:** COMPLETE

**Question:** Can we transfer identity by surgically replacing name-token embeddings in Qwen3's hidden states?

**Results:**
- Same-gender swap (Brad→Morgan): **0.94 SSIM to target** — nearly perfect identity replacement
- Cross-gender (Taylor→Keanu): 0.70 — produces hybrid, Asian male with Keanu features
- Cross-gender (Morgan→Scarlett): 0.56 — mostly Scarlett, slight Morgan influence
- Scaling (0.5x-2.0x): NO effect — identity is directional, not magnitude-based
- Zeroed/random name tokens: produces **default identity (Asian male)** from DiT's prior

**Surprises:**
1. Identity is binary (direction), not continuous (magnitude). Scaling doesn't work.
2. The model has a built-in default face (Asian male) that emerges when identity signal is removed — comes from DiT weights, not text encoder.
3. Cross-gender swaps fail partially because non-name tokens carry gendered context.

**Decision:** → The default identity mechanism is the most interesting lead. Where exactly in the DiT does it come from? If we can intercept it, we may be able to control identity at the denoiser level. Also: try SLERP interpolation (direction-based) instead of linear scaling, and full subject phrase swap for cross-gender.

See research/FINDINGS_TOKEN_SWAP.md for full analysis.

---

## Cycle 10: Architecture Decision — Z-Image Turbo as Primary
**Date:** 2026-03-25 | **Status:** Decision cycle (no experiment)

**Context:** After completing SDXL phase (Exp 1-7 + exploration) and Z-Image validation (Cycle 9), choosing primary architecture for the next research phase.

**Options considered:**
| | SDXL | Z-Image Turbo |
|---|---|---|
| VAE channels | 4 | 16 |
| Sensitivity range | 1.25x | 3.25x |
| Best discrimination | 0.046 | 0.117 (2.5x better) |
| Denoiser | UNet | DiT (transformer) |
| Steps | 20 | 9 |
| Prior research | Extensively studied | Relatively unexplored |
| Conditioning | CLIP | T5 (to investigate) |
| VRAM | ~12GB | ~21GB (needs offload on L4) |

**Decision: Z-Image Turbo.**

**Rationale:**
1. **Novelty** — Z-Image is less explored in the research community. Findings here are more likely to be novel.
2. **Signal strength** — 2.5x better discrimination ratios mean more signal to work with for manipulation experiments.
3. **Channel specialization** — 16 channels with face-dedicated channels (1,6,9,12) opens multi-channel swap experiments that aren't possible with 4 channels.
4. **DiT architecture** — transformer-based denoiser may have different healing dynamics than UNet. Cross-attention in DiT is structurally different.
5. **T5 conditioning** — if Z-Image uses T5 instead of CLIP, the conditioning space analysis is a fresh investigation (not just replicating our CLIP finding).

**Risks:**
- VRAM: 21GB needs CPU offload on L4 (slow). May need A6000/A40 for production runs.
- Less community tooling/documentation than SDXL.
- bfloat16 requirement limits GPU options (no T4).
