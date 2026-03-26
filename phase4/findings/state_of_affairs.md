# State of Affairs -- Phase 4

*Updated 2026-03-26 after Experiments 1-2.*

## The Definitive Picture

Identity encoding in Qwen3-4B is now well-characterised:

1. **Identity lives in name tokens, universally.** Exp 1 proved this with 99 celebrities: mean ArcFace 0.837 (std 0.055) when injecting all name subword tokens. 97/99 above 0.7. No bimodality. The fix was purely in the token matching function.

2. **Silhouette score is the wrong metric at scale.** The multi-layer sweep (Exp 2) shows silhouette never exceeds 0.286 at 99 celebrities, at any layer, with any pooling. But Exp 1 proves the identity signal is strong (0.837 ArcFace). Silhouette measures cluster geometry in high-dimensional cosine space, which is geometrically unfriendly to 99 clusters even when the underlying signal is present. **Do not use silhouette as a proxy for identity encoding quality.**

3. **Phase 2's +0.792 is retired.** The replication attempt got 0.362 — not close. The discrepancy is fully explained: Phase 2 used the same buggy token matching (partial tokens for Scarlett Johansson and Keanu Reeves), and 5 clusters separate easily even with noisy vectors. The +0.792 was an artefact of small n + buggy matching, not a real benchmark.

4. **Deeper layers are slightly better for clustering, but not dramatically.** name_mean silhouette: 0.239 at layer -2, 0.286 at layer -10. The improvement is modest. name_mean consistently outperforms all_mean at every layer. Full dimensionality (2560d) consistently outperforms PCA reduction at 99-celebrity scale.

## What Still Stands from Earlier Phases

- Identity is NOT separable in VAE latent space (Phase 2, unchanged)
- Suffix/EOS tokens have high discrimination ratios due to causal attention (Phase 3 Exp 2, unchanged)
- Triangulation works borderline at layer -2 (Phase 3 Exp 3: 0.373 mean, unchanged)
- Low-identity positions are safe manipulation targets (Phase 3 Exp 4 Part A, unchanged)
- Scene injection failed at layer -2 with partial tokens (Phase 3 Exp 4 Part C: 0.155)
- Celebrity DB: 99 valid, mean consistency 0.72
- Dev Patel outlier explained by Riz Ahmed near-duplicate (Phase 3 Peer Review)

## What Was Retracted or Retired

- **"Bimodal identity encoding"** — RETRACTED in Phase 3.5, CONFIRMED retracted in Phase 4 Exp 1
- **Phase 2's +0.792 silhouette benchmark** — RETIRED in Phase 4 Exp 2. Artefact of buggy matching + small n.
- **Silhouette as identity metric** — inadequate at 99-celebrity scale. Use ArcFace injection scores instead.

## Phase 4 Remaining Priorities

1. **Exp 3: Layer -6/-10 triangulation** — does deeper layer improve the 0.373 mean?
2. **Exp 4: Scene injection retry** — with fixed tokens + best layer, does scene injection work?
3. **Exp 5: Expanded injection** — confirm Exp 1's result at 30+ celebrities with bootstrap CIs
4. **Exp 6: Nearest-neighbour density** — how many celebrities needed for viable triangulation?
