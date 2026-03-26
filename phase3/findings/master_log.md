# Phase 3: Token-Space Identity Injection -- Master Log

## Timeline

| Date | Experiment | Key Finding | Impact |
|------|-----------|-------------|--------|
| 2026-03-26 | Celebrity DB | 99/100 celebrities validated (ArcFace consistency > 0.5). Only John Boyega failed (0.449). 13 celebs/min on Ada 6000. | Database ready for all experiments |
| 2026-03-26 | Exp 1: Qwen3 Clustering | Best silhouette +0.071 (all_mean, full dim, layer -2). **METHODOLOGICAL ISSUE**: Phase 2 used layer -6 which scored +0.792. Layer choice is primary variable, not identity encoding capability. | Exp 1 result is INCONCLUSIVE, not a failure. Layer -6 re-run required in Phase 4. |
| 2026-03-26 | Exp 2: Token Positions | Suffix/EOS tokens have highest discrimination ratio (2.57), beating name tokens (1.82). Silhouette scores near zero everywhere due to cosine distance geometry. 3 high-identity positions identified: [7, 14, 17]. | Causal LLMs compress identity into sequence-final positions. New architectural finding. |
| 2026-03-26 | Exp 3: Triangulation | Part A: high_identity_only best at 0.373 mean ArcFace (threshold 0.4). Dev Patel outlier at 0.725. Part B: 0.236 mean (random faces), Simu Liu nearest-neighbour collapse. | Triangulation works for known celebrities at layer -2. Layer -6 likely pushes over threshold. Open-set needs denser DB. |
| 2026-03-26 | Exp 4: Injection | **BIMODAL IDENTITY ENCODING**: name_tokens injection is 0.927 for Brad Pitt but 0.052 for Tom Hanks. all_tokens always works (0.899 mean). high_identity positions from Exp 2 do NOT generalise. Part C scene injection failed (0.155). | Identity encoding is celebrity-specific. Some faces live in name tokens, others are distributed. Fixed-position injection strategy is not viable. |

## Current State of Affairs

See state_of_affairs.md for full synthesis.

## Resolved Questions

1. **Is identity separable in Qwen3 space?** -- INCONCLUSIVE at layer -2. Phase 2 showed +0.792 at layer -6. Layer choice is the dominant variable. Requires Phase 4 multi-layer sweep.
2. **Which token positions carry identity?** -- PARTIALLY ANSWERED. Suffix/EOS tokens have highest discrimination ratios (causal attention summary effect). But identity encoding is celebrity-specific -- no universal position map exists.
3. **Can triangulation represent unknown faces?** -- YES for held-out celebrities (0.373 mean at layer -2, 0.725 peak). NO for truly unknown faces with sparse celebrity DB (0.236 mean).
4. **Can we inject identity into empty positions?** -- MIXED. Injecting all tokens works perfectly (0.899). Name-token injection is bimodal -- works brilliantly for some celebrities, fails completely for others. Position-targeted injection using Exp 2's map does not generalise.

## Open Questions (for Phase 4)

1. Does layer -6 push triangulation over the 0.4 threshold?
2. Can we replicate Phase 2's +0.792 silhouette at layer -6 with 99 celebrities? (Direct replication check)
3. What predicts whether a celebrity's identity lives in name tokens vs distributed? (Tokenizer subword splitting? Training frequency? Face distinctiveness?)
4. Can a denser celebrity database (500+) fix open-set triangulation?
5. Why does Part C scene injection fail when Part B direct injection succeeds?
