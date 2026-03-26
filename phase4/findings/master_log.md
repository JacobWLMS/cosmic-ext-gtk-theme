# Phase 4: Validated Identity Injection & Multi-Layer Analysis -- Master Log

## Timeline

| Date | Experiment | Key Finding | Impact |
|------|-----------|-------------|--------|
| 2026-03-26 | Exp 1: Fix Token Matching | **All 10 original celebrities above 0.81.** 99-celebrity mean = 0.837 (std 0.055), 97/99 above 0.7. Phase 3.5 retraction fully validated. | Identity lives in name tokens universally. The "bimodal encoding" was a bug. |
| 2026-03-26 | Exp 2: Multi-Layer Sweep | Best silhouette 0.286 at layer -10 (name_mean, full dim). Never crosses 0.3 at any layer. 5-celeb replication: 0.362 vs Phase 2's 0.792. | Phase 2's +0.792 RETIRED as benchmark — artefact of partial token matching + small n. Silhouette is wrong metric at 99-celebrity scale. |

## Resolved Questions

1. **Does fixing token matching move all celebrities above 0.7?** YES. Min was 0.696 (Awkwafina), 97/99 above 0.7. Mean 0.837.
2. **Does Phase 2's +0.792 replicate?** NO. The discrepancy is explained: Phase 2 used buggy token matching (same bug as Phase 3), and 5 clusters is easy for silhouette. The +0.792 was an artefact.
3. **Is silhouette the right metric for identity separability?** NO at 99-celebrity scale. Exp 1 proves identity IS in the tokens (0.837 ArcFace), but silhouette in 2560d with 99 clusters doesn't capture this. The geometry is unfriendly to silhouette even when the signal is present.

## Open Questions

1. Does triangulation improve at deeper layers (layer -10 or -6)?
2. Does scene injection work with fixed token matching + optimal layer?
3. How does database density affect triangulation quality?
