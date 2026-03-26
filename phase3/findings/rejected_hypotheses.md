# Rejected Hypotheses -- Phase 3

## 1. "A universal token position map can identify identity-carrying vs identity-empty positions"

**Hypothesis**: Exp 2 can identify fixed positions that carry identity for all celebrities, enabling targeted injection.

**Evidence against**: Exp 2 identified positions [7, 14, 17] as high-identity using aggregate discrimination ratios across 99 celebrities. Exp 4 tested injecting these positions (high_identity strategy) and found:
- Brad Pitt: 0.915 (works)
- Morgan Freeman: 0.896 (works)
- Leonardo DiCaprio: 0.059 (complete failure)
- Tom Hanks: 0.094 (complete failure)
- Keanu Reeves: 0.080 (complete failure)

**Conclusion**: Identity encoding is celebrity-specific. Aggregate per-position statistics average over fundamentally different encoding strategies and produce a map that works for some entities and fails for others. A fixed position map is not a viable injection strategy.

## 2. "Identity can be injected into scene prompts via position-targeted token replacement"

**Hypothesis**: By injecting identity tokens (from triangulation or direct celebrity embedding) into specific positions of a scene prompt, we can control identity independently of scene description.

**Evidence against**: Exp 4 Part C scored 0.155 mean ArcFace -- essentially random. Three celebrities tested across three scene prompts, all near zero. The scene prompt's token structure overwhelms any injected identity signal at the targeted positions.

**Possible explanations** (untested):
- Scene tokens actively interfere with foreign identity signals at layer -2
- The cap_embedder projection (2560 -> 3840) doesn't preserve position-local identity information
- Identity may need to be encoded holistically across the full sequence, not injected at specific positions
- Layer -2 may be too close to the output head, where representations are optimised for next-token prediction rather than entity encoding

**Note**: This does NOT rule out all forms of identity injection. all_tokens replacement works at 0.899 mean. The failure is specific to position-targeted partial injection, not to embedding manipulation in general.

## 3. "Exp 1 silhouette scores prove identity is not separable in Qwen3"

**Status**: Rejected as premature conclusion, NOT as rejected hypothesis.

**The problem**: Exp 1 tested only layer -2 and got silhouette +0.071. Phase 2 tested layers -1 through -6 and found +0.792 at layer -6 (name_tokens, 5d PCA, 5 celebrities). The comparison is invalid because:
- Different layers (dominant variable)
- Different celebrity counts (5 vs 99, affects cluster density)

Exp 1's result is INCONCLUSIVE about Qwen3's identity encoding capability. It tells us layer -2 is poor for identity clustering at 99-celebrity scale. It does not tell us that Qwen3 can't encode identity -- Phase 2 already showed it can at layer -6.

**Required follow-up**: Multi-layer sweep at 99 celebrities (Phase 4 priority 1).
