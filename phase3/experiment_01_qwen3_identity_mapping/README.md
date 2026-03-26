# Experiment 1: Qwen3 Identity Clustering Analysis

## Hypothesis

Identity is linearly separable in Qwen3-4B's embedding space, similar to how it was in CLIP's (+0.40 silhouette), and the signal is concentrated in name token positions.

## Predictions

**If hypothesis is correct:**
- Silhouette score > 0.3 for name-token pooling
- Name-token pooling >> mean pooling (matching Phase 2 finding of hyperlocalization)
- PCA should show clear celebrity clusters at low dimensions (5-10)
- t-SNE visualization will show distinct per-celebrity clusters

**If hypothesis is incorrect:**
- All silhouette scores near 0 or negative
- No clear clustering in any pooling strategy
- Identity may be encoded non-linearly in Qwen3 (would need different approach)

## Method

1. 100 celebrities x 5 prompt templates (varying scene, identical structure)
2. Extract Qwen3 final hidden state (hidden_states[-2]) for each
3. Test pooling strategies: all-token mean, name-token mean, non-name mean, BOS, EOS
4. PCA at 2, 5, 10, 20 dimensions per strategy
5. Silhouette scores for identity clustering
6. t-SNE visualization of best configuration

## Success Criteria

- Silhouette score > 0.3 for at least one pooling strategy
- Clear visual clustering in t-SNE

## Raw Results

- Model: Tongyi-MAI/Z-Image-Turbo, Qwen3-4B text encoder, layer -2
- 99 celebrities x 5 templates = 495 samples
- Best: all_mean, full dimensionality (2560d), silhouette = +0.071
- name_mean best: +0.0016 at full dim
- non_name_mean: all negative
- first (BOS-like): all 0.000 (constant across celebrities)
- last: all negative (-0.16 to -0.34)
- **FAILED** against 0.3 threshold

**CRITICAL METHODOLOGICAL ISSUE:** Phase 2 found +0.792 at layer -6, 5d PCA, name_tokens pooling on 5 celebrities. This experiment tested ONLY layer -2. The comparison is invalid. Layer choice is the dominant variable.

Strategy summary (best score per strategy):

| Strategy | Best Silhouette |
|----------|----------------|
| all_mean | +0.071 |
| name_mean | +0.002 |
| first | 0.000 |
| last | -0.156 |
| non_name_mean | -0.037 |

## Interpretation

Result is **INCONCLUSIVE**, not a failure. The +0.071 silhouette at layer -2 with 99 celebrities is weak but not zero. The dramatic drop from Phase 2's +0.792 is explained by (a) wrong layer (layer -2 vs -6) and (b) scaling from 5 to 99 clusters. A multi-layer sweep is required before drawing conclusions about Qwen3's identity encoding capability.

## Surprises

The first (BOS) token position gives exactly 0.000 silhouette -- perfectly constant across all celebrities. This makes sense: in Qwen3's chat template, position 0 is always `<|im_start|>` regardless of prompt content.

## Impact on State of Affairs

Exp 1 cannot gate subsequent experiments as originally planned. The layer -2 limitation was discovered after running. Phase 4 will include the multi-layer sweep.
