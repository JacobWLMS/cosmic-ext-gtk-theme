# Experiment 2: Multi-Layer Identity Clustering Sweep

## Date & Hardware

2026-03-26. RTX 6000 Ada 48GB, Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo), Qwen3-4B, bfloat16.

## Hypothesis

Identity separability in Qwen3 increases at deeper layers (further from the output head), peaking around layer -6 as Phase 2 found. The Phase 3 Exp 1 result (+0.071 at layer -2) was primarily a layer choice issue, not a genuine lack of identity encoding.

## Predictions

**If correct:**

- Silhouette scores increase monotonically from layer -2 to layer -6 (or -8)
- Layer -6 name_mean at 5d PCA: silhouette > 0.3 (the original Phase 3 threshold)
- 5-celebrity replication at layer -6: silhouette > 0.6 (close to Phase 2's +0.792)
- Clear peak in the layer curve -- identity is localized to specific layers
- name_mean consistently outperforms all_mean across layers

**If incorrect:**

- All layers score similarly low (< 0.15) at 99 celebrities
- The 5-celebrity replication fails too (< 0.4) -- suggesting Phase 2's +0.792 was an artefact of small n
- No clear peak -- identity information is distributed across layers
- This would mean Qwen3's identity encoding doesn't separate cleanly at 99-celebrity scale regardless of layer

## Method

1. Using celebrity DB (99 celebrities x 5 templates = 495 samples)
2. Extract hidden states at layers [-2, -4, -6, -8, -10, -12]
3. For each layer, compute silhouette using name_mean (FIXED matching) and all_mean pooling at PCA dims [2, 5, 10, 20, full]
4. Direct replication: run layer -6 on 5-celebrity subset (Brad Pitt, Taylor Swift, Morgan Freeman, Scarlett Johansson, Keanu Reeves -- same as Phase 2)
5. t-SNE of best configuration

## Success Criteria

- Layer -6 name_mean 5d: silhouette > 0.3
- 5-celebrity replication: silhouette > 0.6
- Clear peak in layer curve

## Controls

- Layer -2 results should approximately match Phase 3 Exp 1 (+0.071)
- 5-celebrity results at layer -6 should approximately match Phase 2 (+0.792)
- all_mean should consistently underperform name_mean (Phase 2 finding)
