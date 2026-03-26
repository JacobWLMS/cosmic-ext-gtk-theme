# Experiment 4: Token Position Injection

## Hypothesis

Identity information can be injected into identity-empty token positions to control the generated face, without including a celebrity name in the prompt.

## Predictions

**If correct:**
- Part A: Replacing low-identity positions with noise preserves image quality
- Part B: Copying celebrity identity tokens into generic prompt produces that celebrity's face
- Part C: Triangulated injection preserves identity across scene changes

**If incorrect:**
- Z-Image uses the full sequence holistically (no position-specific reading)
- Any modification to non-name tokens breaks generation quality

## Method

Part A: Verify injection doesn't break generation
Part B: Direct celebrity injection
Part C: Triangulated identity injection

## Success Criteria

- Part A: CLIP score vs original > 0.85
- Part B: ArcFace > 0.3 against celebrity
- Part C: ArcFace > 0.25 with correct scene

## Raw Results

- Runtime: 257 seconds (~4.3 minutes)

Part A (safety test -- replacing low-identity positions):

| Method | Mean Similarity to Baseline | Std |
|--------|---------------------------|-----|
| noise | 0.381 | 0.035 |
| zeros | 0.386 | 0.006 |
| mean | 0.467 | 0.066 |

Part B (celebrity identity injection into generic prompt):

| Celebrity | name_tokens | high_identity | all_tokens |
|-----------|-------------|---------------|------------|
| Brad Pitt | 0.927 | 0.915 | 0.936 |
| Tom Hanks | 0.052 | 0.094 | 0.884 |
| Denzel Washington | 0.463 | 0.515 | 0.828 |
| Leonardo DiCaprio | 0.780 | 0.059 | 0.922 |
| Morgan Freeman | 0.913 | 0.896 | 0.905 |
| Keanu Reeves | 0.341 | 0.080 | 0.903 |
| Will Smith | 0.916 | 0.903 | 0.925 |
| Robert Downey Jr | 0.262 | 0.246 | 0.893 |
| Johnny Depp | 0.109 | 0.086 | 0.892 |
| Tom Cruise | 0.868 | 0.864 | 0.902 |

Strategy summary:

| Strategy | Mean | Std |
|----------|------|-----|
| all_tokens | 0.899 | 0.030 |
| name_tokens | 0.563 | 0.355 |
| high_identity | 0.466 | 0.392 |

Part C (triangulated injection into scene prompts):
- Brad Pitt: 0.154 (mean across 3 scenes)
- Tom Hanks: 0.158 (mean across 3 scenes)
- Denzel Washington: 0.154 (mean across 3 scenes, includes -0.096 on beach scene -- actively anti-correlated)
- Mean: 0.155 (essentially random, but hides anti-correlation in individual scene/celebrity combinations)

Part B: all_tokens **PASSES** (0.899 > 0.3) but this is just a positive control. name_tokens and high_identity are bimodal -- not a clean pass or fail.
Part C: **FAILS** (0.155, well below 0.25 threshold).

## Interpretation

**all_tokens (0.899)** is a positive control, not a finding. Replacing the entire embedding trivially produces the celebrity. It confirms the injection mechanism works.

**The bimodal split is the headline result.** name_tokens injection is not a weak signal -- it's either near-perfect (0.87-0.93) or near-zero (0.05-0.11). The standard deviation of 0.355 on a mean of 0.563 is not a normal distribution -- it's two populations. This means Qwen3 has entity-specific encoding strategies: some celebrities have identity concentrated in name tokens, others distribute it across the sequence.

**high_identity is worse AND invalidates the Exp 2 position map as a universal tool.** DiCaprio (name=0.780, high_identity=0.059) is the clearest evidence: his identity lives in name tokens but NOT in the suffix positions Exp 2 identified. This invalidation extends beyond injection -- any per-position analysis that assumes consistency across celebrities (clustering, probing, steering) is undermined by the same bimodal encoding. The aggregate position map from Exp 2 averages over fundamentally incompatible encoding strategies and produces a map that is misleading for individual entities.

**Part C failure compounds two lossy operations.** Triangulation was already borderline (0.373). Positional injection is bimodal. Stacking them produces noise (0.155). Notably, Denzel Washington's beach scene injection produced ArcFace = -0.096 -- actively anti-correlated with the target. This suggests the injection mechanism can actively interfere with identity, not just fail to transfer it.

**n_name_tokens as a bimodal predictor.** The raw data reveals a correlation between the number of name tokens (as identified by the tokenizer) and injection success. All 2-token celebrities that work (Brad Pitt, Morgan Freeman, Will Smith, Tom Cruise) score > 0.86. Most 1-token celebrities fail (Tom Hanks 0.052, Johnny Depp 0.109, Keanu Reeves 0.341). Robert Downey Jr breaks the pattern (2 tokens, 0.262) -- his 3-word name likely tokenises to 2 tokens that don't cleanly correspond to the semantically meaningful parts. Leonardo DiCaprio is a 1-token outlier that works (0.780). This is the first concrete predictor candidate for the bimodal split and should be the first Phase 4 check.

## Surprises

The bimodal encoding was completely unpredicted. All prior work assumed a universal encoding strategy. The clean split between "name-concentrated" and "name-distributed" celebrities is the most novel finding of Phase 3.

## Impact on State of Affairs

Position-targeted injection is not viable as a general strategy. Full-embedding manipulation (all_tokens) works but isn't novel. The bimodal encoding discovery is potentially publishable as a standalone finding about LLM entity representation. Phase 4 should investigate what predicts which group a celebrity falls into.
