# Experiment 2: Per-Token-Position Identity Analysis

## Hypothesis

Specific Qwen3 token positions carry strong identity signal while others are identity-empty, and this pattern is predictable from the token type (name vs scene vs special tokens).

## Predictions

**If correct:**
- Name token positions: high discrimination ratio, silhouette > 0.5
- Scene description positions: low discrimination ratio, silhouette near 0
- Special tokens (BOS/EOS): unknown — key finding
- Positions before name tokens: near 0 (causal attention = can't see future)
- Positions after name tokens: moderate signal (can attend to name)

**If incorrect:**
- Identity signal uniformly distributed across all positions
- Would suggest Qwen3 diffuses identity through attention, making position-targeted injection impossible

## Method

1. 100 celebrities x 5 prompts (identical template, only name varies)
2. For each token position, compute:
   - Within-identity variance
   - Between-identity variance
   - Discrimination ratio (between/within)
   - Single-position silhouette score
3. Plot discrimination ratio and silhouette vs position
4. Group analysis: name tokens, scene tokens, special tokens, adjacent tokens

## Success Criteria

- Clear separation between high-identity and low-identity positions
- At least some non-name positions with silhouette > 0.2 (bleed)
- At least some positions with silhouette near 0.0 (injection targets)

## Raw Results

- 99 celebrities, template 0 only ("portrait photo of {name}, natural lighting, neutral background")
- Sequence lengths: min=18, max=22, analysis at min=18 positions
- Multi-template silhouette (5 samples per celebrity, 495 total) still near zero everywhere
- Discrimination ratio is the informative metric:

By token type:

| Type | Mean Disc Ratio | Max Disc Ratio | Mean Silhouette | Count |
|------|----------------|---------------|-----------------|-------|
| name | 1.193 | 1.818 | -0.160 | 7 |
| post_name | 1.166 | 1.535 | -0.229 | 2 |
| scene | 1.459 | 1.459 | -0.227 | 1 |
| suffix | 1.851 | 2.566 | -0.318 | 2 |
| system_prefix | 0.000 | 0.000 | -0.080 | 5 |
| pre_name | 0.000 | 0.000 | 0.012 | 1 |

High-identity positions (sil > 0.2 OR disc > 1.5): [7, 14, 17]
Low-identity positions (|sil| < 0.05 AND disc < 0.5): 4 positions

## Interpretation

Suffix/EOS tokens have the HIGHEST discrimination ratios (2.57), exceeding name tokens (1.82). This is an architectural consequence of causal attention -- final tokens attend to everything and become identity summaries. However, Exp 4 later showed these positions DON'T generalise as injection targets (works for some celebrities, fails for others). The aggregate statistics hide celebrity-specific encoding strategies.

Silhouette scores are near zero or negative because cosine distance in 2560d space with 99 classes doesn't form tight clusters even when discrimination ratio is high. The metrics tell different stories: discrimination ratio measures variance structure (useful), silhouette measures cluster geometry (uninformative here).

## Surprises

System prefix tokens (pos 0-4: `<|im_start|>`, `user`, `\n`, `portrait`, ` photo`) have exactly 0.0 discrimination ratio -- completely invariant to identity changes. This is expected but confirms the template structure.

Pre_name token (pos 5: ` of`) has 0.0 discrimination ratio -- it cannot see the name tokens due to causal masking. This is a direct confirmation of the causal attention architecture.

## Impact on State of Affairs

The position map from this experiment was used in Exp 3 and 4 but proved unreliable for position-targeted injection. The finding about suffix tokens as identity summaries is architecturally interesting but not actionable for injection.
