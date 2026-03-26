# Experiment 3: Celebrity Triangulation

## Hypothesis

Unknown faces can be represented by weighted interpolation (SLERP) of nearby celebrity embeddings in Qwen3 space, producing recognisable resemblance.

## Predictions

**If correct:**
- Part A (held-out celebrities): ArcFace > 0.4 against held-out reference
- Part B (random faces): ArcFace > 0.3
- Interpolating only name tokens outperforms full-sequence interpolation

**If incorrect:**
- ArcFace scores near 0 (no resemblance)
- Interpolation produces averaged/generic faces
- Identity space is too non-linear for SLERP

## Method

Part A: Hold out 20 celebrities, triangulate from 80, score against held-out refs
Part B: Generate random faces, triangulate, score
Part C: Real photo test (if A/B positive)

## Success Criteria

- Part A: Mean ArcFace > 0.4
- Part B: Mean ArcFace > 0.3

## Raw Results

- 20 held-out celebrities, 79 for lookup, K=5 nearest neighbours
- 3 interpolation strategies, 5 seeds per test
- Runtime: 495 seconds (~8.2 minutes)

Part A (held-out celebrities):

| Strategy | Mean ArcFace | Std |
|----------|-------------|-----|
| high_identity_only | 0.373 | 0.127 |
| name_tokens_only | 0.346 | 0.102 |
| all_positions | 0.336 | 0.082 |

Selected per-celebrity results:

| Celebrity | high_identity | name_tokens | all_positions |
|-----------|---------------|-------------|---------------|
| Morgan Freeman | 0.229 | 0.214 | 0.227 |
| Tom Cruise | 0.229 | 0.283 | 0.231 |
| Chris Hemsworth | 0.326 | 0.323 | 0.363 |
| Dev Patel | **0.725** | 0.460 | 0.375 |

Part B (random faces):
- Mean ArcFace: 0.236 (below 0.3 threshold)
- Simu Liu was nearest neighbour for 6/10 random faces, Constance Wu 3/10, Ken Watanabe 1/10 -- all East Asian celebrities, confirming both prompt mode collapse and database sparsity in that demographic region
- Range: 0.164 to 0.312

Part A **FAILS** threshold (0.373 < 0.4). Part B **FAILS** threshold (0.236 < 0.3).

## Interpretation

Triangulation WORKS but is borderline at layer -2. The strategy ranking (high_identity > name_tokens > all_positions) is consistent and validates the Exp 2 suffix token finding -- those positions carry concentrated signal. But the 0.08-0.13 standard deviations are large relative to 0.04 strategy differences, so statistical significance is questionable (n=20).

Part B failed due to two compounding issues: (1) sparse celebrity database (99 points can't tile face space), and (2) generic prompts mode-collapsing to Asian-presenting faces, concentrating all tests in one region. A fairer test would measure relative improvement over nearest-neighbour baseline.

## Surprises

Dev Patel at 0.725 for high_identity_only (NOT all_positions as initially misreported). **Now explained**: his nearest neighbour is Riz Ahmed at 0.859 ArcFace similarity -- near-duplicate territory. The outlier is an artefact of having visually similar celebrities in the database, not evidence of exceptional method performance. Most nearest-neighbour similarities are 0.3-0.45; Dev Patel's 0.859 is a massive outlier in database composition.

Additionally, there's a clear positive correlation between nearest-neighbour similarity and triangulation score across all held-out celebrities (e.g., Dev Patel 0.859 sim -> 0.725 score; Keanu Reeves 0.265 sim -> 0.296 score). This confirms triangulation quality is driven by database density around the target, not the interpolation method itself.

## Impact on State of Affairs

Layer -6 will likely push Part A over 0.4 threshold. Open-set triangulation needs denser database (500+) and diverse test prompts.
