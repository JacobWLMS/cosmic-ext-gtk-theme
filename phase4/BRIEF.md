# Phase 4: Validated Identity Injection & Multi-Layer Analysis for Z-Image Turbo

## Context

This continues the identity signal research on Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo). Read these files BEFORE starting any work:

- `phase3.5/findings/state_of_affairs.md` — the most current synthesis (supersedes phase3's)
- `phase3/PEER_REVIEW.md` — peer review with raw data findings and charting guidance
- `phase3.5/experiment_01_tokenisation_analysis/README.md` — the tokenisation analysis that retracted Phase 3's headline finding
- `phase3/findings/rejected_hypotheses.md` — what's been falsified
- `research/CYCLES.md` — Cycle 14 (token swap) has critical findings about identity being directional not magnitude-based

### What still stands from Phase 3

- Identity IS separable in Qwen3 text conditioning space (Phase 2: +0.792 at layer -6, 5 celebs)
- Exp 1 at layer -2 with 99 celebs got +0.071 — **inconclusive**, not a failure. Layer choice is the dominant variable.
- Suffix/EOS tokens have highest discrimination ratios (2.57) due to causal attention summary effect
- Triangulation works borderline at layer -2 (0.373 mean ArcFace for held-out celebs)
- Low-identity positions tolerate noise/zeros/mean replacement (safe manipulation targets)
- Scene injection failed at layer -2 (0.155 mean) — scene tokens overwhelm injected identity
- all_tokens injection works universally (0.899 mean) — confirms identity is always present
- Celebrity DB: 99 valid, mean consistency 0.72, saved as pickle
- Dev Patel outlier (0.725) explained: nearest neighbour Riz Ahmed at 0.859 similarity (near-duplicate)

### What was RETRACTED

- **"Bimodal identity encoding" is RETRACTED.** Phase 3.5 proved it was a tokenisation artefact in `find_name_token_positions`. The function used substring matching that missed subword-split name tokens. Celebrities with clean 2-token names (Brad Pitt → `Brad`, `Pitt`) had full injection; celebrities with subword splits (Tom Hanks → `Tom`, `H`, `anks`) had partial injection. Identity encoding is consistent across celebrities.

### Key technical facts

- Qwen3-4B: hidden_size=2560, 36 layers, decoder-only with causal attention
- Pipeline uses hidden_states[-2] by default. Phase 2 found layer -6 optimal for identity.
- Chat template: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- Typical prompt = ~19 tokens. No BOS token. EOS=151645, PAD=151643.
- 8 steps (9 NFE), guidance_scale=0.0 (no CFG)
- Identity is directional, not magnitude-based (Cycle 14: scaling 0.5x-2.0x had NO effect)
- Model has a built-in default face (Asian male) when identity signal is removed — comes from DiT weights
- prompt_embeds is a LIST of variable-length tensors (padding stripped)
- Pipeline accepts pre-computed prompt_embeds — confirmed working for injection

## Hardware

RTX 6000 Ada (48GB VRAM). No CPU offloading needed (~22GB usage). Use bfloat16 throughout. Generate at 512×512 for analysis, 1024×1024 only for final showcase.

## Project Structure

```
phase4/
  shared/
    pipeline.py           # Inherit from phase3, or import directly
    scoring.py            # ArcFace cosine similarity, CLIP similarity
    qwen3_utils.py        # FIXED find_name_token_positions + all phase3 utils
    visualisation.py      # Import from phase3/shared/visualisation.py (already has seaborn toolkit)
    celebrity_db.py       # Load phase3 pickle, extend as needed
    config.py             # Constants, prompt templates

  experiment_01_fix_token_matching/
    README.md
    run.py
    results/

  experiment_02_multi_layer_sweep/
    README.md
    run.py
    results/

  experiment_03_layer6_triangulation/
    README.md
    run.py
    results/

  experiment_04_scene_injection_retry/
    README.md
    run.py
    results/

  experiment_05_expanded_injection/
    README.md
    run.py
    results/

  experiment_06_nearest_neighbour_analysis/
    README.md
    run.py
    results/

  findings/
    master_log.md
    state_of_affairs.md
    serendipity.md
    rejected_hypotheses.md
```

## Documentation Requirements (STRICT — same as Phase 3)

Every experiment README.md MUST contain these sections BEFORE the experiment is run:
- Hypothesis
- Predictions (what we expect if correct AND incorrect)
- Method
- Success criteria (specific numerical thresholds)

After running, add:
- Raw results (tables, exact numbers)
- Plots (saved to results/plots/)
- Interpretation
- Surprises
- Impact on state of affairs

Every numerical result saved as CSV or JSON. Every plot saved to disk. Every generated image saved. The master_log.md updated BEFORE moving to the next experiment.

Sample sizes: minimum 50 seeds for any statistical claim. Use --quick flag (5 seeds) for pipeline testing only.

## Visualisation Requirements

Use the Phase 3 visualisation toolkit (`phase3/shared/visualisation.py`). It already has seaborn-based plotting with the correct style. Follow the charting guidance from `phase3/PEER_REVIEW.md`:

- **Strip plot with violin overlay** for any small-n comparison (n < 50). NEVER use box plots for potentially bimodal data. Always show individual data points.
- **Paired line plots / slope charts** for within-subject strategy comparisons
- **Annotated position heatmaps** for per-token-position analysis
- **Scatter with regression** for correlation analysis (include Pearson r, p-value, label each point)
- **Always include threshold lines** (dashed) when success criteria exist
- **Label outliers by name** when a data point is > 2 std from the mean
- Save ALL plots to `results/plots/` within each experiment directory

## Experiments

### Experiment 1: Fix Token Matching & Validate (PRIORITY — DO THIS FIRST)

**Question**: Does fixing `find_name_token_positions` to capture all subword pieces move the "fails" group into the "works" group?

**Method**:
1. Rewrite `find_name_token_positions` in `qwen3_utils.py` to identify ALL subword tokens belonging to a celebrity name, not just tokens matching by substring. Use the tokeniser's offset mapping or character-to-token alignment.
2. Verify the fix: for all 10 Exp 4 celebrities, print the old matched positions vs new matched positions. Every celebrity should now have 0 missed tokens.
3. Re-run Exp 4 Part B (celebrity injection) with the fixed matching — inject ALL name subword tokens.
4. Score with ArcFace against celebrity references.
5. Run on all 99 celebrities, not just the original 10 — this gives us the full picture.

**Success criteria**:
- All 10 original celebrities score > 0.7 with the fixed matching
- Mean across all 99 celebrities > 0.8
- Standard deviation drops below 0.10 (no more bimodal spread)

**What we learn**: Whether identity truly lives in name tokens for ALL celebrities (Phase 2 claim, validated by Phase 3.5's tokenisation analysis) or whether subword-split names distribute identity differently even when all pieces are captured.

**Critical**: This gates Experiments 4 and 5. If fixing the token matching doesn't work, the injection strategy needs rethinking.

### Experiment 2: Multi-Layer Identity Clustering Sweep

**Question**: At which Qwen3 layer is identity most separable, and does the Phase 2 finding (+0.792 at layer -6) replicate at 99 celebrities?

**Method**:
1. Using the celebrity DB (99 celebrities × 5 prompt templates = 495 samples)
2. Extract hidden states at layers [-2, -4, -6, -8, -10, -12] (wider sweep than originally planned — cheap to do)
3. For each layer, compute silhouette scores using:
   - all_mean pooling at PCA dims [2, 5, 10, 20, full]
   - name_mean pooling (using FIXED token matching) at PCA dims [2, 5, 10, 20, full]
4. **Direct replication check**: also run layer -6 on a 5-celebrity subset (same 5 from Phase 2 if possible) to validate methodology
5. t-SNE/UMAP visualisation of the best configuration

**Success criteria**:
- Layer -6 name_mean at 5d: silhouette > 0.3 (the original Phase 3 threshold)
- 5-celebrity replication: silhouette > 0.6 (close to Phase 2's +0.792)
- Clear peak in silhouette vs layer curve (identity is localised to specific layers)

**What we learn**: The definitive answer on Qwen3 identity separability at scale. If layer -6 at 99 celebrities is still low, the 5→99 scaling is a real finding about Qwen3's embedding capacity, not a methodology issue.

### Experiment 3: Layer -6 Triangulation

**Question**: Does triangulation clear the 0.4 ArcFace threshold at layer -6?

**Prerequisites**: Experiment 2 must show layer -6 has better silhouette than layer -2.

**Method**:
1. Re-run Exp 3 Part A (20 held-out celebrities, 79 lookup, K=5 nearest neighbours)
2. Extract embeddings at layer -6 instead of layer -2
3. Test all three strategies: all_positions, name_tokens_only (with fixed matching), high_identity_only
4. Also test a NEW strategy: name_tokens_all_subwords (all subword pieces of name, not just the ones the old function matched) — this should outperform the old name_tokens_only
5. Compute Pearson/Spearman correlation between nearest_sim and triangulation score

**Success criteria**:
- Mean ArcFace > 0.4 for at least one strategy
- name_tokens_all_subwords outperforms old name_tokens_only
- Nearest-neighbour correlation r > 0.5

**What we learn**: Whether the layer-2 triangulation results were underperforming due to the wrong layer, or whether 0.37 is close to the method's ceiling.

### Experiment 4: Scene Injection Retry (with fixed tokens + layer -6)

**Question**: Does scene injection work when (a) ALL name subword tokens are injected and (b) embeddings are from layer -6?

**Prerequisites**: Experiment 1 must show fixed token matching works. Experiment 2 must show layer -6 is better.

**Method**:
1. For 10 celebrities (mix of previously-working and previously-failing):
   - Take the celebrity's full embedding at layer -6
   - Take a scene prompt embedding ("person walking on a beach", "person cooking in a kitchen", "person in a library")
   - Inject ALL name subword tokens from the celebrity into the corresponding positions of the scene prompt
   - Generate 5 images per combination
   - Score with ArcFace against celebrity reference AND CLIP against the scene description
2. Compare against Phase 3 Exp 4 Part C (which scored 0.155 at layer -2 with partial tokens)
3. Test a new strategy: inject name tokens + suffix tokens together (motivated by Exp 2's finding that suffix tokens carry concentrated identity)

**Success criteria**:
- ArcFace > 0.25 against target celebrity
- CLIP text-image alignment > 0.20 against scene description
- Improvement over Phase 3's 0.155 baseline

**What we learn**: Whether scene injection failure was caused by (a) wrong layer, (b) missing tokens, (c) both, or (d) fundamental incompatibility. This is the most speculative experiment — it might still fail.

### Experiment 5: Expanded Injection Validation (30+ celebrities)

**Question**: Does the fixed name-token injection work consistently across a larger sample?

**Prerequisites**: Experiment 1 must succeed (all original 10 score > 0.7).

**Method**:
1. Select 30 celebrities spanning: high consistency (>0.8), medium (0.65-0.8), varied tokenisation (1-token names, 2-token, 3-token, subword-split)
2. Run name-token injection (fixed matching) at layer -2 AND layer -6
3. Score with ArcFace
4. Analyse: does consistency predict injection quality? Does tokenisation complexity matter beyond the matching fix?
5. Bootstrap 95% confidence intervals on all means

**Success criteria**:
- Mean > 0.8 at layer -2 with fixed matching (confirms tokenisation was the only issue)
- Mean > 0.85 at layer -6
- Std < 0.10 (no bimodality)
- No systematic failures for subword-split names

**What we learn**: Whether the fix generalises beyond the original 10 celebrities. This is the validation experiment that turns Phase 3.5's hypothesis into a confirmed finding.

### Experiment 6: Nearest-Neighbour Density Analysis

**Question**: How does celebrity database density affect triangulation quality, and how many celebrities do we actually need?

**Method**:
1. Using the Exp 3 data (Phase 3 or Phase 4's layer -6 version), compute:
   - Pearson/Spearman correlation between nearest_sim and triangulation score
   - Regression: predict triangulation score from nearest_sim, 2nd-nearest_sim, mean top-5 sim
2. Simulate denser databases:
   - Subsample the 79-celebrity lookup to 20, 40, 60 celebrities
   - At each size, measure mean triangulation score for the 20 held-out celebrities
   - Extrapolate: at what database size does the mean clear 0.4?
3. Diversity analysis:
   - For the Part B random face failure: generate 20 diverse random faces (specify gender, age, ethnicity in prompts)
   - Re-run triangulation with these diverse faces
   - Check if the Simu Liu convergence disappears

**Success criteria**:
- Significant correlation (p < 0.01) between nearest_sim and triangulation score
- Clear scaling curve showing diminishing returns
- Diverse random faces produce diverse nearest neighbours (not all Simu Liu)

**What we learn**: Whether the approach needs 200, 500, or 1000+ celebrities to be practical. This determines whether the triangulation pipeline is viable as a real tool.

## Experiment Priority & Dependencies

```
Exp 1 (fix tokens) ──────────┬──→ Exp 4 (scene retry)
                              ├──→ Exp 5 (expanded validation)
Exp 2 (multi-layer sweep) ───┼──→ Exp 3 (layer-6 triangulation)
                              └──→ Exp 6 (density analysis)
```

1. **Exp 1** (fix token matching) — gates Exp 4 and 5. Do first. No GPU needed for the fix itself, only for re-running injection.
2. **Exp 2** (multi-layer sweep) — can run in parallel with Exp 1. Gates Exp 3.
3. **Exp 3** (layer-6 triangulation) — run after Exp 2 confirms layer -6 is better.
4. **Exp 5** (expanded injection) — run after Exp 1 confirms the fix works.
5. **Exp 4** (scene injection retry) — run after both Exp 1 and Exp 2. Most speculative.
6. **Exp 6** (density analysis) — can run any time after Exp 3 or using Phase 3 data.

If Exp 1 fails (fixing tokens doesn't move all celebrities above 0.7), STOP and reassess. Document the negative result. It would mean identity distribution varies per celebrity for reasons beyond tokenisation — possibly training frequency, face distinctiveness, or something about how Qwen3 represents rare vs common entities.

If Exp 2 shows no improvement at layer -6 vs layer -2 at 99 celebrities, the Phase 2 result (+0.792) was an artefact of small sample size (5 celebrities). Document this as a finding about silhouette score sensitivity to cluster count.

## What NOT to do

- Don't assume Phase 3.5's tokenisation fix will definitely work — TEST it
- Don't skip the direct replication check in Exp 2 (5-celebrity at layer -6)
- Don't run Exp 4 (scene injection) before Exp 1 confirms the token fix works
- Don't generalise findings from Z-Image Turbo to SDXL or other models
- Don't use the old `find_name_token_positions` — always use the fixed version
- Don't report means without confidence intervals
- Don't use box plots — use strip plots with violin overlay per the charting guidance
