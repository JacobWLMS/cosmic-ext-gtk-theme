# Serendipity Log -- Phase 3

## 1. ~~Bimodal Identity Encoding in Qwen3~~ RETRACTED (Exp 4, 2026-03-26)

**RETRACTED in Phase 3.5.** The apparent bimodality was a tokenisation artefact -- `find_name_token_positions` missed subword-split name tokens. See phase3.5/experiment_01_tokenisation_analysis/README.md for full analysis.

Original observation (now explained): When only name-token embeddings are injected into a generic prompt, ArcFace scores split into two distinct populations:

**Name-concentrated group** (identity fully recoverable from name tokens alone):
- Brad Pitt: 0.927
- Will Smith: 0.916
- Morgan Freeman: 0.913
- Tom Cruise: 0.868
- Leonardo DiCaprio: 0.780

**Name-distributed group** (name tokens carry near-zero identity):
- Tom Hanks: 0.052
- Johnny Depp: 0.109
- Robert Downey Jr: 0.262
- Keanu Reeves: 0.341

Denzel Washington (0.463) sits in between, suggesting a continuum rather than strict binary, but the distribution is clearly bimodal.

all_tokens injection works for ALL celebrities (0.899 mean, 0.030 std), confirming identity is always encoded -- the question is where within the sequence.

**Why this matters**: This is a finding about how large language models represent named entities, independent of the diffusion model application. It suggests Qwen3 has entity-specific encoding strategies, not a universal template. This may be publishable as a standalone observation about LLM entity representations.

**Untested hypotheses for what causes the split**:
- Tokenizer effects (subword splitting, common-word tokens like "Tom")
- Training frequency (more common celebrities = more concentrated representation)
- Name ambiguity ("Tom" appears in many celebrity names; "Brad Pitt" is unique)
- But Tom Cruise works fine (0.868), so name ambiguity alone doesn't explain it

## 2. Suffix Tokens as Identity Summaries (Exp 2, 2026-03-26)

In a causal decoder-only model, the final sequence tokens (`<|im_end|>`, newline, `<|im_start|>assistant`) have the highest discrimination ratios for identity (up to 2.57), exceeding the actual name tokens (max 1.82). This is an architectural consequence: these positions attend to everything in the sequence and act as summaries.

**However**: Exp 4 showed this doesn't mean the DiT reads identity exclusively from these positions. The suffix positions work for some celebrities but not others (DiCaprio: name_tokens=0.780, high_identity=0.059). The aggregate statistics from Exp 2 hide celebrity-specific encoding strategies.

## 3. Dev Patel Triangulation Outlier -- EXPLAINED (Exp 3, 2026-03-26)

Dev Patel scored 0.725 ArcFace for high_identity_only triangulation, nearly 3 standard deviations above the 0.373 mean.

**Resolution**: The raw CSV reveals his nearest neighbour is **Riz Ahmed at 0.859 ArcFace similarity** -- that's near-duplicate territory. For comparison, the next-highest nearest-neighbour similarity is Samuel L. Jackson -> Trevor Noah at 0.655. Most are in the 0.3-0.45 range.

This means the 0.725 score is NOT showing the method's ceiling -- it's showing that interpolation trivially succeeds when the lookup table contains someone who looks almost identical. This is an artefact of database composition, not evidence of exceptional method performance. Dev Patel should arguably be excluded from aggregate statistics, or at minimum flagged as an explained outlier driven by Riz Ahmed's presence in the lookup set.
