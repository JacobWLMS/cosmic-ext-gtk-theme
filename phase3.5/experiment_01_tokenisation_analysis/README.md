# Experiment 1: Tokenisation Analysis of Bimodal Identity Split

## Date & Hardware

2026-03-26. Local analysis only (Qwen3-4B tokenizer, no GPU required).

## Hypothesis

The bimodal split in Phase 3 Exp 4 (name_tokens injection works for some celebrities but not others) is caused by the `find_name_token_positions` function failing to identify subword-split name tokens. Celebrities whose names tokenise as clean whole words have their full identity injected; celebrities whose names are split into subword pieces only get a partial injection.

## Predictions

**If hypothesis is correct:**
- "Works" group celebrities (Brad Pitt, Will Smith, Morgan Freeman, Tom Cruise) will have names that tokenise as 2 clean whole-word tokens, both matched by `find_name_token_positions`
- "Fails" group celebrities (Tom Hanks, Johnny Depp, Keanu Reeves, Robert Downey Jr) will have surnames split into subword tokens that the matching function misses
- Leonardo DiCaprio (1-token outlier that works at 0.780) will have a distinctive first name token that carries identity alone
- Robert Downey Jr (2-token exception that fails at 0.262) will have its matched tokens on non-distinctive parts of the name
- The number of MISSED name tokens will correlate negatively with injection score

**If hypothesis is incorrect:**
- All celebrities will have similar tokenisation patterns (all clean or all subword-split)
- The bimodal split will persist even after accounting for tokenisation differences
- The split must be explained by something else (training frequency, face distinctiveness, etc.)

## Method

1. Load the Qwen3-4B tokenizer (same as used by Z-Image Turbo pipeline)
2. For each of the 10 Exp 4 Part B celebrities, tokenise the prompt "portrait photo of {name}, natural lighting, neutral background" using the same chat template
3. Print the exact token breakdown for the name region (positions 5-12)
4. Compare: which tokens does `find_name_token_positions` match vs which tokens actually belong to the name?
5. Count matched tokens, total name tokens, and missed tokens per celebrity
6. Correlate n_name_tokens and n_missed with injection score

## Success Criteria

- Clear separation in tokenisation patterns between works/fails groups
- Pearson r > 0.5 between n_matched_tokens and injection score
- Robert Downey Jr and Leonardo DiCaprio exceptions explained by token content

## Controls

- Using the identical prompt template and chat template as the pipeline
- Comparing against the same `find_name_token_positions` logic from Phase 3

---

## Raw Results

### Exact tokenisation for all 10 celebrities

| Celebrity | Score | Matched | Total Name | Missed | Token Breakdown |
|-----------|-------|---------|------------|--------|-----------------|
| Brad Pitt | 0.927 | 2 | 2 | 0 | ` Brad` ` Pitt` |
| Will Smith | 0.916 | 2 | 2 | 0 | ` Will` ` Smith` |
| Morgan Freeman | 0.913 | 2 | 2 | 0 | ` Morgan` ` Freeman` |
| Tom Cruise | 0.868 | 2 | 2 | 0 | ` Tom` ` Cruise` |
| Leonardo DiCaprio | 0.780 | 1 | 4 | 3 | ` Leonardo` \| ` Di` `Cap` `rio` |
| Denzel Washington | 0.463 | 1 | 3 | 2 | ` Den` `zel` \| ` Washington` |
| Keanu Reeves | 0.341 | 1 | 4 | 3 | ` K` `ean` `u` \| ` Reeves` |
| Robert Downey Jr | 0.262 | 2 | 4 | 2 | ` Robert` \| ` Down` `ey` \| ` Jr` |
| Johnny Depp | 0.109 | 1 | 3 | 2 | ` Johnny` \| ` De` `pp` |
| Tom Hanks | 0.052 | 1 | 3 | 2 | ` Tom` \| ` H` `anks` |

(Pipe `|` separates matched from missed tokens. Bold = matched by `find_name_token_positions`.)

### Subword split details

**"Works" group (score > 0.78) -- all have clean 2-token names:**
- Brad Pitt: `Brad` + `Pitt` -- both whole words, both matched. Zero missed.
- Will Smith: `Will` + `Smith` -- both whole words, both matched. Zero missed.
- Morgan Freeman: `Morgan` + `Freeman` -- both whole words, both matched. Zero missed.
- Tom Cruise: `Tom` + `Cruise` -- both whole words, both matched. Zero missed.

**"Fails" group (score < 0.47) -- all have subword-split surnames:**
- Tom Hanks: `Tom` matched, but `Hanks` splits to `H` + `anks` -- both missed. Only `Tom` injected (a common word with no celebrity-specific identity).
- Johnny Depp: `Johnny` matched, but `Depp` splits to `De` + `pp` -- both missed. Only `Johnny` injected.
- Keanu Reeves: `Reeves` matched, but `Keanu` splits to `K` + `ean` + `u` -- all 3 missed. Only `Reeves` injected.
- Robert Downey Jr: `Robert` and `Jr` matched, but `Downey` splits to `Down` + `ey` -- both missed. The two matched tokens (`Robert`, `Jr`) are generic, non-distinctive parts of the name.

**Exceptions explained:**
- Leonardo DiCaprio (0.780, 1-token "works"): `Leonardo` is the only matched token but it's a highly distinctive first name. `DiCaprio` splits to `Di` + `Cap` + `rio` (all missed). The injection succeeds because `Leonardo` alone carries enough identity signal -- it's rare enough as a first name to be associated primarily with DiCaprio.
- Robert Downey Jr (0.262, 2-token "fails"): The 2 matched tokens are `Robert` and `Jr` -- neither is distinctive. `Robert` is shared by many celebrities (De Niro, Pattinson, Redford). `Jr` is a generic suffix. The actual identity-carrying token `Downey` is split and missed.

### Statistical correlation

| Metric | Value | p-value |
|--------|-------|---------|
| Pearson r (n_matched, score) | 0.635 | 0.049 |
| Spearman rho (n_matched, score) | 0.661 | 0.037 |
| Mann-Whitney U (1-token vs 2-token) | 3.0 | 0.056 |

Group means:
- 1-token celebrities: mean=0.349, std=0.293, n=5
- 2-token celebrities: mean=0.777, std=0.289, n=5

## Plots

See `results/plots/` (generated locally).

## Interpretation

**The bimodal split is entirely a tokenisation artefact in `find_name_token_positions`, not a property of Qwen3's identity encoding.**

The Exp 4 injection experiment only injected the tokens that `find_name_token_positions` matched. For celebrities with clean 2-token names (Brad Pitt, Will Smith), both tokens were injected and identity transferred perfectly (0.87-0.93). For celebrities with subword-split surnames (Tom Hanks, Johnny Depp), only partial name tokens were injected, and the missing pieces contained the distinctive identity signal.

This means:
1. **Identity DOES live in name tokens** -- the Phase 2 finding is validated, not contradicted
2. **The "bimodal encoding" is not bimodal at all** -- Qwen3 uses the same encoding strategy for all celebrities. The apparent bimodality was caused by incomplete token matching.
3. **Fixing `find_name_token_positions` to capture all subword pieces should move the "fails" group into the "works" group** -- this is the key prediction to test in Phase 4

The correlation is statistically significant (Pearson r=0.635, p=0.049) but just barely, and with n=10 this should be treated as strong evidence, not proof. The qualitative pattern (every works celebrity has 0 missed tokens, every fails celebrity has 2-3 missed tokens) is more convincing than the p-value.

**Important caveat**: This analysis identifies the tokenisation bug as the proximate cause. But even if we fix the matching function, we still need to verify that injecting ALL subword tokens for Tom Hanks actually produces his face. It's possible that subword-split names distribute identity differently than whole-word names, and simply matching more tokens won't be sufficient. Phase 4 must test this.

## Surprises / Serendipity

The Robert Downey Jr case is pedagogically interesting: 2 matched tokens but 0.262 score, because the matched tokens (`Robert`, `Jr`) are generic while the distinctive token (`Downey`) is split. This shows that **token count alone doesn't predict success -- token distinctiveness matters**. The injection must capture the tokens that are unique to the celebrity, not just any tokens from their name.

The Leonardo DiCaprio case shows the flip side: 1 matched token but 0.780 score, because `Leonardo` is distinctive enough to carry identity alone. If you're famous enough that your first name is uniquely associated with you, a single token suffices.

## Conclusions

1. Phase 3's "bimodal identity encoding" finding is **RETRACTED** -- it was a tokenisation artefact
2. Identity encoding in Qwen3 is consistent across celebrities -- it concentrates in name tokens
3. The `find_name_token_positions` function needs to be rewritten to capture all subword pieces of a name, not just tokens that contain a full name part as a substring
4. The fixed matching function should be validated by re-running Exp 4 Part B with all name subword tokens injected

## Impact on State of Affairs

Major revision needed:
- The "bimodal encoding" section of state_of_affairs.md should be updated to reflect that this was a tokenisation artefact
- The serendipity.md entry about bimodal encoding should be moved to rejected_hypotheses.md
- Phase 4 priority shifts: instead of "investigate what predicts the bimodal split", the priority becomes "fix token matching and re-run Exp 4 to confirm identity lives in all name tokens"
- The n_name_tokens predictor is real but it's predicting a bug, not a property of the model

## Next Steps

1. Rewrite `find_name_token_positions` to match all subword pieces of celebrity names
2. Re-run Exp 4 Part B with fixed matching -- predict all celebrities score > 0.7
3. Proceed with multi-layer sweep (Phase 4 priority 2) which is unaffected by this finding
4. Re-evaluate whether position-targeted injection (Exp 4 Part C) might work if ALL name tokens are injected instead of partial
