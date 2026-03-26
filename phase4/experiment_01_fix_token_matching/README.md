# Experiment 1: Fix Token Matching & Validate

## Date & Hardware

2026-03-26. RTX 6000 Ada 48GB, Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo), Qwen3-4B layer -2, bfloat16.

## Hypothesis

The Phase 3 Exp 4 "bimodal" injection results were entirely caused by incomplete token matching in find_name_token_positions. Fixing the function to capture ALL subword tokens of a celebrity name will move previously-failing celebrities (Tom Hanks 0.052, Johnny Depp 0.109, etc.) into the working range (>0.7 ArcFace).

## Predictions

**If correct:**

- All 10 original Exp 4 celebrities score > 0.7 with fixed matching
- Tom Hanks, Johnny Depp, Keanu Reeves all jump from <0.35 to >0.7
- Robert Downey Jr jumps from 0.262 to >0.7 (the missed "Downey" tokens now included)
- Mean across 99 celebrities > 0.8 with std < 0.10
- The "bimodal" distribution becomes unimodal, centered around 0.85-0.90

**If incorrect:**

- Some celebrities remain below 0.5 even with all name tokens matched
- Subword-split names distribute identity differently from whole-word names
- The fix is necessary but not sufficient -- other factors (training frequency, name ambiguity) matter
- This would mean Phase 3.5's retraction was premature

## Method

1. Rewrite find_name_token_positions to use character-span alignment: tokenize the full prompt, then find which token positions overlap with the character span of the celebrity name in the original prompt string. This captures ALL subword tokens regardless of how the BPE tokenizer splits the name.
2. Verify: print old vs new matched positions for all 10 Phase 3 celebrities. Every celebrity should now have 0 missed tokens.
3. Re-run injection on all 10 original celebrities with fixed matching (5 seeds each).
4. Then run on all 99 celebrities in the database (5 seeds each).
5. Score with ArcFace against celebrity reference images.

## Success Criteria

- All 10 original celebrities > 0.7
- Mean across 99 celebrities > 0.8
- Std < 0.10

## Controls

- All 10 original celebrities serve as their own controls (Phase 3 scores are the baseline)
- The "works" group (Brad Pitt, Will Smith, etc.) should score similarly to Phase 3 (they already had complete matching)
- all_tokens injection from Phase 3 (0.899 mean) is the theoretical ceiling
