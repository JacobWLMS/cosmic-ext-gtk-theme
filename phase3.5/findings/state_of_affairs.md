# State of Affairs -- Phase 3.5

*Updated 2026-03-26 after tokenisation analysis.*

## Key Revision from Phase 3

**Phase 3's headline finding -- "bimodal identity encoding in Qwen3" -- is RETRACTED.** The apparent bimodality was caused by a bug in `find_name_token_positions` which uses substring matching (`part in token`) to identify name tokens. This works for names that tokenise as clean whole words (Brad Pitt -> `Brad`, `Pitt`) but fails for names with subword-split components (Tom Hanks -> `Tom`, `H`, `anks` -- only `Tom` matched).

The Exp 4 injection experiment only injected the matched tokens. For celebrities with complete matches (0 missed tokens), injection scored 0.87-0.93. For celebrities with missed subword pieces (2-3 missed tokens), injection scored 0.05-0.34. This is not a property of Qwen3's encoding -- it's an artefact of incomplete token selection.

## What We Now Know

1. **Identity encoding in Qwen3 is consistent across celebrities** -- it concentrates in name tokens, as Phase 2 originally found (+0.792 silhouette at layer -6)
2. **The tokeniser is the variable, not the model** -- Qwen3's BPE tokeniser splits uncommon names into subwords, and the matching function missed these pieces
3. **Token distinctiveness matters** -- `Leonardo` alone carries identity (DiCaprio scores 0.780 from one token) because it's a rare first name. `Tom` alone doesn't (Hanks scores 0.052) because it's shared across many entities. `Robert` + `Jr` together fail (Downey Jr scores 0.262) because neither is distinctive.
4. **All other Phase 3 findings stand** -- layer choice, suffix token discrimination, triangulation results, safe manipulation targets are all unaffected by the tokenisation bug

## Remaining Open Questions

1. Does fixing the token matching and re-running Exp 4 Part B move all celebrities above 0.7?
2. Does scene injection (Exp 4 Part C) work when ALL name subwords are injected?
3. Multi-layer sweep results (layer -2 vs -4 vs -6 at 99 celebrities)
4. Layer -6 triangulation -- does it clear 0.4?
