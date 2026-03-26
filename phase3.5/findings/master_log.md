# Phase 3.5: Tokenisation Analysis -- Master Log

## Timeline

| Date | Experiment | Key Finding | Impact |
|------|-----------|-------------|--------|
| 2026-03-26 | Exp 1: Tokenisation | Phase 3's "bimodal identity encoding" is a tokenisation artefact. `find_name_token_positions` misses subword-split surnames. All "works" celebrities have 0 missed tokens; all "fails" have 2-3 missed. | **RETRACTS** bimodal encoding as a finding about Qwen3. Validates identity-in-name-tokens hypothesis. Shifts Phase 4 priority to fixing token matching. |

## Status

Phase 3.5 is a single targeted analysis that resolves the most important open question from Phase 3. No GPU compute was needed -- pure tokeniser analysis.

## Impact on Phase 4 Priorities (revised)

1. ~~Investigate bimodal split predictors~~ -- **RESOLVED**: it's a tokenisation bug
2. **Fix `find_name_token_positions`** and re-run Exp 4 Part B to confirm all celebrities score > 0.7
3. **Multi-layer sweep** (layers -2, -4, -6) on 99 celebrities -- unaffected by this finding
4. **Layer -6 triangulation** -- re-run Exp 3 Part A at layer -6
5. **Re-evaluate scene injection** (Exp 4 Part C) with ALL name tokens injected
