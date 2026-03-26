# State of Affairs -- Phase 3

*Updated 2026-03-26 after Experiments 1-4.*

## What We Know (from Phase 2)

- Identity is NOT separable in VAE latent space (SDXL: -0.079, Z-Image: -0.137 silhouette)
- Identity IS separable in text conditioning space (CLIP mean-pool: +0.403 silhouette)
- Qwen3 name tokens show +0.792 silhouette at **layer -6, PCA 5d, 5 celebrities** -- hyperlocalized identity
- Token interpolation and latent SLERP produce identical blends -- identity is token-driven
- CLS pooling carries ZERO identity signal; mean pooling carries ALL (CLIP-specific finding)
- Z-Image Turbo: 8 steps (9 NFE), guidance_scale=0.0, Qwen3-4B text encoder
- Z-Image has dormant SigLIP pathway (active in Edit/Omni-Base, absent in Turbo)

## What Phase 3 Found

### 1. Layer Choice Dominates Identity Separability

Phase 2's +0.792 silhouette was at layer -6. Phase 3 Exp 1 tested only layer -2 and got +0.071. Within Phase 2's own data, layer -2 scored +0.742 at 5d (name_tokens) vs layer -6 at +0.792. The layer -2 to layer -6 gap is real but modest within Phase 2 (5 celebrities). The dramatic drop from +0.742 to +0.071 when scaling from 5 to 99 celebrities suggests the identity subspace at layer -2 is crowded -- 99 clusters in the same dimensionality don't separate as cleanly as 5.

**Status**: Inconclusive. Multi-layer sweep at 99 celebrities required (Phase 4).

### 2. Causal LLMs Compress Identity into Sequence-Final Positions

Exp 2 found that suffix/EOS tokens (`<|im_end|>`, `\n`, `<|im_start|>assistant`) have discrimination ratios up to 2.57, exceeding name tokens (max 1.82). This is an architectural consequence of causal attention: final tokens attend to everything and act as sequence summaries.

**Important caveat**: High discrimination ratio does NOT mean the DiT reads identity exclusively from these positions. Exp 4 showed that injecting only these positions (high_identity strategy) is unreliable -- it works for some celebrities (Brad Pitt: 0.915) and fails for others (DiCaprio: 0.059). The suffix tokens are identity-concentrated but the DiT reads from the full sequence.

### 3. Celebrity Triangulation Works (Borderline) at Layer -2

Exp 3 Part A: SLERP interpolation of 5 nearest celebrity embeddings produces recognisable faces for held-out celebrities. Mean ArcFace scores:

| Strategy | Mean | Std |
|----------|------|-----|
| high_identity_only | 0.373 | 0.127 |
| name_tokens_only | 0.346 | 0.102 |
| all_positions | 0.336 | 0.082 |

high_identity_only beats all_positions, confirming the suffix positions carry concentrated signal. Dev Patel outlier at 0.725 shows the ceiling is high when nearest neighbours are good.

Exp 3 Part B: Open-set triangulation (random faces against celebrity DB) scored 0.236 mean. Simu Liu was nearest neighbour for 7/10 random faces -- the 99-celebrity DB is too sparse and the generic prompts mode-collapsed.

### 4. Identity Encoding is Celebrity-Specific (Bimodal)

**This is the most novel finding.** Exp 4 Part B showed that injecting only name token embeddings into a generic prompt produces ArcFace scores that split into two populations:

**Name-token-concentrated celebrities** (identity lives in name tokens):
- Brad Pitt: 0.927
- Morgan Freeman: 0.913
- Will Smith: 0.916
- Tom Cruise: 0.868
- Leonardo DiCaprio: 0.780

**Name-token-distributed celebrities** (identity spread across full sequence):
- Tom Hanks: 0.052
- Johnny Depp: 0.109
- Robert Downey Jr: 0.262
- Keanu Reeves: 0.341

all_tokens injection works for ALL celebrities (mean 0.899, std 0.030), confirming identity is always present -- the question is where.

**First predictor candidate: n_name_tokens.** The raw data shows a correlation:
- All 2-token celebrities that work (Brad Pitt, Morgan Freeman, Will Smith, Tom Cruise) score > 0.86
- Most 1-token celebrities fail (Tom Hanks 0.052, Johnny Depp 0.109, Keanu Reeves 0.341)
- Robert Downey Jr breaks the pattern (2 tokens, 0.262) -- 3-word name likely tokenises oddly
- Leonardo DiCaprio is a 1-token outlier that works (0.780)

This is correlational, not causal. Even if verified, it needs a follow-up showing that fixing the token mapping (correctly identifying all subword pieces) actually moves celebrities from fail to works. Otherwise the correlation could be spurious -- maybe the model just doesn't know those faces as well.

**Other untested hypotheses**:
- Training frequency: more common celebrities may have stronger name-token representations
- Name ambiguity: "Tom" appears in many celebrity names but Tom Cruise works fine, so this alone doesn't explain it

### 5. Low-Identity Positions Are Safe Manipulation Targets

Exp 4 Part A confirmed that the 4 low-identity positions identified in Exp 2 tolerate replacement with noise (0.381 similarity to baseline), zeros (0.386), or mean embeddings (0.467) without destroying image coherence. This maps out the "safe to touch" regions of the token sequence for any future embedding manipulation work -- these positions can be modified without degrading generation quality.

### 6. Scene Injection Fails

Exp 4 Part C: Injecting triangulated identity tokens into scene prompts scored 0.155 mean -- essentially random. The scene prompt's own token structure overwhelms the injected identity. This means you cannot control identity independently of scene by manipulating specific token positions at layer -2.

**Possible explanations**:
- The scene tokens actively suppress foreign identity signals
- Layer -2 identity representations are too entangled with surrounding context
- The cap_embedder projection (2560 -> 3840) may not preserve position-local information

## Key Technical Facts (Updated)

- Qwen3-4B: hidden_size=2560, 36 layers, decoder-only with causal attention
- Model ID: Tongyi-MAI/Z-Image-Turbo (NOT Alpha-VLLM)
- Pipeline uses hidden_states[-2] by default. Phase 2 found layer -6 is optimal for identity.
- Chat template: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- enable_thinking=True produces SHORTER sequences (no thinking tags)
- Typical prompt = ~19 tokens (very short). No BOS token. EOS=151645, PAD=151643.
- prompt_embeds is a LIST of variable-length tensors (padding stripped)
- Pipeline accepts pre-computed prompt_embeds -- confirmed working for injection
- RTX 6000 Ada 48GB: model uses ~22GB VRAM, no CPU offloading needed
- Celebrity DB: 99 valid celebrities, mean consistency 0.72, saved as pickle

## Phase 4 Priorities

1. **Multi-layer sweep** (layers -2, -4, -6) on 99 celebrities -- methodological fix for Exp 1
2. **Direct replication**: run layer -6 on Phase 2's original 5 celebrities to validate methodology
3. **Investigate bimodal split**: what predicts name-token vs distributed identity encoding?
4. **Layer -6 triangulation**: re-run Exp 3 at layer -6 to see if it clears 0.4 threshold
5. **Denser celebrity DB**: test with 500+ celebrities for open-set triangulation
