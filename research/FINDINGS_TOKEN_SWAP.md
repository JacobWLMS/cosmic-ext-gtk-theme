# Token Swap Experiment Results

*Date: 2026-03-26 | Platform: research-lab on RunPod L4*

## What We Tested

Surgical replacement of name-token embeddings in Qwen3's hidden states before passing to the Z-Image DiT. Only 2-3 token positions (out of ~19) are modified — everything else stays identical.

## Identity Swap Results

| Swap | Tokens Changed | SSIM to Source | SSIM to Target | Visual Result |
|------|---------------|---------------|---------------|---------------|
| Brad Pitt → Morgan Freeman | 2 | 0.523 | **0.940** | Pure Morgan Freeman. No trace of Brad Pitt. |
| Taylor Swift → Keanu Reeves | 2 | 0.552 | **0.696** | Asian man with Keanu's features + Taylor's context bleed |
| Morgan Freeman → Scarlett Johansson | 2 | 0.480 | **0.564** | Mostly Scarlett, slight Morgan pose/eyebrow influence |

### Key observations:
- **Same-gender swap (Brad→Morgan):** Nearly perfect. 0.94 SSIM means the swapped image is almost pixel-identical to the target.
- **Cross-gender swaps:** Weaker. Non-name tokens carry gendered context that fights with the injected identity. Taylor→Keanu produced an unexpected Asian male hybrid.
- **The swap is too effective for same-gender:** It doesn't blend — it completely replaces. There's nothing left of the source identity.

## Scaling Experiment

| Condition | Visual Result |
|-----------|--------------|
| Original (1.0x) | Brad Pitt |
| 0.5x scale | Brad Pitt (identical) |
| 1.5x scale | Brad Pitt (identical) |
| 2.0x scale | Brad Pitt (identical) |
| Zeroed (0.0x) | Asian man (default face) |
| Random noise | Asian man (default face) |

### Key insight: Identity is direction, not magnitude

Scaling the name-token embeddings by 0.5x or 2.0x produces the exact same Brad Pitt. The model only cares about the *direction* of the embedding vector, not its magnitude. This is consistent with transformer attention mechanics — softmax normalizes, so magnitude washes out.

But zeroing completely (0.0x) or replacing with random noise produces the model's **default identity**: an Asian male face. This is the prior — what the model generates when it has no identity signal at all.

## The Default Identity Problem

When name tokens are zeroed or randomized, Z-Image generates an Asian male face. This is likely:

1. **Training data distribution** — Z-Image is from Alibaba/Tongyi, trained heavily on Chinese-language data and Asian faces
2. **The model's learned "average face"** — when no identity information is provided, the DiT falls back to its most common training example

### Where does this default come from?

The default identity emerges even though the rest of the prompt ("portrait photo of, studio lighting, neutral background") is unchanged. This means:

1. It's NOT in the non-name tokens (those are the same across all conditions)
2. It's NOT in the name tokens (those are zeroed)
3. It must come from the **DiT's own weights** — the denoiser has a built-in prior for what a "portrait photo" looks like when no identity is specified

This is the DiT's unconditional face prior, activated when the identity signal is absent from the conditioning.

### Can we intercept the default?

The default identity emerges during denoising, not during text encoding. Possible interception points:

1. **DiT cross-attention maps** — which spatial positions attend to which token positions? When name tokens are zeroed, the face region must be attending to *something* to produce a coherent face. That something is either the non-name tokens or the DiT's internal representations.

2. **DiT self-attention** — the transformer has self-attention layers that can generate face structure from its own weights, independent of conditioning. This is the "unconditional" generation path.

3. **Timestep conditioning** — the DiT receives timestep embeddings that may carry implicit priors about what to generate at each denoising stage.

## Cross-Gender Swap Analysis

The Taylor Swift → Keanu Reeves result (Asian man with Keanu features) reveals that non-name tokens are NOT fully identity-neutral:

- "portrait photo of {name}" — the word positions around the name carry contextual associations
- Taylor Swift's template context carries female/young/Western associations
- When Keanu's name embeddings are injected into Taylor's context, the conflicting signals produce a hybrid
- The model resolves the conflict by defaulting toward its prior (Asian male) while incorporating Keanu's specific features

This suggests that for clean cross-category swaps, we may need to swap more than just the name tokens — perhaps the entire "subject phrase" needs replacement.

## Implications

### What works:
- Same-gender, same-ethnicity identity swap via name tokens — nearly perfect (0.94 SSIM)
- Identity is a discrete, replaceable signal in name-token embedding space

### What doesn't work:
- Magnitude scaling (0.5x-2.0x) — no effect, identity is directional only
- Cross-gender swap — context tokens carry gendered associations that interfere
- Continuous identity control via scaling — binary (present/absent), not continuous

### What's surprising:
- The model has a built-in default identity (Asian male) that emerges when identity signal is removed
- This default comes from the DiT's weights, not the text encoder
- The interpolation experiment (check canvas) may show whether blending between two identities produces smooth transitions despite scaling not working

## Next Experiments

### High Priority

1. **DiT cross-attention analysis when identity is zeroed**
   - Extract cross-attention maps for: original Brad Pitt, zeroed identity, Morgan Freeman
   - Which spatial positions attend to name tokens? What do they attend to when name tokens are zero?
   - This would reveal where the default identity comes from

2. **Full subject phrase swap (not just name tokens)**
   - Instead of swapping just "Brad Pitt" tokens, swap the entire subject phrase "portrait photo of Brad Pitt"
   - May fix cross-gender issues by replacing gendered context along with identity

3. **Interpolation via embedding direction**
   - Scaling magnitude doesn't work, but rotating the embedding direction might
   - Spherical interpolation (SLERP) between Brad Pitt and Morgan Freeman embedding directions
   - This respects the directional nature of the signal

4. **Test the default identity with different prompts**
   - Zero identity with "portrait photo of [zeroed], studio lighting" → Asian man
   - Zero identity with "portrait photo of [zeroed] woman, studio lighting" → what happens?
   - Zero identity with "painting of [zeroed], oil on canvas" → does the default still appear?

### Medium Priority

5. **Non-celebrity names**
   - Does "John Smith" produce a consistent identity across prompts?
   - Does the model have a default for common names?

6. **Concept generalization beyond faces**
   - "painting in the style of [Picasso]" → swap to [Monet] tokens
   - "a [golden retriever] in a park" → swap to [husky] tokens
   - If these work, name-token manipulation is a general concept replacement mechanism

7. **Layer selection experiment**
   - We're using hidden_states[-2] (what the pipeline uses)
   - But layer -6 showed the strongest identity signal (+0.792 vs +0.742 at layer -2)
   - What happens if we inject embeddings from layer -6 instead?

### Speculative

8. **Can we override the default identity?**
   - If the default comes from DiT weights, can we bias it with a "concept token" that shifts the prior?
   - A learned embedding that replaces the zeroed name tokens and produces a *specific* default face
   - This would be a lightweight alternative to LoRA for identity control

9. **Classifier-free guidance manipulation**
   - Z-Image uses guidance_scale=0.0 (no CFG). What if we use CFG with the swapped embeddings as positive and original as negative?
   - May amplify the identity transfer and suppress the source
