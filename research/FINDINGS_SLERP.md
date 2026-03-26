# SLERP Interpolation: Continuous Identity Control via Directional Blending

*Date: 2026-03-26 | Platform: research-lab on RunPod L4*

## The Discovery

**Linear scaling of name-token embeddings has zero effect on identity** (0.5x, 1.5x, 2.0x all produce identical Brad Pitt). But **SLERP (Spherical Linear Interpolation)** between two identities' name-token embeddings produces genuine, continuous face blending. This confirms identity is encoded as a **direction** in Qwen3's embedding space, not a magnitude.

## SLERP Results: Brad Pitt <-> Morgan Freeman

| Alpha | Visual Result |
|-------|--------------|
| 0% | Pure Brad Pitt |
| 25% | Brad Pitt, slightly more wrinkled (Morgan's age influence starting) |
| 50% | Brad with increasing Morgan influence — wrinkles, skin texture |
| 75% | **Genuine hybrid** — white man with Morgan's hair, Morgan's nose, Brad's general face shape. Clothing merged: Brad's t-shirt + Morgan's jacket |
| 100% | Pure Morgan Freeman |

The 75% result is the standout — a face that is clearly neither Brad Pitt nor Morgan Freeman but a plausible blend of both. Features from each person are selectively combined.

## Why SLERP Works and Linear Scaling Doesn't

**Linear scaling** changes the magnitude of the embedding vector but not its direction. Transformer attention uses dot products followed by softmax, which normalizes magnitude. The identity information is in *where the vector points*, not how long it is.

**SLERP** rotates the vector along the great circle on the hypersphere between two identity directions. This smoothly transitions the direction, which is what the transformer actually responds to.

```
Linear scaling: ||v|| changes, v/||v|| stays the same → no identity change
SLERP: v/||v|| smoothly rotates from identity A to identity B → continuous blending
```

## Evidence

- Canvas images saved: `research/data/slerp_slerp_strip.png`
- 5-step interpolation: 0%, 25%, 50%, 75%, 100%
- Brad Pitt and Morgan Freeman (seed 42, "portrait photo, studio lighting, neutral background")

## Practical Applications

### 1. Post-Face-Swap Refiner
If IP-Adapter or InstantID does a rough face swap, SLERP at 75-90% could refine the merge at the text conditioning level. The text encoder "nudges" the generation toward the target identity without the artifacts of pixel-level face swapping. This could work as the final step in a face swap pipeline.

### 2. Continuous Identity Control
Instead of binary "this person or that person," SLERP gives a continuous dial. Useful for:
- Character design: "someone who looks 70% like Actor A and 30% like Actor B"
- Privacy: generate faces that are "inspired by" but not identical to a real person
- Animation: smoothly morph between characters

### 3. Identity-Preserving Style Transfer
At 25-50% SLERP, subtle features transfer while the dominant identity stays. Could be used to add specific facial features (Morgan's wrinkles, Brad's jawline) to any face.

## Key Insight: Clothing and Scene Also Blend

The 75% SLERP didn't just blend faces — it blended Brad's t-shirt with Morgan's jacket. This means name-token embeddings encode more than facial identity:
- Physical appearance (face, body type, age)
- Associated clothing/style
- Possibly even associated settings and lighting

The Qwen3 model has a rich, multi-dimensional concept of each person, and SLERP traverses all of these dimensions simultaneously.

## Limitations

- Only tested on 2 identities (Brad/Morgan) at 5 alpha values
- Cross-gender SLERP not yet tested (likely more complex)
- The blending is global — it affects everything associated with the name, not just the face
- Need more granular alpha values (11-21 steps) to characterize the transition curve

## Next Steps

### Layer-Targeted SLERP (Highest Priority)
Currently SLERP operates on all 2560 embedding dimensions equally. But our PCA analysis showed identity concentrates in specific directions. If we could:
1. Identify which PCA components carry face vs. clothing vs. scene information
2. SLERP only the face components at high alpha
3. Leave clothing/scene components unchanged

This would give us **face-only identity blending** while preserving the source's clothing, pose, and setting. That's the holy grail for practical face manipulation.

### SLERP with Similar Faces
Brad and Morgan are very different (age, ethnicity, build). Testing SLERP on similar faces (e.g., two white men of similar age) should produce much cleaner blends. This validates the IP-Adapter refiner use case.

### Fine-Grained Alpha Sweep
21 steps from 0% to 100% to characterize exactly where the transition happens. Is it linear? Is there a "phase transition" where the face suddenly flips?
