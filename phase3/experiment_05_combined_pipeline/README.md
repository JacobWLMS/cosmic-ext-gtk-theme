# Experiment 5: Combined Identity Injection Pipeline

## Prerequisites

Only run if Experiments 3 or 4 show positive results.

## Hypothesis

A practical zero-training pipeline can take a reference photo + text prompt and generate identity-consistent images.

## Method

1. Reference photo → ArcFace embedding → nearest celebrities
2. Triangulate Qwen3 embeddings
3. Inject into scene prompt
4. Generate with Z-Image Turbo

## Success Criteria

- ArcFace > 0.3 against reference
- CLIP text-image alignment > 0.25
- Consistent across 5 different scene prompts

## Results

_To be filled after running_
