# Key Finding: Identity Hyperlocalization in LLM-Conditioned Diffusion Models

*Date: 2026-03-25 | Authors: Jacob + Claude*

## Summary

We discovered that **identity information is hyperlocalized in name tokens** of the Qwen3 text encoder used by Z-Image Turbo, achieving a silhouette score of **+0.792** — nearly double CLIP's +0.403. This is the first quantitative measurement of identity clustering in a decoder-only LLM text encoder for diffusion models.

## The Finding

When Z-Image Turbo generates a portrait of "Brad Pitt," the Qwen3 text encoder (Qwen3ForCausalLM, 2560-dim, 36 layers) produces per-token embeddings that are passed to the DiT denoiser. We measured how well these embeddings cluster by celebrity identity across 5 celebrities × 5 varied scene prompts.

### Results by Pooling Strategy

| Pooling Strategy | Best Silhouette | Best Layer | Dims | Interpretation |
|-----------------|----------------|------------|------|----------------|
| **Name tokens only** | **+0.792** | Layer -6 | 5D | Identity concentrated in literal name tokens |
| Max pool | +0.299 | Layer -1 | 2D | Captures some signal via outlier dimensions |
| Mean pool | +0.024 | Layer -1 | 10D | Non-name tokens drown out identity |
| Last token | -0.100 | Layer -1 | 10D | Generation prompt token carries no identity |

For reference: CLIP (SDXL) best was +0.403 with mean pooling.

### Key Observations

1. **Name tokens carry nearly all identity information.** Extracting just the 1-2 tokens corresponding to the celebrity name gives +0.792 silhouette. Mean pooling over all tokens drops to near-zero, meaning non-name tokens (scene descriptions, lighting, etc.) contain essentially no identity signal.

2. **Deeper layers are better.** Identity signal strengthens going deeper into the network: layer -6 (+0.792) > layer -5 (+0.786) > ... > layer -1 (+0.737). This is counterintuitive — the pipeline uses layer -2, not the best layer for identity.

3. **This is the opposite of CLIP.** In CLIP (SDXL), identity is distributed across all token positions (mean pool +0.403, CLS token 0.0). In Qwen3, identity is concentrated in name tokens (name tokens +0.792, mean pool ~0.0). The architectures encode identity in fundamentally different ways.

4. **Last-token pooling fails.** Decoder-only LLMs typically concentrate information in later tokens for next-token prediction. But in Z-Image's usage, the last token is a generation prompt token from the chat template (with `enable_thinking=True`), not a summary of the input. This makes last-token pooling useless for identity.

## Why This Matters

### For the research community

The entire identity-manipulation field (IP-Adapter, InstantID, PhotoMaker, etc.) bypasses the text encoder by injecting identity through external face encoders (ArcFace, CLIP-ViT). Our finding shows the text encoder **already has** a near-perfect identity representation — you just need to know where to look.

No prior work has:
- Measured identity clustering in text encoder hidden states with quantitative metrics
- Compared CLIP vs LLM text encoders for identity localization
- Shown the hyperlocalization of identity in name tokens of decoder-only LLMs
- Analyzed Z-Image Turbo's Qwen3 conditioning space at all

The closest related work is "Follow the Flow" (arxiv:2504.01137) which showed information concentrates in 1-2 tokens in T5 (FLUX) but distributes in CLIP. Our work extends this to identity-specific analysis, decoder-only LLMs, and quantitative clustering metrics.

### For practical applications

If identity lives in name tokens, we can potentially:
1. **Transplant identity** by replacing name-token embeddings between prompts
2. **Interpolate identity** by blending name-token embeddings (70% person A + 30% person B)
3. **Erase identity** by zeroing name-token embeddings to generate generic faces
4. **Control identity strength** by scaling name-token embedding magnitudes

All without retraining, without external face encoders, without latent manipulation — just embedding surgery on the text encoder output.

## Context: How We Got Here

This finding emerged from a systematic investigation across two architectures:

### Phase 1: SDXL (Experiments 1-7 + Exploration)
- Established channel hierarchy: Ch 0 (foundation) > Ch 3 (fingerprint) > Ch 2 ≈ Ch 1
- Proved identity is NOT linearly separable in VAE latent space (PCA, t-SNE, UMAP all negative)
- Discovered UNet heals single-channel perturbations during denoising
- Found identity IS separable in CLIP conditioning space (+0.403 silhouette, mean-pooled)

### Phase 2: Z-Image Turbo (Cycles 9-13)
- Confirmed latent-space inseparability is architecture-general (16ch VAE also negative)
- Found 2.5x better channel discrimination ratios but still no linear separability
- Discovered Qwen3 text encoder (decoder-only LLM, not CLIP/T5)
- **Found +0.792 identity clustering in name tokens** — the headline result

### The Research Path

| Cycle | Question | Result | Decision |
|-------|---------|--------|----------|
| 1-6 | SDXL channel analysis | Channel hierarchy validated | → Test identity subspace |
| 7 | PCA on identity in latent space | All negative silhouette | → Try nonlinear methods |
| 8 | Nonlinear + CLIP exploration | CLIP +0.403, latent still negative | → Pivot to conditioning |
| 9 | Z-Image cross-architecture validation | Latent negative, confirms finding | → Commit to Z-Image |
| 10 | Architecture decision | Z-Image chosen (more signal, less explored) | → Analyze Qwen3 encoder |
| 11 | **Qwen3 conditioning analysis** | **Name tokens +0.792** | → **Token swap experiments** |

## Experimental Details

### Model
- **Z-Image Turbo** (Tongyi-MAI/Z-Image-Turbo)
- Text encoder: Qwen3ForCausalLM (hidden_size=2560, 36 layers, 32 attention heads)
- Tokenizer: Qwen2Tokenizer with chat template + `enable_thinking=True`
- Pipeline extracts `hidden_states[-2]` (second-to-last layer)

### Dataset
- 5 celebrities: Brad Pitt, Taylor Swift, Morgan Freeman, Scarlett Johansson, Keanu Reeves
- 5 prompt templates with varied scenes (studio, park, close-up, formal event, casual)
- 25 unique prompts → 25 embeddings (text encoding is deterministic)

### Method
1. Run each prompt through Qwen3 with `output_hidden_states=True`
2. Collect hidden states from layers -1 through -6
3. Apply 4 pooling strategies: mean, last_token, max, name_tokens
4. PCA to 2D/5D/10D, compute silhouette score with celebrity labels

### Limitations
- Only 5 celebrities tested (all English-language, globally famous)
- Name-token identification uses simple string matching (may miss subword tokenization edge cases)
- Needs validation on: non-Western names, fictional characters, more celebrities, non-celebrity named entities

## Raw Data

Full results in `research/data/section1_qwen3_conditioning.json`.

## Next Steps

1. **Token embedding swap** — replace name tokens between prompts, verify identity transfer in generated images
2. **Embedding interpolation** — continuous identity blending in name-token space
3. **Robustness validation** — 15+ celebrities, non-Western names, edge cases
4. **Comparison with T5** — test same methodology on FLUX (T5-XXL) to complete the CLIP/T5/LLM comparison
5. **Write paper** — "Identity Localization in LLM-Conditioned Diffusion Models"
