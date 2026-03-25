# Z-Image Phase 2 Notebook Design

## Goal

Build a Jupyter notebook (`zimage_phase2.ipynb`) on RunPod that runs four Z-Image Turbo experiments. The notebook must produce rich visuals for the human and comprehensive JSON data for the AI at every step.

## Context

Z-Image Turbo has been selected as our primary architecture (Cycle 10 decision). Key facts discovered during exploration:

- **VAE:** 16 channels, AutoencoderKL, scaling_factor=0.3611
- **Denoiser:** ZImageTransformer2DModel (DiT, not UNet)
- **Text encoder:** Qwen3ForCausalLM — decoder-only LLM with hidden_size=2560, 36 layers
- **Encoding method:** `hidden_states[-2]` (second-to-last layer), variable-length per-token embeddings, chat template with `enable_thinking=True`
- **Inference:** 9 steps, guidance_scale=0.0, bfloat16
- **VRAM:** ~21GB, needs `enable_model_cpu_offload()` on L4 (22GB)
- **Face channels:** 1, 6, 9, 12 (visually + numerically confirmed)
- **Discrimination:** 2.5x better than SDXL, but PCA clustering still negative

## Architecture: Single Notebook, Independent Sections

One notebook with shared model loading at the top. Each experiment section is self-contained: it saves its own JSON results + visuals after completion. Sections can be re-run independently after the model is loaded.

```
Section 0: Setup & Model Load          (~3 min)
Section 1: Qwen3 Conditioning Analysis (~15-20 min)
Section 2: Multi-Channel Face Swap     (~30-40 min, heavy generation)
Section 3: Identity Emergence Timing   (~20-30 min)
Section 4: Summary Dashboard           (~1 min)
```

## Runtime Environment

- **GPU:** RunPod L4 (22GB VRAM)
- **Notebook location:** `/workspace/identity-analysis/zimage_phase2.ipynb`
- **Results saved to:** `/workspace/identity-analysis/results/zimage_phase2/`
- **JSON data saved to:** `/workspace/identity-analysis/research_data/` (for git commit to main repo)

## Section 0: Setup & Model Load

### Code

```python
# Clear GPU, imports, load Z-Image with CPU offload
# Define shared helpers:
#   generate(prompt, seed) -> (image, latent)  — wraps pipe() with latent capture
#   generate_pair(prompt_a, prompt_b, seed) -> (img_a, img_b, lat_a, lat_b)
#   save_json(name, data) -> writes to results dir
#   display_grid(images, titles, cols) -> matplotlib grid with titles
#   display_comparison(rows_of_images, row_labels, col_labels) -> labeled comparison grid
```

### Latent Capture

Z-Image's pipeline doesn't expose intermediate latents by default. Strategy: use `callback_on_step_end` to capture latents at each denoising step.

```python
captured_latents = []
def capture_callback(pipe, step, timestep, callback_kwargs):
    captured_latents.append(callback_kwargs["latents"].detach().cpu().clone())
    return callback_kwargs
```

### Output
- Print: GPU info, VRAM, model architecture summary
- Save: `section0_setup.json` with model config, GPU info, timestamp

## Section 1: Qwen3 Conditioning Space Analysis

### Question
Is identity linearly separable in Qwen3 LLM embeddings? (Novel — no one has tested this on a decoder-only LLM text encoder)

### Predictions (for CYCLES.md)
- **Primary:** Mean-pooled Qwen3 hidden states will show positive identity silhouette (>0.2), similar to CLIP. Language models encode identity in their representations regardless of training objective.
- **Alternative:** Last-token pooling will outperform mean pooling (decoder-only models concentrate information in later positions).
- **If rejected:** Identity separability is specific to contrastive training (CLIP), not a universal property of language representations.

### Method

**Embedding set:** 5 celebrities x 5 prompt templates = 25 unique prompts → 25 embeddings (seeds don't affect text encoding — same prompt always gives same embedding)

**Visual sample set:** 5 celebrities x 3 templates x 1 seed = 15 generated images (for the display grid only)

Celebrities: Brad Pitt, Taylor Swift, Morgan Freeman, Scarlett Johansson, Keanu Reeves

Templates (varied scene/context, same identity):
1. "portrait photo of {name}, studio lighting, neutral background"
2. "photo of {name} walking in a park, natural sunlight"
3. "close-up of {name} laughing, dramatic lighting"
4. "photo of {name} at a formal event, elegant attire"
5. "{name} in a casual setting, relaxed expression"

**Extraction:** Run each prompt through the Qwen3 encoder (via `pipe._encode_prompt`), collect:
- `hidden_states[-2]` (what the pipeline actually uses)
- `hidden_states[-1]` (final layer for comparison)
- `hidden_states[-6]` (deeper layer)
- Attention mask (to know which tokens are real vs padding)

**Pooling strategies to test:**
1. Mean pool over non-padding tokens
2. Last non-padding token (decoder-only standard)
3. Max pool over non-padding tokens
4. Name-tokens only (extract just the celebrity name token embeddings)

**Analysis:**
- PCA in 2D, 5D, 10D → silhouette scores per pooling strategy per layer
- Per-token cosine similarity matrix (tokens from same-identity prompts vs different-identity)
- Token position sensitivity: for each token position, compute cross-identity variance

### Visuals (for human)

1. **Sample generations grid** — 5x3 grid: 5 celebrities x 3 templates. Shows who we're analyzing.
2. **PCA scatter plots** — 2D and 3D, colored by celebrity, for each pooling strategy. Side-by-side comparison: mean pool vs last token vs name-tokens-only.
3. **Silhouette score bar chart** — horizontal bars comparing all pooling x layer combinations. Clear winner highlighted.
4. **Token-level heatmap** — for one prompt pair (same template, different person), show per-token cosine similarity. Highlights which token positions diverge.
5. **Layer-by-layer progression** — line chart of silhouette score from layer -1 to layer -6. Shows where identity signal is strongest in the network.

### Data (JSON)

```json
{
  "section": "qwen3_conditioning",
  "model": "Qwen3ForCausalLM",
  "hidden_size": 2560,
  "layers_tested": [-1, -2, -6],
  "pooling_strategies": ["mean", "last_token", "max", "name_tokens"],
  "n_celebrities": 5,
  "n_templates": 5,
  "n_seeds": 3,
  "silhouette_scores": {
    "mean_pool": {"layer_-1": {"2d": ..., "5d": ..., "10d": ...}, ...},
    "last_token": {...},
    "max_pool": {...},
    "name_tokens": {...}
  },
  "best_config": {"pooling": "...", "layer": ..., "dims": ..., "silhouette": ...},
  "token_similarity_matrix": {...},
  "comparison_with_clip": {"clip_best": 0.403, "qwen3_best": ...}
}
```

## Section 2: Multi-Channel Face Swap

### Question
Can swapping face-dedicated channels (1, 6, 9, 12) simultaneously transfer identity, overwhelming the DiT's healing?

### Predictions
- **Primary:** Swapping all 4 face channels (25% of latent) will produce visible identity blending — more than SDXL single-channel swap (which got healed).
- **Progressive:** More channels swapped = more identity transfer. 1 channel partial, 2 channels moderate, 4 channels strong.
- **If rejected:** DiT healing is as aggressive as UNet healing regardless of how many channels are swapped.

### Method

**Identity pairs:** 3 pairs, 3 seeds each
- Brad Pitt vs Morgan Freeman (same gender, different ethnicity)
- Taylor Swift vs Scarlett Johansson (same gender, similar ethnicity)
- Brad Pitt vs Taylor Swift (different gender)

**Swap timing:** At step 5 of 9 (mid-denoising, ~55%). Use `callback_on_step_end` to intercept and modify `callback_kwargs["latents"]` in-place, then return the modified kwargs. The pipeline continues denoising from the modified state.

```python
def swap_channels_callback(channels_to_swap, source_latent_at_step):
    """Returns a callback that swaps specified channels from source_latent."""
    def callback(pipe, step, timestep, kwargs):
        if step == 4:  # 0-indexed, so step 4 = after step 5
            latents = kwargs["latents"]
            for ch in channels_to_swap:
                latents[:, ch] = source_latent_at_step[:, ch].to(latents.device)
        return kwargs
    return callback
```

This requires first generating the source image with step capture (Section 3's callback), then using the captured step-5 latent as the donor.

**Swap configurations:**
1. No swap (control)
2. Single channel swaps: ch1 only, ch6 only, ch9 only, ch12 only
3. Pair swaps: ch1+6, ch1+9, ch6+12, ch9+12
4. Triple swap: ch1+6+9
5. Full face swap: ch1+6+9+12
6. Non-face control: ch3+4+8+10 (background channels, same count)

**Metrics per swap:**
- SSIM of swapped result to source (identity donor) image
- SSIM of swapped result to target (original identity) image
- Per-channel L2 distance: swapped latent vs source latent, swapped latent vs target latent
- Cosine similarity of flattened latent to source/target

### Visuals (for human)

1. **Source/Target reference** — show the unmodified images for each identity pair and seed
2. **Progressive swap grid** — rows = number of channels swapped (0, 1, 2, 3, 4), columns = seeds. Shows gradual identity bleeding.
3. **Single channel comparison** — 4-column grid: swap ch1 | ch6 | ch9 | ch12. Shows which individual channel has most visual impact.
4. **Face vs non-face swap** — side by side: ch1+6+9+12 swap vs ch3+4+8+10 swap. The control test.
5. **Metric heatmap** — SSIM-to-source and SSIM-to-target as a colored matrix across all swap configs.

### Data (JSON)

```json
{
  "section": "multi_channel_swap",
  "swap_step": 5,
  "total_steps": 9,
  "pairs": [...],
  "swap_configs": {
    "no_swap": {"ssim_to_source": [...], "ssim_to_target": [...], ...},
    "ch1_only": {...},
    "ch1_6_9_12": {...},
    "ch3_4_8_10_control": {...}
  },
  "progressive_transfer_curve": {
    "n_channels": [0, 1, 2, 3, 4],
    "mean_ssim_to_source": [...],
    "mean_ssim_to_target": [...]
  }
}
```

## Section 3: Identity Emergence Timing

### Question
At which of Z-Image's 9 flow-matching steps does identity information lock in?

### Predictions
- **Primary:** Identity emerges proportionally earlier than SDXL. SDXL: steps 7-9 of 20 (~40%). Z-Image: steps 3-5 of 9 (~40-55%).
- **Alternative:** Flow-matching may produce a different emergence pattern — more gradual or more sudden than DDPM.

### Method

**Pairs:** 3 identity pairs, 5 seeds each. Same pairs as Section 2 for consistency.

**Capture:** Use `callback_on_step_end` to save latent at every step (1-9).

**Analysis per step:**
- Per-channel L2 distance between identity pairs (same seed, different person)
- Cosine similarity of full latent between identity pairs
- Decode intermediate latents to images (for visual timeline)

### Visuals (for human)

1. **Step-by-step image timeline** — for one pair + one seed: 9 images showing the generation evolving. Two rows (person A, person B) so you can see when they diverge.
2. **Per-channel divergence curves** — 16 line plots (one per channel), x=step, y=L2 distance between identities. Face channels (1,6,9,12) highlighted in color, others in grey.
3. **Divergence onset markers** — vertical lines on the plot marking when each channel crosses a threshold (e.g., >2 std above step-1 baseline).
4. **SDXL comparison annotation** — overlay SDXL's emergence window (steps 7-9 of 20, normalized to 0-1 scale) on the Z-Image plot.

### Data (JSON)

```json
{
  "section": "identity_emergence",
  "total_steps": 9,
  "pairs": [...],
  "per_step_per_channel_l2": {
    "step_1": {"ch0": [...], "ch1": [...], ...},
    ...
  },
  "emergence_step_per_channel": {
    "ch0": 5, "ch1": 3, ...
  },
  "face_channels_emergence": {"mean": 3.8, "std": 0.5},
  "background_channels_emergence": {"mean": 5.2, "std": 0.8},
  "comparison_with_sdxl": {
    "sdxl_emergence_normalized": 0.4,
    "zimage_emergence_normalized": ...
  }
}
```

## Section 4: Summary Dashboard

Print all key metrics, comparison table (Z-Image phase 2 vs SDXL vs Z-Image phase 1), and save comprehensive `zimage_phase2_results.json` combining all section results.

### Visuals
- Summary table printed to stdout
- Key finding callouts (boxed text for the most important results)

### Data
- Combined JSON with all section results
- Copy to `/workspace/identity-analysis/research_data/zimage_phase2_results.json` for git sync

## Shared Design Patterns

### Incremental saves
Every section saves its own JSON immediately after completing. If the notebook crashes in Section 3, Sections 1-2 data is already on disk.

### Visual consistency
- All matplotlib figures use `figsize=(16, 10)` or wider for readability
- Consistent color scheme: face channels in warm colors (red, orange, yellow, coral), background channels in cool greys
- Celebrity colors consistent across all plots (Brad Pitt = blue, Taylor Swift = red, etc.)
- All grids have titles, axis labels, and legends
- `plt.tight_layout()` on everything
- `dpi=150` for sharp output in Jupyter

### Generation helper
```python
def generate(prompt, seed, capture_steps=False):
    """Generate image + latent, optionally capturing per-step latents."""
    gen = torch.Generator("cuda").manual_seed(seed)
    captured = []
    callback = None
    if capture_steps:
        def cb(pipe, step, timestep, kwargs):
            captured.append(kwargs["latents"].detach().cpu().clone())
            return kwargs
        callback = cb

    result = pipe(
        prompt,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=gen,
        output_type="pil",
        callback_on_step_end=callback,
    )
    # Encode final image back through VAE to get final latent
    latent = pipe.vae.encode(
        pipe.image_processor.preprocess(result.images[0]).to(pipe.vae.device, pipe.vae.dtype)
    ).latent_dist.sample() * pipe.vae.config.scaling_factor

    return {
        "image": result.images[0],
        "latent": latent.detach().cpu(),
        "step_latents": captured if capture_steps else None,
    }
```

### Qwen3 embedding extraction
```python
def get_qwen3_embeddings(prompts, layers=[-1, -2, -6]):
    """Extract Qwen3 hidden states for a list of prompts."""
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = pipe.tokenizer(
            [text], padding="max_length", max_length=512,
            truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(pipe.text_encoder.device)
        mask = inputs.attention_mask.to(pipe.text_encoder.device).bool()

        with torch.no_grad():
            out = pipe.text_encoder(
                input_ids=input_ids, attention_mask=mask, output_hidden_states=True
            )

        token_count = mask[0].sum().item()
        layer_embeds = {}
        for layer_idx in layers:
            h = out.hidden_states[layer_idx][0][:token_count].cpu().float()
            layer_embeds[f"layer_{layer_idx}"] = h

        # Find name token positions
        tokens = pipe.tokenizer.convert_ids_to_tokens(input_ids[0][:token_count])

        results.append({
            "prompt": prompt,
            "token_count": token_count,
            "tokens": tokens,
            "layer_embeds": layer_embeds,
        })
    return results
```

## Estimated Runtime

| Section | Generations | Est. time (CPU offload) |
|---------|------------|------------------------|
| 0: Setup | 0 | 3 min (model load) |
| 1: Conditioning | 15 images (display grid) + 25 embedding extractions (fast, text-only) | 10-15 min |
| 2: Channel swap | 3 pairs x 3 seeds x ~12 configs = ~108 | 40-60 min |
| 3: Emergence | 3 pairs x 5 seeds x 2 = 30 | 20-30 min |
| 4: Summary | 0 | 1 min |
| **Total** | **~153 generations** | **~75-105 min** |

## Optimization Notes

- Section 1 embedding extraction is fast (no image generation needed for the analysis — just text encoder forward pass). Generate a smaller sample grid (5x3=15 images) for visual display separately.
- Section 2 is the bottleneck. Consider reducing to 2 seeds per pair if time is critical.
- All sections print progress after each generation (`seed N/M done`).
- JSON saves happen incrementally — after each sub-test within a section, not just at the end.
