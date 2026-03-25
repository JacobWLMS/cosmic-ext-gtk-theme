# Serendipity Log

Accidental or unexpected findings worth revisiting later.

---

### 1. SDXL VAE channels map to YCbCr color space
**Found during:** Exp 1-2 analysis
**What:** SDXL's 4 VAE channels correspond to luminance (Ch 0), red-cyan chrominance (Ch 1), warm-cool chrominance (Ch 2), and pattern/structure (Ch 3). This matches a HuggingFace blog post on VAE channel semantics.
**Why it matters:** Understanding channel semantics could apply to any SDXL-based task (style transfer, composition control, inpainting), not just identity.
**Follow up:** Does this mapping hold for other VAE architectures (Lumina2, Flux)?

---

### 2. UNet enforces cross-channel consistency during denoising
**Found during:** Exp 5 V2
**What:** Modifying a single channel mid-denoising gets "healed" by subsequent denoising steps. The UNet's learned prior enforces holistic consistency.
**Why it matters:** This is a general property of diffusion denoising, not specific to identity. It implies that ANY single-channel manipulation (not just identity) will be corrected by the UNet. This could explain why many naive latent editing approaches fail.
**Follow up:** Could this healing be exploited? E.g., introduce deliberate perturbations knowing the UNet will "harmonize" them into something coherent.

---

### 4. CLIP CLS token carries NO identity; mean pooling carries ALL of it
**Found during:** Exploration Test C
**What:** CLS token PCA showed 0.0 silhouette (zero clustering) while mean-pooled embeddings showed +0.40. Identity in CLIP is distributed across all token positions, not concentrated in the CLS summary token.
**Why it matters:** This is relevant for any CLIP-based application. If you're using CLS pooling for identity-related tasks, you're throwing away the signal. Mean pooling captures it. This could explain performance differences in face recognition systems using CLIP.
**Follow up:** Which token positions carry the most identity? Is it the name tokens specifically, or is it distributed even across non-name tokens?

---

### 5. Identity signal location follows the diffusion pipeline direction
**Found during:** Synthesis of all experiments
**What:** Identity is separable in conditioning (CLIP: +0.40) → partially separable in mid-denoising (Exp 3: emerges steps 7-9) → inseparable in final latent (Exp 7: all negative). The signal degrades as it flows through the pipeline.
**Why it matters:** This suggests a fundamental property of diffusion models: the denoising process entangles conditioned attributes with spatial/structural information. Any manipulation must happen early in the pipeline (conditioning) or at the conditioning-latent interface (cross-attention), not at the output.
**Follow up:** Is this true for ALL conditioned attributes (style, composition, etc.) or specific to identity?

---

### 6. Z-Image 16ch VAE has dedicated face channels (1, 6, 9, 12)
**Found during:** Z-Image Turbo exploration
**What:** In Z-Image's 16-channel VAE, channels 1, 6, 9, 12 show concentrated red activation on face regions (observed visually) AND the highest identity sensitivity numerically (0.64-0.88 mean diff vs 0.26 for the lowest). The user spotted these visually before the numbers confirmed it.
**Why it matters:** A 16-channel VAE can functionally specialize — some channels for faces, others for background/texture/structure. This is fundamentally different from SDXL's 4-channel VAE where everything is compressed. More channels = more separability = potentially viable channel-level identity manipulation.
**Follow up:** Does channel-level identity swapping actually work on Z-Image? If channels 1+6+9+12 carry identity, swapping those 4 channels might transfer identity without the UNet healing problem (since the swap covers enough channels to be self-consistent).

---

### 7. Z-Image uses Qwen3 (decoder-only LLM) as text encoder — NOT CLIP or T5
**Found during:** Cycle 9 architecture investigation
**What:** Z-Image Turbo's `model_index.json` reveals `text_encoder: Qwen3Model` with `Qwen2Tokenizer`. This is a decoder-only large language model, not a contrastive encoder (CLIP) or encoder-only model (T5). Most diffusion models use CLIP or T5; using a full LLM as text encoder is unusual.
**Why it matters:** Our CLIP conditioning finding (+0.40 identity separability) used mean-pooled contrastive embeddings. Qwen3 produces very different representations — causal, autoregressive, trained on text completion not image-text alignment. If identity is ALSO separable in Qwen3 embeddings, it would suggest identity separability is a universal property of language representations, not specific to CLIP's contrastive training. If NOT separable, it would mean our CLIP finding is architecture-specific.
**Follow up:** Test identity clustering in Qwen3 embeddings. Compare pooling strategies (last token vs mean pool — decoder models typically use last-token pooling, not mean pooling).

---

### 3. Ch 3 identity signal emerges BEFORE Ch 0 foundation
**Found during:** Exp 3
**What:** Ch 3 identity divergence begins at steps 7-9, slightly leading Ch 0. The fingerprint forms before the foundation fully resolves.
**Why it matters:** Suggests the UNet has a specific temporal ordering for identity construction. Could inform when to inject identity conditioning for maximum effect.
**Follow up:** Does this ordering hold for different guidance scales or step counts?
