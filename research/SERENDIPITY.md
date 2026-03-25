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

### 3. Ch 3 identity signal emerges BEFORE Ch 0 foundation
**Found during:** Exp 3
**What:** Ch 3 identity divergence begins at steps 7-9, slightly leading Ch 0. The fingerprint forms before the foundation fully resolves.
**Why it matters:** Suggests the UNet has a specific temporal ordering for identity construction. Could inform when to inject identity conditioning for maximum effect.
**Follow up:** Does this ordering hold for different guidance scales or step counts?
