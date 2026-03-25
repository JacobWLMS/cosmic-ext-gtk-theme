# Identity Signal Analysis in Diffusion Model Latent Spaces

## Project Goal

Investigate whether facial identity information can be isolated, measured, and manipulated in diffusion model latent spaces through frequency-domain analysis. The core hypothesis is that identity signals occupy specific frequency bands and/or VAE channels in the latent space, and can be separated from scene/style/composition information.

## Research Questions

1. **Paired frequency analysis** (Exp 1): When only identity changes in a prompt pair, what changes in the latent's frequency domain?
2. **Within-identity invariance** (Exp 2): What stays constant when the same person appears in different contexts?
3. **Identity emergence** (Exp 3): At which denoising step does identity information appear?
4. **Reference correlation** (Exp 4): How do latent differences correlate with perceptual identity metrics (ArcFace, CLIP)?
5. **Frequency injection** (Exp 5): Can we transplant identity by swapping frequency bands between latents?
6. **Channel importance** (Exp 6): Do specific VAE channels carry more identity information?
7. **PCA on identity** (Exp 7): Is identity a linear subspace in latent space?

## Target Models

- **SDXL** (primary) — Stable Diffusion XL Base 1.0, 4-channel VAE, fp16
- **Lumina2** — Alpha-VLLM/Lumina-Image-2.0, 16-channel VAE, bfloat16
- **Flux** (planned) — to be added

## Architecture

```
identity_analysis/       # Core library
  pipeline.py            # PipelineWrapper: unified SDXL/Lumina2 inference with latent capture
  frequency.py           # FFT utilities: per-channel FFT, band masks, energy computation
  scoring.py             # Identity metrics: ArcFace, CLIP similarity, MSE, SSIM
  plotting.py            # Visualization utilities
  utils.py               # Prompt pairs, face detection, output management

experiments/             # Experiment scripts (each has a run() entry point)
  exp1_paired_frequency.py
  exp2_within_identity.py
  exp3_identity_emergence.py
  exp4_reference_correlation.py
  exp5_frequency_injection.py
  exp6_channel_importance.py
  exp7_pca_identity.py

identity_signal_analysis.ipynb   # Main Colab notebook for running experiments
```

## Running Experiments

Experiments are designed to run on Google Colab with a T4 GPU (or better). The notebook clones this repo, installs deps, and runs each experiment sequentially. Each experiment's `run()` function:
- Loads a model via `PipelineWrapper`
- Generates image pairs/sets across multiple seeds (typically 50)
- Performs frequency-domain analysis on latents
- Saves plots and CSV results to `outputs/<timestamp>/<experiment_name>/`

Key parameters across experiments: `model_type` (sdxl/lumina2), `n_seeds`, `num_steps`, `save_latents`.

## Development Workflow

**All code changes MUST follow this exact process. No exceptions.**

1. **Edit code locally** (experiments, pipeline, etc.)
2. **`git commit && git push`**
3. **In Colab: Runtime → Restart runtime** (Python caches modules — `git pull` alone is NOT enough)
4. **Run setup cells** (clone/pull, imports, config) — the pull cell fetches latest code
5. **Run experiment**

**Why restart?** Python's import system caches modules in memory. Even after `git pull` updates files on disk, `import experiments.exp5` still uses the OLD cached version. `importlib.reload()` is unreliable with nested imports. Runtime restart is the only way to guarantee fresh code.

**Do NOT:**
- Use `importlib.reload()` for production runs
- Use in-memory cell edits (`update_cell`) for code that needs to persist
- Skip the runtime restart after pushing code changes
- Run experiments on stale cached modules

## Compute

- **Colab Pro** with 100 compute units/month — this is a hard budget constraint
- Local: RTX 3080 8GB (faster than T4, free — prefer for iterative dev)
- SDXL runs at fp16, 512x512, ~12-20 inference steps

### Compute Unit Rates (measured March 2026, from mccormickml.com)

| GPU | CU/hour | $/hr | Time from 100 CU | VRAM | FP16 TFLOPS |
|-----|---------|------|-------------------|------|-------------|
| T4 | 1.19 | $0.12 | **84 hours** | 15 GB | 65 |
| L4 | 1.71 | $0.17 | **58 hours** | 22.5 GB | 121 |
| A100 40GB | 5.40 | $0.54 | **18.5 hours** | 40 GB | 312 |
| A100 80GB | 7.52 | $0.75 | **13 hours** | 80 GB | 312 |
| G4 (RTX PRO 6000) | 8.71 | $0.87 | **11.5 hours** | 96 GB | ? |
| H100 | ? | ? | ? | 80 GB | 990 |

Available runtimes: CPU, T4, L4, A100, G4, H100, TPU v5e-1, TPU v6e-1.

Notes:
- T4 does NOT support bfloat16 or FP8 — use float16 only
- L4 supports bfloat16 (needed for Lumina2 which uses bfloat16)
- A100 80GB selected via "High RAM" slider
- G4 is actually an NVIDIA RTX PRO 6000 (Blackwell), 96GB VRAM

### Compute Unit Strategy

100 CU/month gives plenty of room on T4/L4, less so on A100+.

- **T4 (1.19 CU/hr)**: setup, debugging, pilot runs, SDXL experiments (fp16 only)
- **L4 (1.71 CU/hr)**: best value for production runs — 2x T4 speed, bfloat16 support, 22.5GB VRAM fits Lumina2 and possibly Flux with offloading. Only 44% more expensive than T4
- **A100 (5.40 CU/hr)**: Flux full runs if L4 VRAM isn't enough. Batch all Flux work into one session
- **Prefer local RTX 3080 for**: iterative development, code testing, single-experiment debugging (free)
- **Always**: disconnect runtime when idle, minimize wallclock time
- **Before full experiments** (n_seeds=50): pilot run with n_seeds=3-5 on T4 to catch errors
- Do NOT leave expensive runtimes connected while analyzing results — disconnect immediately

## Dependencies

torch, diffusers, transformers, accelerate, insightface, open-clip-torch, scipy, scikit-image, pandas, matplotlib
