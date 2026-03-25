# Identity Signal Analysis in Diffusion Model Latent Spaces

Investigating whether facial identity information can be isolated, measured, and manipulated in diffusion model latent spaces through frequency-domain and channel analysis.

## Setup

Experiments run on **Google Colab** (Pro, L4/T4 GPU). The notebook clones this repo, installs deps, and runs each experiment.

1. Open `identity_signal_analysis.ipynb` in Colab
2. Connect a GPU runtime (L4 recommended)
3. Run setup cells, then the active experiment

## Experiments

| # | Name | Status | Key Finding |
|---|------|--------|-------------|
| 1 | Paired Frequency Analysis | Done | Ch 2 shows largest frequency shifts on identity change |
| 2 | Within-Identity Invariance | Done | Ch 3 is best identity discriminator (ratio 0.046) |
| 3 | Identity Emergence | Done | Identity locks in at steps 7-9 during denoising |
| 5 | Channel Identity Transplant | Done | UNet heals single-channel swaps; Ch 3 swap shows most identity shift |
| 6 | Channel Importance (Zeroing) | Done | Ch 0 catastrophic, Ch 3 significant, Ch 1/2 minimal |
| 7 | PCA on Identity | Next | Is identity a linear subspace in Ch 3? |

See [RESEARCH.md](RESEARCH.md) for full findings and the hierarchical channel model.

## Architecture

```
identity_analysis/       # Core library (pipeline, FFT, scoring, plotting)
experiments/             # Experiment scripts (each has run() entry point)
identity_signal_analysis.ipynb   # Colab notebook
EXP{N}_RESEARCH.md      # Per-experiment reports
RESEARCH.md              # Cross-experiment synthesis
```

## Model

SDXL Base 1.0 (fp16), 1024x1024, 12 steps, guidance 7.5 + negative prompt.
