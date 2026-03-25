# Identity Signal Analysis in Diffusion Model Latent Spaces

Exploring whether facial identity information can be isolated and manipulated in diffusion model latent spaces through frequency-domain analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run a specific experiment
python run_experiment.py --experiment 1 --model sdxl --seeds 50

# Quick mode (5 seeds)
python run_experiment.py --experiment 1 --model sdxl --quick

# Run all priority experiments
python run_experiment.py --experiment 1 2 3 6 --model sdxl --seeds 50

# Skip saving raw latents to save disk space
python run_experiment.py --experiment 1 --model sdxl --no-save-latents
```

## Experiments

1. **Paired Latent Frequency Analysis** - What changes in frequency domain when only identity changes?
2. **Within-Identity Latent Invariance** - What stays constant for the same person across contexts?
3. **Step-by-Step Identity Emergence** - At what denoising step does identity lock in?
4. **Timestep-Matched Reference Correlation** - Can we correlate reference face latents with mid-denoising latents?
5. **Naive Frequency Injection Test** - Does brute-force frequency injection shift identity?
6. **Channel Importance Analysis** - Do specific VAE channels carry more identity info?
7. **PCA on Identity** - Is identity a linear subspace in latent space?

## Hardware

Designed for RTX 3080 (10GB VRAM) with CPU offloading. All generations at 512x512.
