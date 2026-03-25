"""Experiment 2: Within-Identity Latent Invariance.

Question: What stays constant when the same person appears in different contexts?
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from identity_analysis.frequency import (
    compute_frequency_band_energy,
    fft_2d_per_channel,
)
from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import (
    plot_frequency_heatmaps,
    plot_image_grid,
    plot_line_chart,
)
from identity_analysis.utils import get_celebrity_prompts, get_output_dir


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 2: Within-Identity Latent Invariance."""
    out_dir = get_output_dir("experiment_2", output_base)
    print(f"Experiment 2 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    celeb_prompts = get_celebrity_prompts()
    n_prompts_per_celeb = 20

    results_rows = []
    celeb_latents = {}  # celeb -> list of latent arrays
    celeb_freq_mags = {}  # celeb -> list of frequency magnitude arrays
    celeb_band_energies = {}  # celeb -> list of [C, n_bands] arrays

    for celeb, prompts in celeb_prompts.items():
        print(f"\nGenerating for: {celeb}")
        celeb_latents[celeb] = []
        celeb_freq_mags[celeb] = []
        celeb_band_energies[celeb] = []

        prompts_to_use = prompts[:n_prompts_per_celeb]

        for p_idx, prompt in enumerate(tqdm(prompts_to_use, desc=f"  {celeb}")):
            # Use seed = prompt index to get variety
            seed = p_idx
            res = pipe.generate(
                prompt, seed,
                save_step_latents=False,
            )

            # Re-encode through VAE for clean latent
            encoded = pipe.encode_image(res["image"])
            celeb_latents[celeb].append(encoded)

            # FFT analysis
            mag, phase = fft_2d_per_channel(encoded)
            celeb_freq_mags[celeb].append(mag)

            band_energy = compute_frequency_band_energy(encoded)
            celeb_band_energies[celeb].append(band_energy)

            # Save example images
            if p_idx < 4:
                res["image"].save(
                    out_dir / "plots" / f"{celeb.replace(' ', '_')}_sample_{p_idx}.png"
                )

            if save_latents:
                np.save(
                    out_dir / "latents" / f"{celeb.replace(' ', '_')}_prompt{p_idx}.npy",
                    encoded,
                )

            del res, encoded
            gc.collect()

    # Analysis: per-channel variance within each celebrity vs across celebrities
    n_channels = celeb_latents[list(celeb_latents.keys())[0]][0].shape[1]

    within_variance = {}  # celeb -> [C] array of per-channel variance
    within_freq_variance = {}  # celeb -> [C] array of per-channel freq magnitude variance
    celeb_means = {}  # celeb -> [C, H, W] mean latent

    for celeb in celeb_latents:
        stacked = np.concatenate(celeb_latents[celeb], axis=0)  # [N, C, H, W]
        celeb_means[celeb] = np.mean(stacked, axis=0)  # [C, H, W]
        within_variance[celeb] = np.var(stacked, axis=0).mean(axis=(1, 2))  # [C]

        freq_stacked = np.stack(celeb_freq_mags[celeb])  # [N, C, H, W]
        within_freq_variance[celeb] = np.var(freq_stacked, axis=0).mean(axis=(1, 2))  # [C]

    # Across-celebrity variance (variance of means)
    all_means = np.stack(list(celeb_means.values()))  # [n_celebs, C, H, W]
    across_variance = np.var(all_means, axis=0).mean(axis=(1, 2))  # [C]

    # Plot within vs across variance per channel
    avg_within = np.mean(list(within_variance.values()), axis=0)
    plot_line_chart(
        np.arange(n_channels),
        {
            "Within-Identity Variance (avg)": avg_within,
            "Across-Identity Variance": across_variance,
        },
        out_dir / "plots" / "variance_within_vs_across.png",
        xlabel="Channel",
        ylabel="Variance",
        title=f"Latent Variance: Within vs Across Identity ({model_type.upper()})",
    )

    # Discrimination ratio: across / within (higher = more discriminative)
    discrimination_ratio = across_variance / (avg_within + 1e-10)
    plot_line_chart(
        np.arange(n_channels),
        {"Discrimination Ratio (across/within)": discrimination_ratio},
        out_dir / "plots" / "discrimination_ratio.png",
        xlabel="Channel",
        ylabel="Ratio",
        title=f"Channel Discrimination Ratio ({model_type.upper()})",
    )

    # Frequency domain analysis: same thing but for frequency magnitudes
    avg_within_freq = np.mean(list(within_freq_variance.values()), axis=0)

    all_freq_means = []
    for celeb in celeb_freq_mags:
        all_freq_means.append(np.mean(celeb_freq_mags[celeb], axis=0))
    across_freq_variance = np.var(all_freq_means, axis=0).mean(axis=(1, 2))

    plot_line_chart(
        np.arange(n_channels),
        {
            "Within-Identity Freq Variance (avg)": avg_within_freq,
            "Across-Identity Freq Variance": across_freq_variance,
        },
        out_dir / "plots" / "freq_variance_within_vs_across.png",
        xlabel="Channel",
        ylabel="Frequency Magnitude Variance",
        title=f"Frequency Variance: Within vs Across Identity ({model_type.upper()})",
    )

    # Frequency band analysis
    n_bands = celeb_band_energies[list(celeb_band_energies.keys())[0]][0].shape[1]
    for celeb in celeb_band_energies:
        stacked = np.stack(celeb_band_energies[celeb])  # [N, C, n_bands]
        mean_energy = np.mean(stacked, axis=0)  # [C, n_bands]
        var_energy = np.var(stacked, axis=0)  # [C, n_bands]

        results_rows.append({
            "celebrity": celeb,
            "type": "within_variance",
            **{f"ch{c}_band{b}_var": float(var_energy[c, b])
               for c in range(n_channels) for b in range(n_bands)},
            **{f"ch{c}_spatial_var": float(within_variance[celeb][c])
               for c in range(n_channels)},
        })

    # Plot mean frequency magnitude heatmap per celebrity
    for celeb in celeb_freq_mags:
        mean_mag = np.mean(celeb_freq_mags[celeb], axis=0)
        plot_frequency_heatmaps(
            mean_mag,
            out_dir / "plots" / f"mean_freq_mag_{celeb.replace(' ', '_')}.png",
            title=f"Mean Frequency Magnitude: {celeb} ({model_type.upper()})",
        )

    # Save results
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # Save discrimination ratios
    disc_df = pd.DataFrame({
        "channel": range(n_channels),
        "within_variance": avg_within,
        "across_variance": across_variance,
        "discrimination_ratio": discrimination_ratio,
        "within_freq_variance": avg_within_freq,
        "across_freq_variance": across_freq_variance,
    })
    disc_df.to_csv(out_dir / "discrimination_ratios.csv", index=False)

    print(f"\nExperiment 2 complete. Results saved to {out_dir}")
    print(f"Most discriminative channels: {np.argsort(discrimination_ratio)[::-1]}")
    pipe.cleanup()
    return out_dir
