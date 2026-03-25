"""Experiment 1: Paired Latent Frequency Analysis.

Question: When only identity changes, what changes in the latent's frequency domain?
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
    plot_frequency_band_comparison,
    plot_frequency_heatmaps,
    plot_image_grid,
)
from identity_analysis.utils import get_output_dir, get_prompt_pairs


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
):
    """Run Experiment 1: Paired Latent Frequency Analysis."""
    out_dir = get_output_dir("experiment_1", output_base)
    print(f"Experiment 1 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    prompt_pairs = get_prompt_pairs()
    seeds = list(range(n_seeds))

    all_diff_magnitudes = []
    all_energy_a = []
    all_energy_b = []
    results_rows = []

    for pair_idx, (prompt_a, prompt_b) in enumerate(prompt_pairs):
        print(f"\nPrompt pair {pair_idx + 1}/{len(prompt_pairs)}:")
        print(f"  A: {prompt_a[:60]}...")
        print(f"  B: {prompt_b[:60]}...")

        for seed in tqdm(seeds, desc=f"  Seeds (pair {pair_idx + 1})"):
            # Generate paired images
            res_a = pipe.generate(
                prompt_a, seed,
                save_step_latents=False,
                output_dir=out_dir if save_latents else None,
            )
            res_b = pipe.generate(
                prompt_b, seed,
                save_step_latents=False,
                output_dir=out_dir if save_latents else None,
            )

            lat_a = res_a["final_latent"]
            lat_b = res_b["final_latent"]

            # Compute difference
            diff = lat_a - lat_b
            diff_mag, diff_phase = fft_2d_per_channel(diff)
            all_diff_magnitudes.append(diff_mag)

            # Frequency band energy
            energy_a = compute_frequency_band_energy(lat_a)
            energy_b = compute_frequency_band_energy(lat_b)
            all_energy_a.append(energy_a)
            all_energy_b.append(energy_b)

            # Per-channel difference stats
            for c in range(diff_mag.shape[0]):
                results_rows.append({
                    "pair_idx": pair_idx,
                    "seed": seed,
                    "channel": c,
                    "diff_magnitude_mean": float(np.mean(diff_mag[c])),
                    "diff_magnitude_std": float(np.std(diff_mag[c])),
                    "diff_magnitude_max": float(np.max(diff_mag[c])),
                    "raw_diff_mean": float(np.mean(np.abs(diff[0, c]))),
                })

            if save_latents:
                np.save(
                    out_dir / "latents" / f"pair{pair_idx}_seed{seed}_diff.npy",
                    diff,
                )

            # Save example images for first seed
            if seed == 0:
                res_a["image"].save(
                    out_dir / "plots" / f"pair{pair_idx}_identity_a.png"
                )
                res_b["image"].save(
                    out_dir / "plots" / f"pair{pair_idx}_identity_b.png"
                )

            del res_a, res_b, lat_a, lat_b, diff
            gc.collect()

    # Aggregate results
    all_diff_magnitudes = np.array(all_diff_magnitudes)
    mean_diff_mag = np.mean(all_diff_magnitudes, axis=0)

    # Plot average difference frequency magnitude per channel
    plot_frequency_heatmaps(
        mean_diff_mag,
        out_dir / "plots" / "avg_diff_frequency_magnitude.png",
        title=f"Avg Frequency Magnitude of Identity Difference ({model_type.upper()})",
    )

    # Plot frequency band energy comparison (averaged)
    mean_energy_a = np.mean(all_energy_a, axis=0)
    mean_energy_b = np.mean(all_energy_b, axis=0)
    plot_frequency_band_comparison(
        mean_energy_a,
        mean_energy_b,
        out_dir / "plots" / "frequency_band_comparison.png",
        label_a="Identity A",
        label_b="Identity B",
        title=f"Frequency Band Energy by Identity ({model_type.upper()})",
    )

    # Per-channel summary
    n_channels = mean_diff_mag.shape[0]
    channel_summary = []
    for c in range(n_channels):
        channel_summary.append({
            "channel": c,
            "mean_diff_magnitude": float(np.mean(mean_diff_mag[c])),
            "max_diff_magnitude": float(np.max(mean_diff_mag[c])),
        })

    # Save CSV
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    pd.DataFrame(channel_summary).to_csv(
        out_dir / "channel_summary.csv", index=False
    )

    print(f"\nExperiment 1 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
