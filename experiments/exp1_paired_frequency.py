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
    num_steps: int = 20,
):
    """Run Experiment 1: Paired Latent Frequency Analysis."""
    import torch; torch.cuda.empty_cache(); gc.collect()
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
        print(f"\n{'─' * 60}")
        print(f"Pair {pair_idx + 1}/{len(prompt_pairs)}:")
        print(f"  A: {prompt_a}")
        print(f"  B: {prompt_b}")

        pair_diff_mags = []
        pair_raw_diffs = []

        for seed in tqdm(seeds, desc=f"  Generating"):
            # Batched pair generation — ~2x faster than sequential
            need_images = (seed == 0)  # only decode images for first seed
            res_a, res_b = pipe.generate_pair(
                prompt_a, prompt_b, seed,
                num_inference_steps=num_steps,
                decode_images=need_images,
            )

            lat_a = res_a["final_latent"]
            lat_b = res_b["final_latent"]

            # Compute difference
            diff = lat_a - lat_b
            diff_mag, diff_phase = fft_2d_per_channel(diff)
            all_diff_magnitudes.append(diff_mag)
            pair_diff_mags.append(diff_mag)
            pair_raw_diffs.append(np.mean(np.abs(diff[0]), axis=(1, 2)))

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
            if need_images:
                res_a["image"].save(
                    out_dir / "plots" / f"pair{pair_idx}_identity_a.png"
                )
                res_b["image"].save(
                    out_dir / "plots" / f"pair{pair_idx}_identity_b.png"
                )

            del res_a, res_b, lat_a, lat_b, diff
            gc.collect()

        # Print per-pair results
        pair_mean_mag = np.mean(pair_diff_mags, axis=0)  # [C, H, W]
        pair_mean_raw = np.mean(pair_raw_diffs, axis=0)  # [C]
        n_ch = pair_mean_mag.shape[0]
        print(f"\n  Results for pair {pair_idx + 1}:")
        print(f"  {'Ch':<5} {'Freq Mag':<12} {'Raw Diff':<12}")
        for c in range(n_ch):
            print(f"  {c:<5} {np.mean(pair_mean_mag[c]):<12.2f} {pair_mean_raw[c]:<12.4f}")
        top_ch = np.argmax([np.mean(pair_mean_mag[c]) for c in range(n_ch)])
        print(f"  → Most active channel: Ch {top_ch}")

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

    # Print key results
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT 1 RESULTS: Paired Latent Frequency Analysis")
    print(f"{'=' * 60}")
    print(f"Model: {model_type.upper()} | Seeds: {n_seeds} | Prompt pairs: {len(prompt_pairs)}")
    print(f"\nPer-Channel Identity Difference (frequency domain):")
    print(f"  {'Channel':<10} {'Mean Mag':<15} {'Max Mag':<15} {'Raw Diff':<15}")
    print(f"  {'-'*55}")

    df = pd.DataFrame(results_rows)
    for c in range(n_channels):
        ch_df = df[df["channel"] == c]
        print(f"  Ch {c:<6} {ch_df['diff_magnitude_mean'].mean():<15.2f} "
              f"{ch_df['diff_magnitude_max'].max():<15.2f} "
              f"{ch_df['raw_diff_mean'].mean():<15.4f}")

    ranked = sorted(channel_summary, key=lambda x: x["mean_diff_magnitude"], reverse=True)
    print(f"\n  Most identity-sensitive channel: Ch {ranked[0]['channel']} "
          f"(mean mag: {ranked[0]['mean_diff_magnitude']:.2f})")
    print(f"  Least identity-sensitive channel: Ch {ranked[-1]['channel']} "
          f"(mean mag: {ranked[-1]['mean_diff_magnitude']:.2f})")

    # Frequency band analysis
    energy_diff = np.abs(mean_energy_a - mean_energy_b)
    band_diff_total = energy_diff.sum(axis=0)  # sum across channels per band
    top_band = np.argmax(band_diff_total)
    print(f"\n  Frequency band with largest identity difference: Band {top_band}/9 "
          f"({'low' if top_band < 3 else 'mid' if top_band < 7 else 'high'} frequency)")

    print(f"\nPlots saved to: {out_dir / 'plots'}")
    print(f"{'=' * 60}")

    df.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(channel_summary).to_csv(out_dir / "channel_summary.csv", index=False)

    pipe.cleanup()
    return out_dir
