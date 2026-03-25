"""Experiment 4: Timestep-Matched Reference Correlation.

Question: Can we meaningfully correlate a reference face latent with a mid-denoising latent?
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from identity_analysis.frequency import (
    compute_frequency_band_energy,
    fft_2d_per_channel,
    frequency_band_mask,
    spatial_windowed_fft,
)
from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_line_chart
from identity_analysis.utils import (
    detect_face_bbox,
    get_output_dir,
    pixel_bbox_to_latent_bbox,
)


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 4: Timestep-Matched Reference Correlation."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_4", output_base)
    print(f"Experiment 4 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    n_steps = num_steps
    check_steps = [10, 20, 30, 40]
    n_bands = 10

    seeds = list(range(min(n_seeds, 20)))
    results_rows = []

    # Reference face: generate a clear portrait
    ref_prompt = "portrait photo of a young woman, studio lighting, detailed face"
    target_prompt = "portrait photo of an old man reading a book, library background"

    print("Generating reference face...")
    ref_result = pipe.generate(ref_prompt, seed=42, save_step_latents=False)
    ref_image = ref_result["image"]
    ref_latent = ref_result["final_latent"]
    ref_image.save(out_dir / "plots" / "reference_face.png")

    # Detect face in reference for spatial analysis
    ref_face_bbox = detect_face_bbox(ref_image)
    ref_latent_bbox = None
    if ref_face_bbox:
        ref_latent_bbox = pixel_bbox_to_latent_bbox(
            ref_face_bbox, pipe.vae_scale_factor
        )
        print(f"  Reference face bbox (pixel): {ref_face_bbox}")
        print(f"  Reference face bbox (latent): {ref_latent_bbox}")

    # Encode reference through VAE
    ref_encoded = pipe.encode_image(ref_image)

    for seed in tqdm(seeds, desc="Seeds"):
        print(f"\n  Generating target image (seed {seed})...")
        target_result = pipe.generate(
            target_prompt, seed=seed,
            num_inference_steps=n_steps,
            save_step_latents=True,
            output_dir=out_dir if save_latents else None,
        )
        target_image = target_result["image"]
        target_image.save(out_dir / "plots" / f"target_seed{seed}.png")

        # Detect face in target
        target_face_bbox = detect_face_bbox(target_image)
        target_latent_bbox = None
        if target_face_bbox:
            target_latent_bbox = pixel_bbox_to_latent_bbox(
                target_face_bbox, pipe.vae_scale_factor
            )

        for step_idx in check_steps:
            if step_idx >= len(target_result["step_latents"]):
                continue

            gen_latent = target_result["step_latents"][step_idx]

            # Noise the reference latent to match this timestep
            timestep = pipe.pipe.scheduler.timesteps[step_idx].item()
            noised_ref = pipe.add_noise_to_latent(ref_encoded, timestep, seed=seed)

            # Global FFT correlation
            ref_mag, _ = fft_2d_per_channel(noised_ref)
            gen_mag, _ = fft_2d_per_channel(gen_latent)

            # Per-frequency-band correlation
            C, H, W = ref_mag.shape
            for band_idx in range(n_bands):
                low = band_idx / n_bands
                high = (band_idx + 1) / n_bands
                mask = frequency_band_mask(H, W, "custom", low, high)

                for c in range(C):
                    ref_band = ref_mag[c] * mask
                    gen_band = gen_mag[c] * mask

                    ref_flat = ref_band.flatten()
                    gen_flat = gen_band.flatten()

                    if np.std(ref_flat) > 0 and np.std(gen_flat) > 0:
                        corr = float(np.corrcoef(ref_flat, gen_flat)[0, 1])
                    else:
                        corr = 0.0

                    results_rows.append({
                        "seed": seed,
                        "step": step_idx,
                        "band": band_idx,
                        "channel": c,
                        "correlation": corr,
                        "region": "global",
                    })

            # Spatial (face-region) FFT correlation
            if ref_latent_bbox and target_latent_bbox:
                try:
                    ref_face_mag, _ = spatial_windowed_fft(noised_ref, ref_latent_bbox)
                    gen_face_mag, _ = spatial_windowed_fft(gen_latent, target_latent_bbox)

                    # Resize to match if needed
                    min_h = min(ref_face_mag.shape[1], gen_face_mag.shape[1])
                    min_w = min(ref_face_mag.shape[2], gen_face_mag.shape[2])
                    ref_face_crop = ref_face_mag[:, :min_h, :min_w]
                    gen_face_crop = gen_face_mag[:, :min_h, :min_w]

                    for c in range(C):
                        ref_flat = ref_face_crop[c].flatten()
                        gen_flat = gen_face_crop[c].flatten()
                        if np.std(ref_flat) > 0 and np.std(gen_flat) > 0:
                            corr = float(np.corrcoef(ref_flat, gen_flat)[0, 1])
                        else:
                            corr = 0.0
                        results_rows.append({
                            "seed": seed,
                            "step": step_idx,
                            "band": -1,  # full spectrum in face region
                            "channel": c,
                            "correlation": corr,
                            "region": "face",
                        })
                except Exception:
                    pass

            del gen_latent, noised_ref
            gc.collect()

        del target_result
        gc.collect()

    # Analysis and plotting
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # Plot: correlation vs frequency band, per step (global)
    global_df = df[df["region"] == "global"]
    if not global_df.empty:
        for step in check_steps:
            step_df = global_df[global_df["step"] == step]
            if step_df.empty:
                continue

            avg_corr_by_band = step_df.groupby("band")["correlation"].mean().values
            n_channels = step_df["channel"].nunique()

            # Per-channel
            channel_data = {}
            for c in range(n_channels):
                ch_df = step_df[step_df["channel"] == c]
                channel_data[f"Ch {c}"] = ch_df.groupby("band")["correlation"].mean().values

            plot_line_chart(
                np.arange(n_bands),
                channel_data,
                out_dir / "plots" / f"correlation_vs_band_step{step}.png",
                xlabel="Frequency Band (low → high)",
                ylabel="Pearson Correlation",
                title=f"Ref-Gen Correlation at Step {step} ({model_type.upper()})",
            )

    # Plot: correlation vs step (averaged over bands)
    if not global_df.empty:
        step_data = {}
        n_channels = global_df["channel"].nunique()
        for c in range(n_channels):
            ch_df = global_df[global_df["channel"] == c]
            step_data[f"Ch {c}"] = ch_df.groupby("step")["correlation"].mean().values

        plot_line_chart(
            np.array(check_steps[:len(list(step_data.values())[0])]),
            step_data,
            out_dir / "plots" / "correlation_vs_step.png",
            xlabel="Denoising Step",
            ylabel="Mean Correlation (all bands)",
            title=f"Ref-Gen Correlation vs Step ({model_type.upper()})",
        )

    print(f"\nExperiment 4 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
