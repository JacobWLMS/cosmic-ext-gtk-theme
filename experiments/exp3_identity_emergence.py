"""Experiment 3: Step-by-Step Identity Emergence.

Question: At what denoising step does identity information appear in the latent?
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
)
from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_image_grid, plot_line_chart
from identity_analysis.scoring import ArcFaceScorer
from identity_analysis.utils import get_output_dir


CELEBRITY_PROMPTS = [
    ("Brad Pitt", "portrait photo of Brad Pitt, detailed face, studio lighting"),
    ("Taylor Swift", "portrait photo of Taylor Swift, detailed face, studio lighting"),
    ("Morgan Freeman", "portrait photo of Morgan Freeman, detailed face, studio lighting"),
    ("Scarlett Johansson", "portrait photo of Scarlett Johansson, detailed face, studio lighting"),
    ("Keanu Reeves", "portrait photo of Keanu Reeves, detailed face, studio lighting"),
]


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 3: Step-by-Step Identity Emergence."""
    import torch; torch.cuda.empty_cache(); gc.collect()
    out_dir = get_output_dir("experiment_3", output_base)
    print(f"Experiment 3 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    scorer = ArcFaceScorer()

    seeds = list(range(min(n_seeds, 20)))  # Cap at 20 for this experiment (many decode steps)
    n_steps = num_steps
    results_rows = []

    # For each celebrity, generate images and track identity emergence
    for celeb_name, prompt in CELEBRITY_PROMPTS:
        print(f"\nProcessing: {celeb_name}")

        # Generate a "reference" image with seed 0 (best quality seed)
        ref_result = pipe.generate(
            prompt, seed=0,
            num_inference_steps=n_steps,
            save_step_latents=True,
            output_dir=out_dir,
        )
        ref_image = ref_result["image"]
        ref_image.save(out_dir / "plots" / f"{celeb_name.replace(' ', '_')}_reference.png")

        # ArcFace embedding of the reference (final) image
        ref_embedding = scorer.get_embedding(ref_image)
        if ref_embedding is None:
            print(f"  WARNING: No face detected in reference for {celeb_name}, skipping.")
            continue

        # Decode each denoising step and compute ArcFace similarity
        step_similarities = []
        step_freq_energies = []

        for step_idx in tqdm(range(n_steps), desc=f"  Decoding steps"):
            step_latent = ref_result["step_latents"][step_idx]

            # Decode the intermediate latent
            try:
                step_image = pipe.decode_latent(step_latent)
            except Exception:
                step_similarities.append(None)
                step_freq_energies.append(None)
                continue

            # ArcFace similarity
            step_embedding = scorer.get_embedding(step_image)
            if step_embedding is not None:
                sim = float(np.dot(ref_embedding, step_embedding))
            else:
                sim = None
            step_similarities.append(sim)

            # Frequency band energy of the latent at this step
            energy = compute_frequency_band_energy(step_latent)
            step_freq_energies.append(energy)

            # Save some intermediate images for visual inspection
            if step_idx % 10 == 0 or step_idx == n_steps - 1:
                step_image.save(
                    out_dir / "plots" / f"{celeb_name.replace(' ', '_')}_step{step_idx:03d}.png"
                )

            del step_image
            gc.collect()

            results_rows.append({
                "celebrity": celeb_name,
                "seed": 0,
                "step": step_idx,
                "arcface_similarity": sim,
            })

        # Plot ArcFace similarity vs step
        valid_steps = [i for i, s in enumerate(step_similarities) if s is not None]
        valid_sims = [step_similarities[i] for i in valid_steps]

        if valid_sims:
            plot_line_chart(
                np.array(valid_steps),
                {celeb_name: np.array(valid_sims)},
                out_dir / "plots" / f"arcface_vs_step_{celeb_name.replace(' ', '_')}.png",
                xlabel="Denoising Step",
                ylabel="ArcFace Cosine Similarity",
                title=f"Identity Emergence: {celeb_name} ({model_type.upper()})",
            )

        # Plot frequency band energy evolution
        valid_energies = [
            (i, e) for i, e in enumerate(step_freq_energies) if e is not None
        ]
        if valid_energies:
            n_channels = valid_energies[0][1].shape[0]
            n_bands = valid_energies[0][1].shape[1]
            steps_arr = np.array([v[0] for v in valid_energies])

            # Plot total energy per channel over steps
            channel_total_energy = {}
            for c in range(n_channels):
                energies = np.array([v[1][c].sum() for v in valid_energies])
                channel_total_energy[f"Ch {c}"] = energies

            plot_line_chart(
                steps_arr,
                channel_total_energy,
                out_dir / "plots" / f"freq_energy_vs_step_{celeb_name.replace(' ', '_')}.png",
                xlabel="Denoising Step",
                ylabel="Total Frequency Energy",
                title=f"Frequency Energy Evolution: {celeb_name} ({model_type.upper()})",
            )

        del ref_result
        gc.collect()

    # Combined plot: all celebrities on one chart
    df = pd.DataFrame(results_rows)
    if not df.empty and "arcface_similarity" in df.columns:
        combined_data = {}
        for celeb_name, _ in CELEBRITY_PROMPTS:
            celeb_df = df[(df["celebrity"] == celeb_name) & df["arcface_similarity"].notna()]
            if not celeb_df.empty:
                combined_data[celeb_name] = celeb_df["arcface_similarity"].values

        if combined_data:
            max_len = max(len(v) for v in combined_data.values())
            x = np.arange(max_len)
            plot_line_chart(
                x,
                combined_data,
                out_dir / "plots" / "arcface_vs_step_all_celebs.png",
                xlabel="Denoising Step",
                ylabel="ArcFace Cosine Similarity",
                title=f"Identity Emergence: All Celebrities ({model_type.upper()})",
            )

    df.to_csv(out_dir / "results.csv", index=False)
    print(f"\nExperiment 3 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
