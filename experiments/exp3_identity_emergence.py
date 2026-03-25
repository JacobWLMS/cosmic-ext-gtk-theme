"""Experiment 3: Step-by-Step Identity Emergence.

Question: At what denoising step does identity information appear in the latent?

Key focus (informed by Exp 1+2):
- Track Ch 2 (identity magnitude) and Ch 3 (identity fingerprint) separately
- Measure per-channel divergence between two different identities at each step
- Compare with ArcFace similarity curve to cross-reference latent vs perceptual identity
- Use multiple seeds to get variance estimates
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
from identity_analysis.plotting import plot_line_chart
from identity_analysis.scoring import ArcFaceScorer
from identity_analysis.utils import get_output_dir


# Two identity pairs — one large gap, one subtle
IDENTITY_PAIRS = [
    {
        "name": "age_gap",
        "prompt_a": "portrait photo of a young woman, detailed face, studio lighting, neutral background",
        "prompt_b": "portrait photo of an elderly woman, detailed face, studio lighting, neutral background",
    },
    {
        "name": "gender",
        "prompt_a": "portrait photo of a man, detailed face, studio lighting, neutral background",
        "prompt_b": "portrait photo of a woman, detailed face, studio lighting, neutral background",
    },
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

    seeds = list(range(min(n_seeds, 10)))  # Cap at 10 — each seed captures all steps
    results_rows = []

    for pair in IDENTITY_PAIRS:
        pair_name = pair["name"]
        prompt_a = pair["prompt_a"]
        prompt_b = pair["prompt_b"]
        print(f"\n{'='*60}")
        print(f"Pair: {pair_name}")
        print(f"  A: {prompt_a}")
        print(f"  B: {prompt_b}")

        # Accumulate per-step metrics across seeds
        all_step_ch_diffs = []       # [n_seeds, n_steps, 4] per-channel mean abs diff
        all_step_freq_diffs = []     # [n_seeds, n_steps, 4] per-channel freq mag diff
        all_step_arcface_a = []      # [n_seeds, n_steps] arcface sim to final A
        all_step_arcface_b = []      # [n_seeds, n_steps] arcface sim to final B

        for seed in tqdm(seeds, desc=f"  Seeds"):
            # Generate both with step latent capture
            res_a = pipe.generate(
                prompt_a, seed,
                num_inference_steps=num_steps,
                save_step_latents=True,
                output_dir=None,
            )
            res_b = pipe.generate(
                prompt_b, seed,
                num_inference_steps=num_steps,
                save_step_latents=True,
                output_dir=None,
            )

            # Reference: ArcFace embeddings of final images
            ref_emb_a = scorer.get_embedding(res_a["image"])
            ref_emb_b = scorer.get_embedding(res_b["image"])

            # Save sample images for first seed
            if seed == 0:
                res_a["image"].save(out_dir / "plots" / f"{pair_name}_final_a.png")
                res_b["image"].save(out_dir / "plots" / f"{pair_name}_final_b.png")

            seed_ch_diffs = []
            seed_freq_diffs = []
            seed_arcface_a = []
            seed_arcface_b = []

            for step_idx in range(num_steps):
                lat_a = res_a["step_latents"][step_idx]  # [1, C, H, W]
                lat_b = res_b["step_latents"][step_idx]

                # Per-channel mean absolute difference in latent space
                diff = lat_a - lat_b
                ch_diffs = np.mean(np.abs(diff[0]), axis=(1, 2))  # [C]
                seed_ch_diffs.append(ch_diffs)

                # Per-channel frequency magnitude difference
                mag_a, _ = fft_2d_per_channel(lat_a)
                mag_b, _ = fft_2d_per_channel(lat_b)
                freq_diff = np.mean(np.abs(mag_a - mag_b), axis=(1, 2))  # [C]
                seed_freq_diffs.append(freq_diff)

                # ArcFace at key steps only (decoding is expensive)
                if step_idx % 5 == 0 or step_idx == num_steps - 1:
                    try:
                        img_a = pipe.decode_latent(lat_a)
                        emb_a = scorer.get_embedding(img_a)
                        sim_a = float(np.dot(ref_emb_a, emb_a)) if (ref_emb_a is not None and emb_a is not None) else None
                        del img_a
                    except Exception:
                        sim_a = None

                    try:
                        img_b = pipe.decode_latent(lat_b)
                        emb_b = scorer.get_embedding(img_b)
                        sim_b = float(np.dot(ref_emb_b, emb_b)) if (ref_emb_b is not None and emb_b is not None) else None
                        del img_b
                    except Exception:
                        sim_b = None
                else:
                    sim_a = None
                    sim_b = None

                seed_arcface_a.append(sim_a)
                seed_arcface_b.append(sim_b)

                # Record per-step data
                for c in range(len(ch_diffs)):
                    results_rows.append({
                        "pair": pair_name,
                        "seed": seed,
                        "step": step_idx,
                        "channel": c,
                        "spatial_diff": float(ch_diffs[c]),
                        "freq_diff": float(freq_diff[c]),
                        "arcface_sim_a": sim_a,
                        "arcface_sim_b": sim_b,
                    })

            all_step_ch_diffs.append(np.array(seed_ch_diffs))      # [n_steps, C]
            all_step_freq_diffs.append(np.array(seed_freq_diffs))
            all_step_arcface_a.append(seed_arcface_a)
            all_step_arcface_b.append(seed_arcface_b)

            del res_a, res_b
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate across seeds
        mean_ch_diffs = np.mean(all_step_ch_diffs, axis=0)    # [n_steps, C]
        std_ch_diffs = np.std(all_step_ch_diffs, axis=0)
        mean_freq_diffs = np.mean(all_step_freq_diffs, axis=0)
        std_freq_diffs = np.std(all_step_freq_diffs, axis=0)

        steps_arr = np.arange(num_steps)
        n_channels = mean_ch_diffs.shape[1]

        # PLOT 1: Per-channel spatial divergence over steps (the key plot)
        plot_line_chart(
            steps_arr,
            {f"Ch {c}": mean_ch_diffs[:, c] for c in range(n_channels)},
            out_dir / "plots" / f"{pair_name}_spatial_divergence_per_channel.png",
            xlabel="Denoising Step",
            ylabel="Mean |latent_A - latent_B| per channel",
            title=f"Identity Divergence by Channel Over Steps: {pair_name}",
            yerr_dict={f"Ch {c}": std_ch_diffs[:, c] for c in range(n_channels)},
        )

        # PLOT 2: Per-channel frequency divergence over steps
        plot_line_chart(
            steps_arr,
            {f"Ch {c}": mean_freq_diffs[:, c] for c in range(n_channels)},
            out_dir / "plots" / f"{pair_name}_freq_divergence_per_channel.png",
            xlabel="Denoising Step",
            ylabel="Mean |FFT(A) - FFT(B)| per channel",
            title=f"Frequency Identity Divergence by Channel: {pair_name}",
            yerr_dict={f"Ch {c}": std_freq_diffs[:, c] for c in range(n_channels)},
        )

        # PLOT 3: Ch2 vs Ch3 head-to-head (the Exp1 vs Exp2 comparison)
        plot_line_chart(
            steps_arr,
            {
                "Ch 2 (magnitude)": mean_ch_diffs[:, 2],
                "Ch 3 (fingerprint)": mean_ch_diffs[:, 3],
                "Ch 0 (scene)": mean_ch_diffs[:, 0],
            },
            out_dir / "plots" / f"{pair_name}_ch2_vs_ch3_emergence.png",
            xlabel="Denoising Step",
            ylabel="Mean |latent_A - latent_B|",
            title=f"Ch2 (Magnitude) vs Ch3 (Fingerprint) Emergence: {pair_name}",
        )

        # PLOT 4: Normalized divergence (each channel relative to its own final value)
        final_ch_diffs = mean_ch_diffs[-1]
        normalized = mean_ch_diffs / (final_ch_diffs[np.newaxis, :] + 1e-10)
        plot_line_chart(
            steps_arr,
            {f"Ch {c}": normalized[:, c] for c in range(n_channels)},
            out_dir / "plots" / f"{pair_name}_normalized_divergence.png",
            xlabel="Denoising Step",
            ylabel="Fraction of Final Divergence",
            title=f"Normalized Identity Emergence (when does each channel reach final state?): {pair_name}",
        )

        # PLOT 5: ArcFace emergence (sampled steps)
        arcface_a_mean = []
        arcface_steps = []
        for step_idx in range(num_steps):
            vals = [all_step_arcface_a[s][step_idx] for s in range(len(seeds))
                    if all_step_arcface_a[s][step_idx] is not None]
            if vals:
                arcface_a_mean.append(np.mean(vals))
                arcface_steps.append(step_idx)

        if arcface_a_mean:
            plot_line_chart(
                np.array(arcface_steps),
                {"ArcFace similarity to final": np.array(arcface_a_mean)},
                out_dir / "plots" / f"{pair_name}_arcface_emergence.png",
                xlabel="Denoising Step",
                ylabel="ArcFace Cosine Similarity to Final Image",
                title=f"Perceptual Identity Emergence: {pair_name}",
            )

        # Print summary
        print(f"\n  Step where each channel reaches 50% of final divergence:")
        for c in range(n_channels):
            half_val = final_ch_diffs[c] * 0.5
            half_step = np.argmax(mean_ch_diffs[:, c] >= half_val)
            print(f"    Ch {c}: step {half_step}/{num_steps}")

        print(f"  Step where each channel reaches 90% of final divergence:")
        for c in range(n_channels):
            nine_val = final_ch_diffs[c] * 0.9
            nine_step = np.argmax(mean_ch_diffs[:, c] >= nine_val)
            print(f"    Ch {c}: step {nine_step}/{num_steps}")

    # Save all results
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # Summary CSV: per-step, per-channel means across seeds
    summary_rows = []
    for pair in IDENTITY_PAIRS:
        pair_df = df[df["pair"] == pair["name"]]
        for step in range(num_steps):
            step_df = pair_df[pair_df["step"] == step]
            for c in range(4):
                ch_df = step_df[step_df["channel"] == c]
                summary_rows.append({
                    "pair": pair["name"],
                    "step": step,
                    "channel": c,
                    "spatial_diff_mean": ch_df["spatial_diff"].mean(),
                    "spatial_diff_std": ch_df["spatial_diff"].std(),
                    "freq_diff_mean": ch_df["freq_diff"].mean(),
                    "freq_diff_std": ch_df["freq_diff"].std(),
                })
    pd.DataFrame(summary_rows).to_csv(out_dir / "step_summary.csv", index=False)

    print(f"\nExperiment 3 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
