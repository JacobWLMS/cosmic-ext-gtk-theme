"""Experiment 5: Naive Frequency Injection Test.

Question: If we just brute-force inject identity frequencies, does anything happen?
Only run if Experiments 1-2 show identifiable frequency bands.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from identity_analysis.frequency import (
    apply_frequency_mask,
    fft_2d_per_channel,
    frequency_band_mask,
    ifft_2d_per_channel,
)
from identity_analysis.pipeline import PipelineWrapper
from identity_analysis.plotting import plot_image_grid
from identity_analysis.scoring import ArcFaceScorer, compute_mse, compute_ssim
from identity_analysis.utils import (
    detect_face_bbox,
    get_output_dir,
    pixel_bbox_to_latent_bbox,
)


def _inject_frequencies(
    target_latent: np.ndarray,
    ref_latent: np.ndarray,
    band: str = "mid",
    face_bbox: tuple | None = None,
) -> np.ndarray:
    """Replace frequency bands in target with those from reference.

    If face_bbox is provided, only replace within the face region.
    """
    if target_latent.ndim == 4:
        target = target_latent[0].copy()
    else:
        target = target_latent.copy()
    if ref_latent.ndim == 4:
        ref = ref_latent[0]
    else:
        ref = ref_latent

    C, H, W = target.shape

    if face_bbox:
        y1, x1, y2, x2 = face_bbox
        target_region = target[:, y1:y2, x1:x2]
        ref_region = ref[:, y1:y2, x1:x2]
        rH, rW = y2 - y1, x2 - x1
    else:
        target_region = target
        ref_region = ref
        rH, rW = H, W

    # Determine frequency cutoffs
    if band == "low":
        cutoff_low, cutoff_high = 0.0, 0.33
    elif band == "mid":
        cutoff_low, cutoff_high = 0.33, 0.66
    elif band == "high":
        cutoff_low, cutoff_high = 0.66, 1.0
    elif band == "all":
        cutoff_low, cutoff_high = 0.0, 1.0
    else:
        cutoff_low, cutoff_high = 0.0, 1.0

    mask = frequency_band_mask(rH, rW, "custom", cutoff_low, cutoff_high)

    for c in range(C):
        # FFT both
        target_fft = np.fft.fftshift(np.fft.fft2(target_region[c]))
        ref_fft = np.fft.fftshift(np.fft.fft2(ref_region[c]))

        # Replace masked frequencies
        target_fft = target_fft * (1 - mask) + ref_fft * mask

        # IFFT back
        result = np.real(np.fft.ifft2(np.fft.ifftshift(target_fft)))

        if face_bbox:
            target[c, y1:y2, x1:x2] = result
        else:
            target[c] = result

    return target[np.newaxis]


def run(
    model_type: str = "sdxl",
    n_seeds: int = 50,
    save_latents: bool = True,
    output_base: str = "outputs",
    num_steps: int = 20,
):
    """Run Experiment 5: Naive Frequency Injection Test."""
    out_dir = get_output_dir("experiment_5", output_base)
    print(f"Experiment 5 output: {out_dir}")

    pipe = PipelineWrapper(model_type)
    scorer = ArcFaceScorer()

    seeds = list(range(min(n_seeds, 20)))
    n_steps = num_steps
    results_rows = []

    # Reference face
    ref_prompt = "portrait photo of a young woman with red hair, studio lighting, detailed face"
    print("Generating reference face...")
    ref_result = pipe.generate(ref_prompt, seed=99, save_step_latents=False)
    ref_image = ref_result["image"]
    ref_latent = ref_result["final_latent"]
    ref_image.save(out_dir / "plots" / "reference.png")

    ref_encoded = pipe.encode_image(ref_image)

    ref_face_bbox = detect_face_bbox(ref_image)
    ref_embedding = scorer.get_embedding(ref_image)
    if ref_embedding is None:
        print("WARNING: No face in reference image. Results may be unreliable.")

    # Target generation
    target_prompt = "portrait photo of an old man with a beard, outdoor lighting"

    bands_to_test = ["low", "mid", "high", "all"]

    for seed in tqdm(seeds, desc="Seeds"):
        # Generate target normally
        target_result = pipe.generate(
            target_prompt, seed=seed,
            num_inference_steps=n_steps,
            save_step_latents=True,
            output_dir=out_dir if save_latents else None,
        )
        target_image = target_result["image"]
        target_latent = target_result["final_latent"]
        target_image.save(out_dir / "plots" / f"target_seed{seed}.png")

        target_face_bbox = detect_face_bbox(target_image)
        target_latent_bbox = None
        if target_face_bbox:
            target_latent_bbox = pixel_bbox_to_latent_bbox(
                target_face_bbox, pipe.vae_scale_factor
            )

        # === Approach 1: Post-generation frequency replacement ===
        for band in bands_to_test:
            # Global replacement
            injected = _inject_frequencies(target_latent, ref_encoded, band=band)
            injected_image = pipe.decode_latent(injected)
            injected_image.save(
                out_dir / "plots" / f"injected_global_{band}_seed{seed}.png"
            )

            arcface_sim = None
            if ref_embedding is not None:
                inj_emb = scorer.get_embedding(injected_image)
                if inj_emb is not None:
                    arcface_sim = float(np.dot(ref_embedding, inj_emb))

            results_rows.append({
                "seed": seed,
                "approach": "post_generation",
                "band": band,
                "region": "global",
                "arcface_similarity": arcface_sim,
                "mse_vs_target": compute_mse(target_image, injected_image),
                "ssim_vs_target": compute_ssim(target_image, injected_image),
            })

            # Face-region replacement
            if target_latent_bbox:
                injected_face = _inject_frequencies(
                    target_latent, ref_encoded,
                    band=band, face_bbox=target_latent_bbox,
                )
                injected_face_image = pipe.decode_latent(injected_face)
                injected_face_image.save(
                    out_dir / "plots" / f"injected_face_{band}_seed{seed}.png"
                )

                arcface_sim_face = None
                if ref_embedding is not None:
                    inj_face_emb = scorer.get_embedding(injected_face_image)
                    if inj_face_emb is not None:
                        arcface_sim_face = float(np.dot(ref_embedding, inj_face_emb))

                results_rows.append({
                    "seed": seed,
                    "approach": "post_generation",
                    "band": band,
                    "region": "face",
                    "arcface_similarity": arcface_sim_face,
                    "mse_vs_target": compute_mse(target_image, injected_face_image),
                    "ssim_vs_target": compute_ssim(target_image, injected_face_image),
                })

                del injected_face, injected_face_image

            del injected, injected_image
            gc.collect()

        # === Approach 2: Initial noise modification ===
        # Noise the reference to match step 0 (pure noise level)
        if len(pipe.pipe.scheduler.timesteps) > 0:
            initial_timestep = pipe.pipe.scheduler.timesteps[0].item()
            noised_ref = pipe.add_noise_to_latent(ref_encoded, initial_timestep, seed=seed)

            for band in ["mid", "all"]:
                modified_noise = _inject_frequencies(
                    target_result["step_latents"][0] if target_result["step_latents"]
                    else target_latent,
                    noised_ref, band=band,
                )
                # Can't easily re-denoise from modified noise without hacking the pipeline
                # So just decode to see what the modified initial state looks like
                modified_image = pipe.decode_latent(modified_noise)
                modified_image.save(
                    out_dir / "plots" / f"noise_inject_{band}_seed{seed}.png"
                )

                results_rows.append({
                    "seed": seed,
                    "approach": "initial_noise",
                    "band": band,
                    "region": "global",
                    "arcface_similarity": None,  # Noise decode won't have faces
                    "mse_vs_target": None,
                    "ssim_vs_target": None,
                })

                del modified_noise, modified_image
                gc.collect()

        # === Approach 3: Mid-generation injection (step 25) ===
        mid_step = n_steps // 2
        if mid_step < len(target_result["step_latents"]):
            mid_latent = target_result["step_latents"][mid_step]
            timestep = pipe.pipe.scheduler.timesteps[mid_step].item()
            noised_ref_mid = pipe.add_noise_to_latent(ref_encoded, timestep, seed=seed)

            for band in ["mid", "all"]:
                injected_mid = _inject_frequencies(
                    mid_latent, noised_ref_mid, band=band,
                )
                injected_mid_image = pipe.decode_latent(injected_mid)
                injected_mid_image.save(
                    out_dir / "plots" / f"mid_inject_{band}_seed{seed}.png"
                )

                arcface_sim_mid = None
                if ref_embedding is not None:
                    mid_emb = scorer.get_embedding(injected_mid_image)
                    if mid_emb is not None:
                        arcface_sim_mid = float(np.dot(ref_embedding, mid_emb))

                results_rows.append({
                    "seed": seed,
                    "approach": "mid_generation",
                    "band": band,
                    "region": "global",
                    "arcface_similarity": arcface_sim_mid,
                    "mse_vs_target": compute_mse(target_image, injected_mid_image),
                    "ssim_vs_target": compute_ssim(target_image, injected_mid_image),
                })

                del injected_mid, injected_mid_image
                gc.collect()

        del target_result, target_image, target_latent
        gc.collect()

    # Save results
    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # Summary plot: ArcFace similarity by approach and band
    if not df.empty and df["arcface_similarity"].notna().any():
        summary = df[df["arcface_similarity"].notna()].groupby(
            ["approach", "band", "region"]
        )["arcface_similarity"].agg(["mean", "std"]).reset_index()
        summary.to_csv(out_dir / "summary.csv", index=False)

        # Plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        approaches = summary["approach"].unique()
        x = np.arange(len(summary))
        labels = [
            f"{r['approach']}\n{r['band']}\n{r['region']}"
            for _, r in summary.iterrows()
        ]
        ax.bar(x, summary["mean"], yerr=summary["std"], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("ArcFace Similarity to Reference")
        ax.set_title(f"Frequency Injection Results ({model_type.upper()})")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "injection_results_summary.png", dpi=150)
        plt.close()

    print(f"\nExperiment 5 complete. Results saved to {out_dir}")
    pipe.cleanup()
    return out_dir
